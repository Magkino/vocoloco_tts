/**
 * TTS Web Worker (ES Module) — runs OmniVoice inference via ONNX Runtime Web.
 * Uses @huggingface/transformers for proper Qwen2 BPE tokenization.
 */

// onnxruntime-web/webgpu: the native WebGPU EP (the default/`all` bundles use
// the deprecated JSEP backend); its WASM EP serves the CPU fallback too
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';
import { AutoTokenizer, env as tfEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
import { estimateTargetTokens } from '../duration-estimator.js';
import { addEndPunctuation } from '../sentence-buffer.js?v=2'; // ?v: a stale cached copy lacks this export
import { trimAndFade, peakAbs } from '../audio-postprocess.js';
import { timeStretch } from '../time-stretch.js';
import { GpuPostProcessor } from './gpu-postprocess.js?v=2';
import { unmaskSchedule } from './unmask-schedule.js';
import { inspectFile, loadFile } from './model-files.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

// Maximize performance — multi-threading requires cross-origin isolation (COOP/COEP headers)
ort.env.wasm.numThreads = self.crossOriginIsolated ? (navigator.hardwareConcurrency || 4) : 1;
ort.env.wasm.simd = true;

// Configure transformers.js to load tokenizer from our server
tfEnv.allowLocalModels = false;

let mainSession = null;
let decoderSession = null;
let encoderSession = null;
let tokenizer = null;
let config = null;
let gpuPostProc = null;
let modelBase = null;
let sessionEps = ['wasm'];
let onGpu = false;

// ─── Cancellation ───────────────────────────────────────────────────────────
// Job-scoped: each synthesize message snapshots the cancel counter when it
// ARRIVES (not when it starts), so a cancel posted while the job sits behind
// a queued encode still applies to it.

let cancelCounter = 0;
let activeCancelBaseline = 0;
let activeJobId = null;

function isCancelRequested() { return cancelCounter > activeCancelBaseline; }

class CancelledError extends Error {}

// Force a macrotask turn so an incoming 'cancel' message can be delivered
// between diffusion steps even on the WASM backend (whose session.run promise
// may resolve without yielding to the event loop).
const _yieldChannel = new MessageChannel();
let _yieldResolve = null;
_yieldChannel.port1.onmessage = () => { const r = _yieldResolve; _yieldResolve = null; if (r) r(); };
function yieldMacrotask() {
  return new Promise(res => { _yieldResolve = res; _yieldChannel.port2.postMessage(0); });
}

// ─── Model files ────────────────────────────────────────────────────────────

const CACHE_NAME = 'omnivoice-models-v1';

// ─── Tensor helper ──────────────────────────────────────────────────────────

function T_(type, data, dims) { return new ort.Tensor(type, data, dims); }

// ─── Seeded PRNG (mulberry32) for deterministic generation ──────────────────

function mulberry32(seed) {
  let s = seed | 0;
  return function() {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rng = Math.random; // default, overridden per generation

// Pre-allocated buffers for hot-path computation (avoids GC pressure)
const _cLP = new Float32Array(1025);
const _uLP = new Float32Array(1025);
const _g = new Float32Array(1025);

function logSoftmaxInto(arr, offset, len, out) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) { const v = arr[offset + i]; if (v > max) max = v; }
  let sum = 0;
  for (let i = 0; i < len; i++) sum += Math.exp(arr[offset + i] - max);
  const lse = max + Math.log(sum);
  for (let i = 0; i < len; i++) out[i] = arr[offset + i] - lse;
}

// Same math and logits layout as the WGSL shader in gpu-postprocess.js (cond
// and uncond may be the same array, see makeBatchedPasses).
function cpuPostProcess(cond, uncond, { C, T: numTargetTokens, V, maskId, condStride, targetOff, uncondStride, uncondOff = 0, guidanceScale, layerPenalty }, pred, scores) {
  const gScale1 = 1 + guidanceScale;
  for (let c = 0; c < C; c++) {
    const layerScore = layerPenalty * c;
    for (let t = 0; t < numTargetTokens; t++) {
      const cOff = (c * condStride + targetOff + t) * V;
      const uOff = uncondOff + (c * uncondStride + t) * V;
      logSoftmaxInto(cond, cOff, V, _cLP);
      logSoftmaxInto(uncond, uOff, V, _uLP);
      let mx = -Infinity;
      for (let v = 0; v < V; v++) {
        const gv = gScale1 * _cLP[v] - guidanceScale * _uLP[v];
        _g[v] = gv;
        if (gv > mx) mx = gv;
      }
      let sm = 0;
      for (let v = 0; v < V; v++) sm += Math.exp(_g[v] - mx);
      const lse = mx + Math.log(sm);
      let bestV = 0, bestS = -Infinity;
      for (let v = 0; v < V; v++) {
        if (v === maskId) continue;
        const lp = _g[v] - lse;
        if (lp > bestS) { bestS = lp; bestV = v; }
      }
      const idx = c * numTargetTokens + t;
      pred[idx] = bestV;
      scores[idx] = bestS - layerScore;
    }
  }
}

// ─── Prepare inference inputs ───────────────────────────────────────────────

async function prepareInferenceInputs(text, numTargetTokens, tok, cfg, opts = {}) {
  const { refText = null, refAudioTokens = null, lang = null, instruct = null, denoise = true } = opts;
  const C = cfg.num_audio_codebook;
  const maskId = cfg.audio_mask_id;

  // Build style string
  let styleText = '';
  if (denoise) styleText += '<|denoise|>';
  styleText += `<|lang_start|>${lang || 'None'}<|lang_end|>`;
  styleText += `<|instruct_start|>${instruct || 'None'}<|instruct_end|>`;

  // Build text string
  let fullText = refText ? addEndPunctuation(refText) + ' ' + text.trim() : text.trim();
  fullText = fullText.replace(/[\r\n]+/g, '').replace(/[ \t]+/g, ' ');
  const wrappedText = `<|text_start|>${fullText}<|text_end|>`;

  // Tokenize using transformers.js (proper Qwen2 BPE)
  const styleEncoded = await tok(styleText, { add_special_tokens: false });
  const textEncoded = await tok(wrappedText, { add_special_tokens: false });
  // transformers.js returns Tensors — extract as plain number arrays
  const styleIds = Array.from(styleEncoded.input_ids.data, Number);
  const textIds = Array.from(textEncoded.input_ids.data, Number);

  // Sequence layout: [style | text | ref_audio? | target_masked]
  const refLen = refAudioTokens ? refAudioTokens[0].length : 0;
  const totalLen = styleIds.length + textIds.length + refLen + numTargetTokens;

  const inputIds = new BigInt64Array(C * totalLen);

  // Style tokens (replicated across codebooks)
  for (let c = 0; c < C; c++)
    for (let i = 0; i < styleIds.length; i++)
      inputIds[c * totalLen + i] = BigInt(styleIds[i]);

  // Text tokens
  const textOff = styleIds.length;
  for (let c = 0; c < C; c++)
    for (let i = 0; i < textIds.length; i++)
      inputIds[c * totalLen + textOff + i] = BigInt(textIds[i]);

  // Reference audio tokens
  const refOff = textOff + textIds.length;
  if (refAudioTokens) {
    for (let c = 0; c < C; c++)
      for (let t = 0; t < refLen; t++)
        inputIds[c * totalLen + refOff + t] = BigInt(refAudioTokens[c][t]);
  }

  // Target = all mask
  const targetOff = refOff + refLen;
  for (let c = 0; c < C; c++)
    for (let t = 0; t < numTargetTokens; t++)
      inputIds[c * totalLen + targetOff + t] = BigInt(maskId);

  // Audio mask: true for audio positions (ref + target)
  const audioMask = new Uint8Array(totalLen);
  const audioStart = refAudioTokens ? refOff : targetOff;
  for (let i = audioStart; i < totalLen; i++) audioMask[i] = 1;

  return { inputIds, audioMask, totalLen, numTargetTokens, targetOff, C };
}

// ─── Top-k unmask using partial selection ───────────────────────────────────

function topKUnmask(scores, pred, tokens, n, k) {
  // Find k-th largest score using nth_element-style partition
  // For small k (typically 2-300), a simple selection is fast enough
  const indices = new Int32Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (scores[i] > -Infinity) indices[count++] = i;
  }
  // Partial sort: only find top k
  for (let i = 0; i < Math.min(k, count); i++) {
    let maxIdx = i;
    for (let j = i + 1; j < count; j++) {
      if (scores[indices[j]] > scores[indices[maxIdx]]) maxIdx = j;
    }
    if (maxIdx !== i) { const tmp = indices[i]; indices[i] = indices[maxIdx]; indices[maxIdx] = tmp; }
    tokens[indices[i]] = BigInt(pred[indices[i]]);
  }
}

// ─── Iterative unmasking generation loop ────────────────────────────────────

// Classifier-free guidance needs a conditional pass (full sequence) and an
// unconditional one (target only) per step. Two layouts:
//  - batched: one (2, C, L) run, the unconditional half padded to L
//  - split: two runs, nothing padded
// On WebGPU each run costs ~30 ms of fixed overhead (measured on an M-series
// Mac), so batched is faster until the padding outweighs it — long cloned-voice
// references. On the CPU there is no such overhead: always split.
const PASSES = new URL(self.location.href).searchParams.get('passes'); // test override: 'batched' | 'split'
const SPLIT_MIN_PAD = 200; // padded positions above which split runs win on WebGPU

function makeBatchedPasses({ inputIds, audioMask, totalLen: L, numTargetTokens: T, targetOff, C }, maskId, V) {
  const ids = new BigInt64Array(2 * C * L).fill(BigInt(maskId));
  for (let c = 0; c < C; c++) {
    ids.set(inputIds.subarray(c * L, (c + 1) * L), c * L);
    for (let t = 0; t < T; t++) ids[(C + c) * L + t] = inputIds[c * L + targetOff + t];
  }
  const mask = new Uint8Array(2 * L);
  mask.set(audioMask);
  mask.fill(1, L, L + T);
  const attn = new Uint8Array(2 * L * L);
  attn.fill(1, 0, L * L);
  for (let q = 0; q < T; q++) attn.fill(1, L * L + q * L, L * L + q * L + T);
  for (let p = T; p < L; p++) attn[L * L + p * L + p] = 1; // padding attends to itself only
  return {
    name: 'batched',
    feeds: [{
      input_ids: T_('int64', ids, [2, C, L]),
      audio_mask: T_('bool', mask, [2, L]),
      attention_mask: T_('bool', attn, [2, 1, L, L]),
    }],
    layout: { condStride: L, targetOff, uncondStride: L, uncondOff: C * L * V },
    setTarget(c, t, v) { ids[c * L + targetOff + t] = v; ids[(C + c) * L + t] = v; },
  };
}

function makeSplitPasses({ inputIds, audioMask, totalLen: L, numTargetTokens: T, targetOff, C }, maskId) {
  const condIds = inputIds.slice();
  const uncondIds = new BigInt64Array(C * T).fill(BigInt(maskId));
  return {
    name: 'split',
    feeds: [{
      input_ids: T_('int64', condIds, [1, C, L]),
      audio_mask: T_('bool', audioMask, [1, L]),
      attention_mask: T_('bool', new Uint8Array(L * L).fill(1), [1, 1, L, L]),
    }, {
      input_ids: T_('int64', uncondIds, [1, C, T]),
      audio_mask: T_('bool', new Uint8Array(T).fill(1), [1, T]),
      attention_mask: T_('bool', new Uint8Array(T * T).fill(1), [1, 1, T, T]),
    }],
    layout: { condStride: L, targetOff, uncondStride: T, uncondOff: 0 },
    setTarget(c, t, v) { condIds[c * L + targetOff + t] = v; uncondIds[c * T + t] = v; },
  };
}

// pred / scores for the current step: GPU post-processing straight from ORT's
// output buffers when possible, otherwise on the CPU.
async function postProcessStep(condOut, uncondOut, params, pred, scores) {
  if (gpuPostProc && condOut.location === 'gpu-buffer' && uncondOut.location === 'gpu-buffer') {
    try {
      await gpuPostProc.run(condOut.gpuBuffer, uncondOut.gpuBuffer,
        { ...params, condElems: condOut.size, uncondElems: uncondOut.size }, pred, scores);
      return 'GPU-PP';
    } catch (e) {
      console.warn('[gpu-postprocess] failed, falling back to CPU:', e.message);
      gpuPostProc.destroy();
      gpuPostProc = null;
    }
  }
  const cond = condOut.location === 'cpu' ? condOut.data : await condOut.getData();
  const uncond = uncondOut === condOut ? cond : (uncondOut.location === 'cpu' ? uncondOut.data : await uncondOut.getData());
  cpuPostProcess(cond, uncond, params, pred, scores);
  return 'CPU-PP';
}

async function generateIterative(inp, cfg, numStep, guidanceScale, tShift, layerPenalty = 5.0, posTemp = 5.0) {
  const { totalLen: condLen, numTargetTokens, C } = inp;
  const maskId = cfg.audio_mask_id;
  const V = cfg.audio_vocab_size;
  const nPos = C * numTargetTokens;

  const split = PASSES ? PASSES === 'split' : (!onGpu || condLen - numTargetTokens > SPLIT_MIN_PAD);
  const passes = split ? makeSplitPasses(inp, maskId) : makeBatchedPasses(inp, maskId, V);
  const ppParams = { C, T: numTargetTokens, V, maskId, guidanceScale, layerPenalty, ...passes.layout };

  const tokens = new BigInt64Array(nPos).fill(BigInt(maskId));
  const pred = new Int32Array(nPos);
  const scores = new Float32Array(nPos);
  const bigMaskId = BigInt(maskId);

  const sched = unmaskSchedule(nPos, numStep, tShift);

  // With logits left on the GPU, run() can return before the GPU is done, so
  // part of the model time shows up in the post-processing readback.
  let totalMs = 0, totalRunMs = 0, totalPPMs = 0, ppLabel = 'CPU-PP';
  for (let step = 0; step < numStep; step++) {
    // Let a pending 'cancel' message land, then honor it between steps
    await yieldMacrotask();
    if (isCancelRequested()) throw new CancelledError('cancelled');
    const k = sched[step];
    if (k <= 0) continue;
    const stepT0 = performance.now();

    const outs = [];
    try {
      for (const feeds of passes.feeds) outs.push((await mainSession.run(feeds)).audio_logits);
      const ppT0 = performance.now();
      totalRunMs += ppT0 - stepT0;
      ppLabel = await postProcessStep(outs[0], outs[outs.length - 1], ppParams, pred, scores);
      totalPPMs += performance.now() - ppT0;
    } finally {
      for (const o of outs) o.dispose();
    }

    // Gumbel noise + mask already-unmasked (fused)
    if (posTemp > 0) {
      const invTemp = 1 / posTemp;
      for (let i = 0; i < nPos; i++) {
        if (tokens[i] !== bigMaskId) { scores[i] = -Infinity; continue; }
        scores[i] = scores[i] * invTemp + (-Math.log(-Math.log(rng() + 1e-10) + 1e-10));
      }
    } else {
      for (let i = 0; i < nPos; i++)
        if (tokens[i] !== bigMaskId) scores[i] = -Infinity;
    }

    // Partial top-k using quickselect instead of full sort
    topKUnmask(scores, pred, tokens, nPos, k);

    // Feed the new tokens to both passes
    for (let c = 0; c < C; c++)
      for (let t = 0; t < numTargetTokens; t++)
        passes.setTarget(c, t, tokens[c * numTargetTokens + t]);

    const stepMs = performance.now() - stepT0;
    totalMs += stepMs;
    postMessage({
      type: 'progress', stage: 'generating', jobId: activeJobId,
      step: step + 1, numStep, stepMs: Math.round(stepMs),
      detail: `Step ${step + 1}/${numStep} (${stepMs.toFixed(0)}ms)`,
    });
  }
  const per = (ms) => `${ms.toFixed(0)}ms (${(ms / numStep).toFixed(0)}ms/step)`;
  console.log(`[perf] ${numStep} steps, ${passes.name}, ${numTargetTokens} target / ${condLen} cond tokens in ${per(totalMs)} | run: ${per(totalRunMs)} | ${ppLabel}: ${per(totalPPMs)} | JS-other: ${per(totalMs - totalRunMs - totalPPMs)}`);

  return tokens;
}

// ─── Decode & post-process ──────────────────────────────────────────────────

async function decodeTokens(tokens, C, T) {
  postMessage({ type: 'progress', stage: 'decoding', detail: 'Converting tokens to audio...' });
  const codes = new BigInt64Array(C * T);
  codes.set(tokens);
  const r = await decoderSession.run({ audio_codes: new ort.Tensor('int64', codes, [1, C, T]) });
  return r.audio_values.data;
}

function postProcessAudio(pcm, sr, normalize = true) {
  const out = trimAndFade(pcm, sr).pcm;
  let peak = peakAbs(out);
  if (normalize && peak > 1e-6) {
    const s = 0.5 / peak;
    for (let i = 0; i < out.length; i++) out[i] *= s;
    peak = 0.5;
  }
  return { pcm: out, peak };
}

// ─── Init ───────────────────────────────────────────────────────────────────

async function init(modelBaseUrl, forceCPU) {
  try {
    modelBase = modelBaseUrl;
    // Detect WebGPU — used for ONNX acceleration and GPU post-processing
    // Append ?cpu to the page URL to force CPU-only mode for testing
    let hasWorkingGPU = false;
    if (!forceCPU && typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        hasWorkingGPU = !!adapter;
      } catch {}
    }
    if (forceCPU) console.log('[init] Forced CPU mode via ?cpu flag');
    if (!hasWorkingGPU) {
      console.warn('[init] No WebGPU — ONNX will use WASM, post-processing will use CPU. Expect slower inference.');
      postMessage({ type: 'progress', stage: 'loading', detail: 'No WebGPU detected — running in CPU mode (slower)' });
    }

    postMessage({ type: 'progress', stage: 'loading', phase: 'config', detail: 'Loading config...' });
    config = await (await fetch(`${modelBaseUrl}/omnivoice-config.json`)).json();

    postMessage({ type: 'progress', stage: 'loading', phase: 'tokenizer', detail: 'Loading tokenizer (Qwen2 BPE)...' });
    tokenizer = await AutoTokenizer.from_pretrained('Gigsu/vocoloco-onnx');

    // ── Load model data ────────────────────────────────────────────────────
    // The voice encoder is not part of this: it's only needed to clone a new
    // voice and is loaded on first use (ensureEncoder).
    const dataFiles = await (await fetch(`${modelBaseUrl}/omnivoice-main-manifest.json`)).json();
    const urls = [...dataFiles.map(f => `${modelBaseUrl}/${f}`), `${modelBaseUrl}/omnivoice-decoder.onnx`];

    const cache = await caches.open(CACHE_NAME);
    const files = await Promise.all(urls.map(u => inspectFile(cache, u)));
    const pending = files.filter(f => !f.complete).length;
    const cachedBytes = files.reduce((s, f) => s + f.cachedBytes, 0);
    const totalBytes = files.every(f => f.size > 0) ? files.reduce((s, f) => s + f.size, 0) : null;

    postMessage({
      type: 'plan',
      firstRun: pending === urls.length && cachedBytes === 0,
      resuming: pending > 0 && cachedBytes > 0,
      totalBytes, cachedBytes,
      fileCount: urls.length, filesToDownload: pending,
    });

    let loadedBytes = cachedBytes, lastPost = 0;
    const postDownload = (force = false) => {
      const now = performance.now();
      if (!force && now - lastPost < 150) return;
      lastPost = now;
      postMessage({
        type: 'progress', stage: 'downloading', loadedBytes, totalBytes,
        detail: `Downloading ${(loadedBytes / 1e6).toFixed(0)} MB…`,
      });
    };
    if (pending) postDownload(true);
    else postMessage({ type: 'progress', stage: 'loading', phase: 'cache-load', detail: 'Loading from cache...' });
    const bufs = await Promise.all(urls.map((u, i) => loadFile(cache, u, {
      size: files[i].size,
      onBytes: (n) => { loadedBytes += n; postDownload(); },
    })));
    if (pending) postDownload(true);
    const decBuf = bufs.pop();
    const externalData = dataFiles.map((fname, i) => ({ path: fname, data: bufs[i] }));

    // ── Create ONNX sessions ─────────────────────────────────────────────
    let actualBackend = 'cpu';
    postMessage({ type: 'progress', stage: 'loading', phase: 'session-main', detail: 'Creating model session...' });
    if (hasWorkingGPU) {
      try {
        mainSession = await ort.InferenceSession.create(`${modelBaseUrl}/omnivoice-main-split.onnx`, {
          executionProviders: ['webgpu'], externalData, graphOptimizationLevel: 'all', enableCpuMemArena: true,
          // logits stay on the GPU, post-processing reads them there
          preferredOutputLocation: { audio_logits: 'gpu-buffer' },
        });
        actualBackend = 'webgpu';
      } catch (e) {
        console.warn('[init] Main model WebGPU failed, falling back to WASM:', e.message);
        mainSession = null;
      }
    }
    if (!mainSession) {
      mainSession = await ort.InferenceSession.create(
        `${modelBaseUrl}/omnivoice-main-split.onnx`,
        { executionProviders: ['wasm'], externalData, graphOptimizationLevel: 'all', enableCpuMemArena: true }
      );
      actualBackend = 'cpu';
    }
    console.log(`[init] Main model backend: ${actualBackend}, threads: ${ort.env.wasm.numThreads}`);

    // GPU post-processing runs on ORT's own device, so it can read the logits
    // buffers directly (a second device would also contend for the GPU)
    if (actualBackend === 'webgpu') {
      try {
        gpuPostProc = new GpuPostProcessor(await ort.env.webgpu.device);
        console.log('[init] GPU post-processor ready');
      } catch (e) {
        console.warn('[init] GPU post-processor unavailable, using CPU fallback:', e.message);
        gpuPostProc = null;
      }
    }

    onGpu = actualBackend === 'webgpu';
    sessionEps = onGpu ? ['webgpu', 'wasm'] : ['wasm'];

    postMessage({ type: 'progress', stage: 'loading', phase: 'session-decoder', detail: 'Creating decoder session...' });
    decoderSession = await ort.InferenceSession.create(decBuf, { executionProviders: sessionEps });

    // Warm up with dummy data to compile GPU shaders
    postMessage({ type: 'progress', stage: 'loading', phase: 'warmup', detail: 'Warming up...' });
    try {
      for (const b of [2, 1]) { // batched and split passes
        const out = await mainSession.run({
          input_ids: T_('int64', new BigInt64Array(b * 8 * 4).fill(1024n), [b, 8, 4]),
          audio_mask: T_('bool', new Uint8Array(b * 4), [b, 4]),
          attention_mask: T_('bool', new Uint8Array(b * 16).fill(1), [b, 1, 4, 4]),
        });
        out.audio_logits.dispose();
      }
      await decoderSession.run({ audio_codes: T_('int64', new BigInt64Array(8 * 2), [1, 8, 2]) });
    } catch (e) { /* warm-up errors are non-fatal */ }

    postMessage({ type: 'ready', backend: actualBackend, encoderAvailable: true });
  } catch (err) {
    postMessage({ type: 'error', message: `Init failed: ${err.message}` });
  }
}

// ─── Voice encoder (loaded on first use) ────────────────────────────────────

let encoderBytes = null;     // Promise<ArrayBuffer> — the download, outside the job queue
let encoderFailed = false;   // the session can't be created on this device

function fetchEncoderBytes() {
  if (!modelBase) return Promise.reject(new Error('models not loaded yet'));
  if (!encoderBytes) {
    encoderBytes = (async () => {
      const cache = await caches.open(CACHE_NAME);
      const url = `${modelBase}/omnivoice-encoder-fixed.onnx`;
      const info = await inspectFile(cache, url);
      let loaded = info.cachedBytes, lastPost = 0;
      const post = (force = false) => {
        const now = performance.now();
        if (!force && now - lastPost < 250) return;
        lastPost = now;
        const of = info.size ? ` of ${(info.size / 1e6).toFixed(0)} MB` : ' MB';
        postMessage({
          type: 'progress', stage: 'encoder-download', loadedBytes: loaded, totalBytes: info.size || null,
          detail: `Downloading the voice encoder (one-time) — ${(loaded / 1e6).toFixed(0)}${of}`,
        });
      };
      if (!info.complete) post(true);
      const buf = await loadFile(cache, url, { size: info.size, onBytes: (n) => { loaded += n; post(); } });
      if (!info.complete) postMessage({ type: 'progress', stage: 'encoder-download', done: true, detail: 'Voice encoder ready' });
      return buf;
    })();
    encoderBytes.catch(() => { encoderBytes = null; }); // a failed download can be retried
  }
  return encoderBytes;
}

// Runs inside the job queue (session creation must not overlap other ORT calls)
async function ensureEncoder() {
  if (encoderSession) return true;
  if (encoderFailed) return false;
  const buf = await fetchEncoderBytes();
  try {
    try {
      encoderSession = await ort.InferenceSession.create(buf, { executionProviders: sessionEps });
    } catch (e) {
      console.warn('Encoder WebGPU failed, falling back to WASM:', e.message);
      encoderSession = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    }
  } catch (e) {
    console.warn('[encoder] unavailable on this device:', e.message);
    encoderFailed = true;
    postMessage({ type: 'encoder-status', available: false });
    return false;
  }
  return true;
}

// ─── Reference audio encoding ───────────────────────────────────────────────

// Runs the ~654 MB encoder over 24 kHz mono PCM and returns the audio tokens
// as a flat Int32Array [C*T] (codebook-major — same layout as the ONNX output).
async function encodeRefPcm(refAudio) {
  const pcmF32 = new Float32Array(refAudio);
  // Clip to hop_length alignment (hop=960 for 24kHz)
  const hopLength = 960;
  const clipLen = pcmF32.length - (pcmF32.length % hopLength);
  const aligned = pcmF32.slice(0, clipLen);
  const inputTensor = new ort.Tensor('float32', aligned, [1, 1, aligned.length]);
  const encResult = await encoderSession.run({ input_values: inputTensor });
  const codesData = encResult.audio_codes.data; // BigInt64Array
  const codeDims = encResult.audio_codes.dims; // [1, C, T]
  const C = Number(codeDims[1]);
  const tokenCount = Number(codeDims[2]);
  const tokens = new Int32Array(C * tokenCount);
  for (let i = 0; i < tokens.length; i++) tokens[i] = Number(codesData[i]);
  return { tokens, tokenCount, numCodebooks: C, duration: aligned.length / config.sampling_rate };
}

// Expand a flat Int32Array [C*T] into the nested [C][T] number arrays that
// prepareInferenceInputs consumes.
function expandTokens(flat, C) {
  const T = Math.floor(flat.length / C);
  const out = [];
  for (let c = 0; c < C; c++) {
    const row = new Array(T);
    for (let t = 0; t < T; t++) row[t] = flat[c * T + t];
    out.push(row);
  }
  return out;
}

async function encodeReference({ requestId, refAudio }) {
  try {
    if (!(await ensureEncoder())) {
      postMessage({ type: 'encode-error', requestId, code: 'encoder-unavailable', message: 'Voice encoder not available on this device' });
      return;
    }
    postMessage({ type: 'progress', stage: 'encoding', detail: 'Analyzing voice...' });
    const r = await encodeRefPcm(refAudio);
    postMessage({ type: 'encoded', requestId, tokens: r.tokens, tokenCount: r.tokenCount, numCodebooks: r.numCodebooks, duration: r.duration }, [r.tokens.buffer]);
  } catch (err) {
    postMessage({ type: 'encode-error', requestId, code: 'encode-failed', message: err.message });
  }
}

// ─── Synthesize ─────────────────────────────────────────────────────────────

const SHORT_TEXT_TOKENS = 100; // ~4 s
const HEADROOM_TOKENS = 12;    // ~0.5 s

async function synthesize(params) {
  const {
    jobId = null,
    text, lang = null, refAudio = null, refText = null, refTokens = null, refRateTokens = null,
    instruct = null,
    numStep = 32, guidanceScale = 2.0, tShift = 0.1, speed = 1.0, // OmniVoice defaults
    tempo = 1.0, // speed slider: stretches the finished audio, pitch unchanged
    seed = null,
    returnTokens = false, normalize = true,
    denoise = true, // OmniVoice default; the app turns it off when there is a reference
  } = params;

  try {
    // Use seeded PRNG for deterministic output when seed is provided
    rng = seed != null ? mulberry32(seed) : Math.random;
    const C = config.num_audio_codebook;

    // Resolve reference tokens for voice cloning:
    // 1. pre-encoded tokens (cached voice / chunk chaining) — no encoder needed
    // 2. raw PCM + encoder
    // 3. raw PCM without encoder — warn and proceed uncloned
    let refAudioTokens = null;
    if (refTokens && refTokens.length >= C) {
      refAudioTokens = expandTokens(refTokens, C);
      postMessage({ type: 'progress', stage: 'encoding', detail: `Using cached voice (${refAudioTokens[0].length} tokens)` });
    } else if (refAudio && await ensureEncoder()) {
      postMessage({ type: 'progress', stage: 'encoding', detail: 'Encoding reference audio...' });
      const enc = await encodeRefPcm(refAudio);
      refAudioTokens = expandTokens(enc.tokens, enc.numCodebooks);
      postMessage({ type: 'progress', stage: 'encoding', detail: `Encoded: ${enc.tokenCount} tokens (${enc.duration.toFixed(1)}s)` });
    } else if (refAudio) {
      postMessage({ type: 'progress', stage: 'warning', detail: 'Voice cloning unavailable on this device (not enough memory for encoder)' });
    }

    // Duration estimation — the (refText, refTokens) pair must stay consistent:
    // mixing a real token count with the default text (or vice versa) skews the
    // estimate ~10x. estimateTargetTokens falls back to its internal defaults
    // whenever either half is missing.
    // refRateTokens: a chained reference's token count without the headroom
    // below, so headroom never makes the following chunks slower.
    const estRefText = refAudioTokens ? refText : null;
    const estRefTokens = refAudioTokens ? (refRateTokens ?? refAudioTokens[0].length) : null;
    const estimate = estimateTargetTokens(text, estRefText, estRefTokens, speed);
    // Short texts get headroom: the estimate scales with text length, the
    // slowdown at the end of a sentence doesn't, and short outputs ran out of
    // time mid-word. Unused time ends as trailing silence, which gets trimmed.
    const headroom = estimate < SHORT_TEXT_TOKENS ? HEADROOM_TOKENS : 0;
    let numTargetTokens = Math.min(estimate + headroom, 700);

    postMessage({ type: 'progress', stage: 'preparing', detail: `Target: ${numTargetTokens} tokens` });

    const inputs = await prepareInferenceInputs(text, numTargetTokens, tokenizer, config, {
      lang, instruct, refText: refAudioTokens ? refText : null, refAudioTokens,
      denoise,
    });

    const tokens = await generateIterative(inputs, config, numStep, guidanceScale, tShift);

    if (isCancelRequested()) throw new CancelledError('cancelled');
    const rawPcm = await decodeTokens(tokens, C, numTargetTokens);

    postMessage({ type: 'progress', stage: 'postprocessing', detail: 'Processing audio...' });
    const stretched = timeStretch(rawPcm, config.sampling_rate, tempo);
    const { pcm, peak } = postProcessAudio(stretched, config.sampling_rate, normalize);

    const reply = { type: 'audio', jobId, pcm, sampleRate: config.sampling_rate, peak };
    const transfers = [pcm.buffer];
    if (returnTokens) {
      // Generated tokens are already valid reference-audio tokens — expose them
      // flat so the app can chain them as the voice reference for later chunks.
      const flat = new Int32Array(tokens.length);
      for (let i = 0; i < tokens.length; i++) flat[i] = Number(tokens[i]);
      reply.tokens = flat;
      reply.tokenCount = numTargetTokens;
      reply.rateTokens = numTargetTokens - headroom;
      transfers.push(flat.buffer);
    }
    postMessage(reply, transfers);
  } catch (err) {
    if (err instanceof CancelledError) {
      postMessage({ type: 'cancelled', jobId });
      return;
    }
    postMessage({ type: 'error', jobId, message: `Synthesis failed: ${err.message}\n${err.stack}` });
  }
}

// ─── Message handler ────────────────────────────────────────────────────────
// 'cancel' is handled synchronously (never queued) so it can interrupt a
// running synthesis; everything else is serialized through a job queue since
// the pipeline shares mutable module-level buffers.

let jobQueue = Promise.resolve();

async function handleMessage(msg) {
  if (msg.type === 'init') {
    await init(msg.modelBaseUrl, msg.forceCPU);
  } else if (msg.type === 'synthesize') {
    activeJobId = msg.jobId ?? null;
    activeCancelBaseline = msg._cancelBaseline ?? cancelCounter;
    await synthesize(msg);
    activeJobId = null;
  } else if (msg.type === 'encode-reference') {
    await encodeReference(msg);
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'cancel') { cancelCounter++; return; }
  // Snapshot the cancel counter at arrival so a cancel that lands while this
  // job is still queued (e.g. behind an encode) is honored when it runs.
  if (msg.type === 'synthesize') msg._cancelBaseline = cancelCounter;
  const enqueue = () => {
    jobQueue = jobQueue.then(() => handleMessage(msg)).catch((err) => console.error('[worker] job failed:', err));
  };
  // The encoder downloads outside the queue (generation keeps running
  // meanwhile); the encode job is queued once the file is here.
  if (msg.type === 'prefetch-encoder') {
    if (!encoderSession && !encoderFailed) fetchEncoderBytes().catch((err) => console.warn('[encoder] download failed:', err.message));
    return;
  }
  if (msg.type === 'encode-reference' && !encoderSession && !encoderFailed) {
    fetchEncoderBytes().then(enqueue, (err) => postMessage({
      type: 'encode-error', requestId: msg.requestId, code: 'encode-failed',
      message: `Could not download the voice encoder: ${err.message}`,
    }));
    return;
  }
  enqueue();
};
