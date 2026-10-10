/**
 * TTS Web Worker (ES Module) — runs OmniVoice on our own WebGPU engine
 * (engine/), or on ONNX Runtime Web where the engine can't run (no WebGPU, no
 * shader-f16). Text goes through our Qwen2 tokenizer (engine/qwen2-tokenizer.js).
 */

import { estimateTargetTokens } from '../duration-estimator.js?v=2';
import { addEndPunctuation } from '../sentence-buffer.js?v=3'; // ?v: a stale cached copy lacks this export
import { trimAndFade, peakAbs } from '../audio-postprocess.js?v=3';
import { timeStretch } from '../time-stretch.js';
import { GpuPostProcessor } from './gpu-postprocess.js?v=2';
import { unmaskSchedule } from './unmask-schedule.js?v=2';
import { inspectFile, loadFile, openDiskFile, diskHasFiles } from './model-files.js?v=4';
import { voicePitch, refInfoKey, genderInstruct } from './voice-pitch.js';
import { OmniVoiceEngine, httpSource } from './engine/omnivoice-engine.js';
import { AudioDecoder } from './engine/audio-decoder.js';
import { AudioEncoder } from './engine/audio-encoder.js';
import { Qwen2Tokenizer } from './engine/qwen2-tokenizer.js';

// The engine's files, next to the ONNX models: main model (f16 weights), decoder
// and voice encoder (f32), each a .json manifest and a .bin with the weights
const ENGINE_FILES = { main: 'engine/omnivoice-f16', decoder: 'engine/decoder-f32', encoder: 'engine/encoder-f32' };
let engine = null;       // main model on the engine (OmniVoiceEngine)
let enginePlan = null;   // its plan for the current generation (enginePasses)
let engineDevice = null;
let audioDecoder = null; // the engine's decoder (replaces ORT's decoderSession)
let audioEncoder = null; // the engine's voice encoder (replaces ORT's encoderSession), loaded on first use

// Phones and tablets (app.js): one big model in GPU memory at a time. iOS kills a tab that
// holds more than ~1.5 GB, and the main model (~1 GB) next to the voice encoder or Whisper
// is more. There this worker is ended while those run (app.js, parkSpeech) and the voice
// encoder has a worker of its own (encoder-worker.js).
let lowMemory = false;

// onnxruntime-web/webgpu: the native WebGPU EP (the default/`all` bundles use
// the deprecated JSEP backend); its WASM EP serves the CPU fallback too.
// Only loaded where the engine can't run.
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
let ort = null;
async function loadOrt() {
  ort ??= await import(`${ORT_CDN}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_CDN;
  // Maximize performance — multi-threading requires cross-origin isolation (COOP/COEP headers)
  ort.env.wasm.numThreads = self.crossOriginIsolated ? (navigator.hardwareConcurrency || 4) : 1;
  ort.env.wasm.simd = true;
}

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

// Model download consent (see downloadFiles): resolved by the app's 'start-download'
let downloadOk = false;
let onDownloadOk = null;
function approveDownload() { downloadOk = true; onDownloadOk?.(); onDownloadOk = null; }
function downloadApproved() { return downloadOk ? Promise.resolve() : new Promise((resolve) => { onDownloadOk = resolve; }); }

// The memory check starts once app.js has noted it (its 'memory-check-go'), see probeMemory
let onMemoryGo = null;

// Whether model files were stored on this device before (a download agreed to then, maybe by
// an earlier version): the engine's on disk, ONNX Runtime's in the Cache API. No network.
async function storedBefore() {
  if (await diskHasFiles()) return true;
  return (await (await caches.open(CACHE_NAME)).keys()).length > 0;
}

// A network step that never answers fails after `ms` instead of leaving the app spinning
function withTimeout(promise, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${what} timed out`)), ms); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

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
const TOKENIZER_CACHE = 'vocoloco-tokenizer-v1';

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

// OmniVoice's non-verbal tags, tokenized on their own like upstream's
// _tokenize_with_nonverbal_tags: in context the BPE merges a tag with its
// neighbours (" [" for the space before it), and the model read "[sigh]" as a word
const NONVERBAL_TAG = /\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]/g;

function tokenizeWithNonverbalTags(text, tok) {
  const ids = [];
  const add = (s) => { if (s) ids.push(...tok.encode(s)); };
  let last = 0;
  for (const m of text.matchAll(NONVERBAL_TAG)) {
    add(text.slice(last, m.index));
    add(m[0]);
    last = m.index + m[0].length;
  }
  add(text.slice(last));
  return ids;
}

async function prepareInferenceInputs(text, numTargetTokens, tok, cfg, opts = {}) {
  const { refText = null, refAudioTokens = null, lang = null, instruct = null, denoise = true } = opts;
  const C = cfg.num_audio_codebook;
  const maskId = cfg.audio_mask_id;

  // Build style string (upstream adds <|denoise|> only with a reference)
  let styleText = '';
  if (denoise && refAudioTokens) styleText += '<|denoise|>';
  styleText += `<|lang_start|>${lang || 'None'}<|lang_end|>`;
  styleText += `<|instruct_start|>${instruct || 'None'}<|instruct_end|>`;

  // Build text string
  let fullText = refText ? addEndPunctuation(refText) + ' ' + text.trim() : text.trim();
  fullText = fullText.replace(/[\r\n]+/g, '').replace(/[ \t]+/g, ' ');
  const wrappedText = `<|text_start|>${fullText}<|text_end|>`;

  const styleIds = tok.encode(styleText);
  const textIds = tokenizeWithNonverbalTags(wrappedText, tok);

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

// The engine's plan (working buffers, bind groups) for this generation's shape (L, T);
// freed after the generation (synthesize), long ones take hundreds of MB
async function enginePasses(inp) {
  const { inputIds, audioMask, totalLen: L, numTargetTokens: T } = inp;
  await engine.tuneFor(L, T); // once: the fastest tiles on this device
  enginePlan = engine.plan(L, T);
  await enginePlan.setInputs(Int32Array.from(inputIds, Number), audioMask.indexOf(1));
  return { layout: {} };
}

// pred / scores for the current step: GPU post-processing straight from ORT's
// output buffers when possible, otherwise on the CPU.
async function postProcessStep(condOut, uncondOut, params, pred, scores) {
  if (gpuPostProc && condOut.location === 'gpu-buffer' && uncondOut.location === 'gpu-buffer') {
    try {
      await gpuPostProc.run(condOut.gpuBuffer, uncondOut.gpuBuffer,
        { ...params, condElems: condOut.size, uncondElems: uncondOut.size }, pred, scores);
      return;
    } catch (e) {
      console.warn('[gpu-postprocess] failed, falling back to CPU:', e.message);
      gpuPostProc.destroy();
      gpuPostProc = null;
    }
  }
  const cond = condOut.location === 'cpu' ? condOut.data : await condOut.getData();
  const uncond = uncondOut === condOut ? cond : (uncondOut.location === 'cpu' ? uncondOut.data : await uncondOut.getData());
  cpuPostProcess(cond, uncond, params, pred, scores);
}

// ─── Reference pitch ────────────────────────────────────────────────────────

const refInfos = new Map(); // reference tokens → { pitch } of their audio

// known: { key, pitch } the app kept with a saved voice; a fresh measurement is posted
// as 'ref-info' for the app to keep, so a voice's reference is decoded once, not once per session
async function referenceInfo(refAudioTokens, C, known = null) {
  const T = refAudioTokens[0].length;
  const tokens = new Int32Array(C * T);
  for (let c = 0; c < C; c++) tokens.set(refAudioTokens[c], c * T);
  const key = refInfoKey(tokens, C);
  if (!refInfos.has(key) && known?.key === key) refInfos.set(key, { pitch: known.pitch });
  if (!refInfos.has(key)) {
    const pcm = await decodeTokens(BigInt64Array.from(tokens, BigInt), C, T, { quiet: true });
    if (refInfos.size >= 8) refInfos.clear();
    refInfos.set(key, { pitch: voicePitch(pcm, config.sampling_rate) });
    postMessage({ type: 'ref-info', key, ...refInfos.get(key) });
  }
  return refInfos.get(key);
}

async function generateIterative(inp, cfg, numStep, guidanceScale, tShift, layerPenalty = 5.0, posTemp = 5.0) {
  const { totalLen: condLen, numTargetTokens, C } = inp;
  const maskId = cfg.audio_mask_id;
  const V = cfg.audio_vocab_size;
  const nPos = C * numTargetTokens;

  const split = !onGpu || condLen - numTargetTokens > SPLIT_MIN_PAD;
  const passes = engine ? await enginePasses(inp) : split ? makeSplitPasses(inp, maskId) : makeBatchedPasses(inp, maskId, V);
  const ppParams = { C, T: numTargetTokens, V, maskId, guidanceScale, layerPenalty, ...passes.layout };

  const tokens = new BigInt64Array(nPos).fill(BigInt(maskId));
  const pred = new Int32Array(nPos);
  const scores = new Float32Array(nPos);
  const bigMaskId = BigInt(maskId);

  const sched = unmaskSchedule(nPos, numStep, tShift);

  for (let step = 0; step < numStep; step++) {
    // Let a pending 'cancel' message land, then honor it between steps
    await yieldMacrotask();
    if (isCancelRequested()) throw new CancelledError('cancelled');
    const k = sched[step];
    if (k <= 0) continue;
    const stepT0 = performance.now();

    if (engine) { // one pass (cond + uncond rows) and the post-processing on the engine's GPU
      enginePlan.setTargets(tokens);
      await enginePlan.step(pred, scores, ppParams);
    } else {
      const outs = [];
      try {
        for (const feeds of passes.feeds) outs.push((await mainSession.run(feeds)).audio_logits);
        await postProcessStep(outs[0], outs[outs.length - 1], ppParams, pred, scores);
      } finally {
        for (const o of outs) o.dispose();
      }
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

    // Feed the new tokens to both passes (the engine reads them from tokens at the next step)
    if (!engine) for (let c = 0; c < C; c++)
      for (let t = 0; t < numTargetTokens; t++)
        passes.setTarget(c, t, tokens[c * numTargetTokens + t]);

    const stepMs = performance.now() - stepT0;
    postMessage({
      type: 'progress', stage: 'generating', jobId: activeJobId,
      step: step + 1, numStep, stepMs: Math.round(stepMs),
      detail: `Step ${step + 1}/${numStep} (${stepMs.toFixed(0)}ms)`,
    });
  }

  return tokens;
}

// ─── Decode & post-process ──────────────────────────────────────────────────

async function decodeTokens(tokens, C, T, { quiet = false } = {}) {
  if (!quiet) postMessage({ type: 'progress', stage: 'decoding', detail: 'Converting tokens to audio...' });
  if (audioDecoder) {
    try {
      return await audioDecoder.decode(Int32Array.from(tokens, Number), T);
    } finally {
      audioDecoder.releasePlans(); // its working buffers grow with the clip
    }
  }
  const codes = new BigInt64Array(C * T);
  codes.set(tokens);
  const r = await decoderSession.run({ audio_codes: new ort.Tensor('int64', codes, [1, C, T]) });
  return r.audio_values.data;
}

function postProcessAudio(pcm, sr, normalize = true, edges = {}) {
  const out = trimAndFade(pcm, sr, edges).pcm;
  let peak = peakAbs(out);
  if (normalize && peak > 1e-6) {
    const s = 0.5 / peak;
    for (let i = 0; i < out.length; i++) out[i] *= s;
    peak = 0.5;
  }
  return { pcm: out, peak };
}

// ─── Init ───────────────────────────────────────────────────────────────────

// Posts the download plan for `urls`, waits for the app's go-ahead where it asks
// first (confirmDownload: mobile data, storage) and gets every file with
// load(cache, url, { size, onBytes }), reporting the download's progress
async function downloadFiles(urls, confirmDownload, load) {
  const cache = await caches.open(CACHE_NAME);
  const files = await Promise.all(urls.map(u => inspectFile(cache, u)));
  const pending = files.filter(f => !f.complete).length;
  const cachedBytes = files.reduce((s, f) => s + f.cachedBytes, 0);
  const totalBytes = files.every(f => f.size > 0) ? files.reduce((s, f) => s + f.size, 0) : null;

  // Not after a consent this session (init's askFirst): the app shows the download already
  if (!downloadOk) {
    postMessage({
      type: 'plan',
      firstRun: pending === urls.length && cachedBytes === 0,
      resuming: pending > 0 && cachedBytes > 0,
      totalBytes, cachedBytes,
      fileCount: urls.length, filesToDownload: pending,
    });
  }
  if (pending && confirmDownload) await downloadApproved();

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
  const out = await Promise.all(urls.map((u, i) => load(cache, u, {
    size: files[i].size,
    onBytes: (n) => { loadedBytes += n; postDownload(); },
  })));
  if (pending) postDownload(true);
  return out;
}

// 'models-gone': a model file the engine reads from vanished (the browser cleared the site's
// storage, e.g. when the device ran low on space); app.js offers the download again
function errorCode(err, otherwise = undefined) {
  return err?.name === 'NotFoundError' ? 'models-gone' : otherwise;
}

// The engine's weights live in the app's disk store (OPFS) and are read from there
// (the main model's file stays open: its text embedding rows are read per generation),
// or straight from the network where there is no disk store
async function engineSource(cache, url, opts) {
  return await openDiskFile(url, opts) ?? httpSource(url);
}

// The text tokenizer (engine/qwen2-tokenizer.js) from the model repo's tokenizer.json (11 MB), kept
// in its own cache (a compressed response has no size up front, which the disk store needs)
async function loadTokenizer(url) {
  const bytes = await loadFile(await caches.open(TOKENIZER_CACHE), url);
  const tok = new Qwen2Tokenizer(JSON.parse(new TextDecoder().decode(bytes)));
  await caches.delete('transformers-cache').catch(() => {}); // where earlier versions kept it
  return tok;
}

// A GPU the engine runs on: its f16 math needs 'shader-f16'
async function engineAdapter() {
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    return adapter?.features.has('shader-f16') ? adapter : null;
  } catch { return null; }
}

async function loadEngine(adapter, man, source, decMan, decSource) {
  postMessage({ type: 'progress', stage: 'loading', phase: 'session-main', detail: 'Loading the model...' });
  const lim = adapter.limits;
  engineDevice = await adapter.requestDevice({
    requiredFeatures: ['shader-f16', 'timestamp-query'].filter((f) => adapter.features.has(f)),
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
    },
  });
  // The system can take the GPU away (iOS under memory pressure): say so, app.js restarts the engine
  const device = engineDevice;
  device.lost.then((info) => {
    if (info.reason === 'destroyed' || device !== engineDevice) return;
    postMessage({ type: 'error', code: 'gpu-lost', message: `The GPU stopped (${info.message || 'device lost'})` });
  });
  engine = await OmniVoiceEngine.load(engineDevice, man, source, {
    act: 'f16',
    segments: 8, // command buffers per step: fewer, longer ones made the page stutter on the Mac
    onProgress: (f) => postMessage({ type: 'progress', stage: 'loading', phase: 'session-main', detail: `Loading the model ${Math.round(f * 100)}%` }),
  });
  postMessage({ type: 'progress', stage: 'loading', phase: 'session-decoder', detail: 'Loading the decoder...' });
  audioDecoder = await AudioDecoder.load(engineDevice, decMan, decSource);
}

function releaseEngine() {
  for (const o of [enginePlan, engine, audioDecoder, engineDevice]) try { o?.destroy(); } catch { /* already gone */ }
  enginePlan = engine = audioDecoder = engineDevice = null;
}

// Whether `bytes` of GPU memory can be held at once, on a device of its own: 64 MB buffers,
// each cleared on the GPU so it is really backed, then freed. Where they don't fit, the system
// reports out-of-memory, or (iOS) kills the tab, which app.js notices on the next load.
async function probeMemory(bytes) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (!adapter) return false;
  const device = await adapter.requestDevice();
  const PIECE = 64 << 20;
  const bufs = [];
  let ok = true;
  try {
    device.pushErrorScope('out-of-memory');
    for (let left = bytes; left > 0; left -= PIECE) {
      const b = device.createBuffer({ size: Math.min(PIECE, left), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      bufs.push(b);
      const enc = device.createCommandEncoder();
      enc.clearBuffer(b);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone(); // piece by piece, so the system sees it grow
    }
    await new Promise((r) => setTimeout(r, 300)); // and has a moment to object
    if (await device.popErrorScope()) ok = false;
  } catch { ok = false; }
  for (const b of bufs) b.destroy();
  device.destroy();
  return ok;
}

// lowMemory: phones and tablets, see lowMemory above; there ONNX Runtime (~2.4 GB) never fits,
// so there is no fallback. probeBytes: the memory check before the download (probeMemory).
// confirmDownload: ask before downloading. Where no model was ever stored, before anything at
// all goes over the network (storedBefore); otherwise in downloadFiles, when a file is missing
async function init({ modelBaseUrl, forceCPU = false, confirmDownload = false, lowMemory: lm = false, probeBytes = 0 }) {
  lowMemory = lm;
  try {
    // forceCPU: ?cpu on the page URL runs the WebAssembly fallback
    const adapter = forceCPU ? null : await engineAdapter();
    if (!adapter && lowMemory) {
      postMessage({ type: 'error', code: 'no-engine', message: "This browser can't run the voice model on a phone or tablet: it needs WebGPU with 16-bit floats (iOS or iPadOS 26 in any browser, or Chrome 121 or newer on Android)." });
      return;
    }
    // On the device only: a phone that can't hold the model hears so before being asked to download it
    if (adapter && probeBytes) {
      postMessage({ type: 'progress', stage: 'loading', phase: 'memory', detail: 'Checking memory...' });
      // app.js notes the check before it runs: if iOS kills the tab, the next load knows why
      await new Promise((resolve) => { onMemoryGo = resolve; postMessage({ type: 'memory-check', state: 'start' }); });
      const ok = await probeMemory(probeBytes);
      postMessage({ type: 'memory-check', ok });
      if (!ok) return; // app.js explains
    }
    if (confirmDownload && !await storedBefore()) {
      // The size is an estimate here: the exact one would take a request to Hugging Face
      postMessage({ type: 'plan', firstRun: true, resuming: false, totalBytes: null, cachedBytes: 0, filesToDownload: 1, backend: adapter ? 'engine' : 'onnx' });
      await downloadApproved();
    }

    postMessage({ type: 'progress', stage: 'loading', phase: 'config', detail: 'Loading config...' });
    config = await withTimeout(fetch(`${modelBaseUrl}/omnivoice-config.json`).then((r) => r.json()), 60e3, 'Loading the model config');

    postMessage({ type: 'progress', stage: 'loading', phase: 'tokenizer', detail: 'Loading tokenizer...' });
    tokenizer = await withTimeout(loadTokenizer(`${modelBaseUrl}/tokenizer.json`), 120e3, 'Loading the tokenizer');

    const url = (f) => `${modelBaseUrl}/${f}`;
    const manifest = async (f) => {
      const r = await fetch(`${url(f)}.json`, { cache: 'no-store' });
      if (!r.ok) throw new Error(`${f}.json: ${r.status}`);
      return r.json();
    };
    const manifests = adapter && await Promise.all([manifest(ENGINE_FILES.main), manifest(ENGINE_FILES.decoder)])
      .catch((e) => {
        if (lowMemory) throw e;
        console.warn('[init] No engine files, using ONNX Runtime:', e.message);
        return null;
      });
    if (manifests) {
      // The voice encoder is not part of this: it's only needed to clone a new
      // voice and is loaded on first use (ensureEncoder)
      const [man, decMan] = manifests;
      const [source, decSource] = await downloadFiles([`${url(ENGINE_FILES.main)}.bin`, `${url(ENGINE_FILES.decoder)}.bin`], confirmDownload, engineSource);
      try {
        await loadEngine(adapter, man, source, decMan, decSource);
      } catch (e) {
        releaseEngine();
        if (lowMemory) throw e;
        console.warn('[init] The engine failed, falling back to ONNX Runtime:', e.message);
      }
    }
    if (engine) {
      onGpu = true;
      // The ONNX models of earlier versions aren't needed where the engine runs
      await caches.delete(CACHE_NAME).catch(() => {});
      await postReady(modelBaseUrl, 'engine', `${url(ENGINE_FILES.encoder)}.bin`);
      return;
    }
    if (adapter) downloadOk = false; // agreed to the engine's download, not to ONNX Runtime's (asked again)
    await initOrt(modelBaseUrl, forceCPU, confirmDownload);
  } catch (err) {
    postMessage({ type: 'error', message: `Init failed: ${err.message}` });
  }
}

async function initOrt(modelBaseUrl, forceCPU, confirmDownload) {
  // Detect WebGPU — used for ONNX acceleration and GPU post-processing
  let hasWorkingGPU = false;
  if (!forceCPU && navigator.gpu) {
    try { hasWorkingGPU = !!(await navigator.gpu.requestAdapter()); } catch { /* none */ }
  }
  if (!hasWorkingGPU) {
    console.warn('[init] No WebGPU: ONNX will use WASM, post-processing will use CPU. Expect slower inference.');
    postMessage({ type: 'progress', stage: 'loading', detail: 'No WebGPU detected, running in CPU mode (slower)' });
  }
  await loadOrt();

  // ── Load model data ────────────────────────────────────────────────────
  const dataFiles = await (await fetch(`${modelBaseUrl}/omnivoice-main-manifest.json`)).json();
  const urls = [...dataFiles.map(f => `${modelBaseUrl}/${f}`), `${modelBaseUrl}/omnivoice-decoder.onnx`];
  const bufs = await downloadFiles(urls, confirmDownload, loadFile);
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
  // Drop our copy of the weights now that the session has them
  bufs.length = 0;
  externalData.length = 0;

  // GPU post-processing runs on ORT's own device, so it can read the logits
  // buffers directly (a second device would also contend for the GPU)
  if (actualBackend === 'webgpu') {
    try {
      gpuPostProc = new GpuPostProcessor(await ort.env.webgpu.device);
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

  await postReady(modelBaseUrl, actualBackend, `${modelBaseUrl}/omnivoice-encoder-fixed.onnx`);
}

// Ready, with whether cloning a voice will need the encoder download first
async function postReady(modelBaseUrl, backend, encoderUrl) {
  const encoderInfo = await inspectFile(await caches.open(CACHE_NAME), encoderUrl).catch(() => null);
  modelBase = modelBaseUrl; // from here on the encoder can be fetched (fetchEncoder)
  postMessage({
    type: 'ready', backend, encoderAvailable: true,
    encoderCached: !!encoderInfo?.complete, encoderBytes: encoderInfo?.size || null,
  });
}

// ─── Voice encoder (loaded on first use) ────────────────────────────────────
// The engine runs it on WebGPU (engine/audio-encoder.js): each stage's weights are
// uploaded for an encode and dropped after it (in WebKit freed only once their bind groups are
// garbage collected). Phones encode in encoder-worker.js instead.

let encoderFiles = null;     // Promise — the download, outside the job queue
let encoderFailed = false;   // ONNX Runtime: the session can't be created on this device

// The encoder's weights, downloaded once with load(cache, url, opts)
async function downloadEncoder(url, load) {
  const cache = await caches.open(CACHE_NAME);
  const info = await inspectFile(cache, url);
  let loaded = info.cachedBytes, lastPost = 0;
  const post = (force = false) => {
    const now = performance.now();
    if (!force && now - lastPost < 250) return;
    lastPost = now;
    const amount = info.size ? `${Math.min(100, Math.floor((loaded / info.size) * 100))}%` : `${(loaded / 1e6).toFixed(0)} MB`;
    postMessage({
      type: 'progress', stage: 'encoder-download', loadedBytes: loaded, totalBytes: info.size || null,
      detail: `Downloading voice encoder ${amount}`,
    });
  };
  if (!info.complete) post(true);
  const out = await load(cache, url, { size: info.size, onBytes: (n) => { loaded += n; post(); } });
  if (!info.complete) postMessage({ type: 'progress', stage: 'encoder-download', done: true, detail: 'Voice encoder ready' });
  return out;
}

// Engine: [manifest, weights source]; ONNX Runtime: the model's bytes
function fetchEncoder() {
  if (!modelBase) return Promise.reject(new Error('models not loaded yet'));
  if (!encoderFiles) {
    const base = `${modelBase}/${ENGINE_FILES.encoder}`;
    encoderFiles = engine
      ? Promise.all([
        fetch(`${base}.json`, { cache: 'no-store' }).then((r) => r.json()),
        downloadEncoder(`${base}.bin`, engineSource),
      ])
      : downloadEncoder(`${modelBase}/omnivoice-encoder-fixed.onnx`, loadFile);
    encoderFiles.catch(() => { encoderFiles = null; }); // a failed download can be retried
  }
  return encoderFiles;
}

const needsEncoder = () => (engine ? !audioEncoder : !encoderSession && !encoderFailed);

// Runs inside the job queue (session creation must not overlap other ORT calls)
async function ensureEncoder() {
  if (engine) {
    if (lowMemory) return false; // no room next to the main model: phones encode in encoder-worker.js
    if (!audioEncoder) {
      const [man, source] = await fetchEncoder();
      audioEncoder = await AudioEncoder.load(engineDevice, man, source, { resident: false });
    }
    return true;
  }
  if (encoderSession) return true;
  if (encoderFailed) return false;
  const buf = await fetchEncoder();
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
  if (audioEncoder) {
    const { codes, T } = await audioEncoder.encode(aligned);
    return { tokens: codes, tokenCount: T, numCodebooks: codes.length / T, duration: aligned.length / config.sampling_rate };
  }
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
    postMessage({ type: 'encode-error', requestId, code: errorCode(err, 'encode-failed'), message: err.message });
  }
}

// ─── Synthesize ─────────────────────────────────────────────────────────────

const SHORT_TEXT_TOKENS = 100; // ~4 s
const HEADROOM_TOKENS = 12;    // ~0.5 s

async function synthesize(params) {
  const {
    jobId = null,
    text, lang = null, refAudio = null, refText = null, refTokens = null, refRateTokens = null,
    refInfo: knownRefInfo = null, // a saved voice's measured reference (referenceInfo)
    instruct = null,
    numStep = 32, guidanceScale = 2.0, tShift = 0.1, speed = 1.0, // OmniVoice defaults
    tempo = 1.0, // speed slider: stretches the finished audio, pitch unchanged
    seed = null,
    returnTokens = false, normalize = true,
    leadMs = 100, trailMs = 100, // silence kept before and after the speech (the app keeps more at the ends of the whole text)
    denoise = true, // OmniVoice default; the app turns it off when there is a reference
    autoGender = true, // reference voices without instruct: name their gender, see genderInstruct (voice-pitch.js)
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
    const estRefText = refAudioTokens && refText ? addEndPunctuation(refText) : null; // like upstream's prompt
    const estRefTokens = refAudioTokens ? (refRateTokens ?? refAudioTokens[0].length) : null;
    const estimate = estimateTargetTokens(text, estRefText, estRefTokens, speed);
    // Short texts get headroom: the estimate scales with text length, the
    // slowdown at the end of a sentence doesn't, and short outputs ran out of
    // time mid-word. Unused time ends as trailing silence, which gets trimmed.
    const headroom = estimate < SHORT_TEXT_TOKENS ? HEADROOM_TOKENS : 0;
    let numTargetTokens = Math.min(estimate + headroom, 700);

    postMessage({ type: 'progress', stage: 'preparing', detail: `Target: ${numTargetTokens} tokens` });

    const refInfo = refAudioTokens && autoGender && !instruct ? await referenceInfo(refAudioTokens, C, knownRefInfo) : null;
    const inputs = await prepareInferenceInputs(text, numTargetTokens, tokenizer, config, {
      lang, instruct: instruct ?? genderInstruct(refInfo?.pitch), refText: refAudioTokens ? refText : null, refAudioTokens,
      denoise,
    });

    let tokens;
    try {
      tokens = await generateIterative(inputs, config, numStep, guidanceScale, tShift);
    } finally {
      enginePlan?.destroy(); // see enginePasses
      enginePlan = null;
    }

    if (isCancelRequested()) throw new CancelledError('cancelled');
    const rawPcm = await decodeTokens(tokens, C, numTargetTokens);

    postMessage({ type: 'progress', stage: 'postprocessing', detail: 'Processing audio...' });
    const stretched = timeStretch(rawPcm, config.sampling_rate, tempo);
    const { pcm, peak } = postProcessAudio(stretched, config.sampling_rate, normalize, { leadMs, trailMs });

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
    postMessage({ type: 'error', jobId, code: errorCode(err), message: `Synthesis failed: ${err.message}\n${err.stack}` });
  }
}

// ─── Message handler ────────────────────────────────────────────────────────
// 'cancel' is handled synchronously (never queued) so it can interrupt a
// running synthesis; everything else is serialized through a job queue since
// the pipeline shares mutable module-level buffers.

let jobQueue = Promise.resolve();

async function handleMessage(msg) {
  if (msg.type === 'init') {
    await init(msg);
  } else if (msg.type === 'synthesize') {
    activeJobId = msg.jobId ?? null;
    activeCancelBaseline = msg._cancelBaseline ?? cancelCounter;
    await synthesize(msg);
    activeJobId = null;
  } else if (msg.type === 'encode-reference') {
    await encodeReference(msg);
  } else if (msg.type === 'release-model') { // phones: the jobs before are done, app.js ends this worker
    postMessage({ type: 'released' });
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'cancel') { cancelCounter++; return; }
  if (msg.type === 'start-download') { approveDownload(); return; }
  if (msg.type === 'memory-check-go') { onMemoryGo?.(); onMemoryGo = null; return; }
  // Snapshot the cancel counter at arrival so a cancel that lands while this
  // job is still queued (e.g. behind an encode) is honored when it runs.
  if (msg.type === 'synthesize') msg._cancelBaseline = cancelCounter;
  const enqueue = () => {
    jobQueue = jobQueue.then(() => handleMessage(msg)).catch((err) => console.error('[worker] job failed:', err));
  };
  // The encoder downloads outside the queue (generation keeps running
  // meanwhile); the encode job is queued once the file is here.
  if (msg.type === 'prefetch-encoder') {
    if (needsEncoder()) fetchEncoder().catch((err) => console.warn('[encoder] download failed:', err.message));
    return;
  }
  if (msg.type === 'encode-reference' && needsEncoder()) {
    fetchEncoder().then(enqueue, (err) => postMessage({
      type: 'encode-error', requestId: msg.requestId, code: 'encode-failed',
      message: `Could not download the voice encoder: ${err.message}`,
    }));
    return;
  }
  enqueue();
};
