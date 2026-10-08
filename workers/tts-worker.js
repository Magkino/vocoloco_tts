/**
 * TTS Web Worker (ES Module) — runs OmniVoice inference via ONNX Runtime Web.
 * Uses @huggingface/transformers for proper Qwen2 BPE tokenization.
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.mjs';
import { AutoTokenizer, env as tfEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
import { estimateTargetTokens } from '../duration-estimator.js';
import { addEndPunctuation } from '../sentence-buffer.js?v=2'; // ?v: a stale cached copy lacks this export
import { trimAndFade, peakAbs } from '../audio-postprocess.js';
import { GpuPostProcessor } from './gpu-postprocess.js';
import { unmaskSchedule } from './unmask-schedule.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

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

// ─── Cache API ─────────────────────────────────────────────────────────────

const CACHE_NAME = 'omnivoice-models-v1';

// ─── Fetch with progress + Cache API caching ──────────────────────────────

async function fetchWithProgress(url, onProgress, onCached) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    if (onCached) onCached(buf.byteLength);
    return buf;
  }

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Fetch failed: ${resp.status} for ${url}`);
  const contentLength = parseInt(resp.headers.get('Content-Length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (onProgress) onProgress(loaded, contentLength || null);
  }
  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
  const buf = result.buffer;

  // Store in Cache API — no structured clone needed, stores as a Response blob
  try {
    await cache.put(url, new Response(buf, {
      headers: { 'Content-Length': String(buf.byteLength), 'Content-Type': 'application/octet-stream' }
    }));
  } catch (e) { console.warn('Cache store failed:', e); }
  return buf;
}

// ─── Tensor helper ──────────────────────────────────────────────────────────

function T(type, data, dims) { return new ort.Tensor(type, data, dims); }

// ─── Log-softmax over a slice of a Float32Array ─────────────────────────────

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

function cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, pred, scores) {
  const gScale1 = 1 + guidanceScale;
  for (let c = 0; c < C; c++) {
    const layerScore = layerPenalty * c;
    for (let t = 0; t < numTargetTokens; t++) {
      const cOff = (c * maxLen + targetOff + t) * V;
      const uOff = ((C + c) * maxLen + t) * V;
      logSoftmaxInto(logits, cOff, V, _cLP);
      logSoftmaxInto(logits, uOff, V, _uLP);
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

let pred_buf = null, scores_buf = null;

// ─── Iterative unmasking generation loop ────────────────────────────────────

async function generateIterative(inp, cfg, numStep, guidanceScale, tShift, layerPenalty = 5.0, posTemp = 5.0) {
  const { inputIds, audioMask, totalLen, numTargetTokens, targetOff, C } = inp;
  const maskId = cfg.audio_mask_id;
  const V = cfg.audio_vocab_size;

  const condLen = totalLen;
  const uncondLen = numTargetTokens;
  const maxLen = condLen;

  // Batch input_ids: (2, C, maxLen) — cond + uncond
  const bIds = new BigInt64Array(2 * C * maxLen).fill(BigInt(maskId));
  for (let c = 0; c < C; c++)
    for (let s = 0; s < condLen; s++)
      bIds[c * maxLen + s] = inputIds[c * totalLen + s];
  for (let c = 0; c < C; c++)
    for (let t = 0; t < uncondLen; t++)
      bIds[(C + c) * maxLen + t] = inputIds[c * totalLen + targetOff + t];

  // Batch audio_mask: (2, maxLen)
  const bMask = new Uint8Array(2 * maxLen);
  for (let s = 0; s < condLen; s++) bMask[s] = audioMask[s];
  for (let t = 0; t < uncondLen; t++) bMask[maxLen + t] = 1;

  // Batch attention_mask: (2, 1, maxLen, maxLen)
  const bAttn = new Uint8Array(2 * maxLen * maxLen);
  for (let q = 0; q < condLen; q++)
    for (let k = 0; k < condLen; k++)
      bAttn[q * maxLen + k] = 1;
  for (let q = 0; q < uncondLen; q++)
    for (let k = 0; k < uncondLen; k++)
      bAttn[maxLen * maxLen + q * maxLen + k] = 1;
  for (let p = uncondLen; p < maxLen; p++)
    bAttn[maxLen * maxLen + p * maxLen + p] = 1;

  // Token state
  const tokens = new BigInt64Array(C * numTargetTokens).fill(BigInt(maskId));
  pred_buf = null; scores_buf = null;

  const sched = unmaskSchedule(numTargetTokens * C, numStep, tShift);

  if (gpuPostProc) {
    try { gpuPostProc.prepare(C, maxLen, V, numTargetTokens); }
    catch (e) { console.warn('[gpu-postprocess] prepare failed:', e.message); gpuPostProc.destroy(); gpuPostProc = null; }
  }

  let totalInferenceMs = 0, totalModelMs = 0, totalGpuPPMs = 0;
  for (let step = 0; step < numStep; step++) {
    // Let a pending 'cancel' message land, then honor it between steps
    await yieldMacrotask();
    if (isCancelRequested()) throw new CancelledError('cancelled');
    const k = sched[step];
    if (k <= 0) continue;
    const stepT0 = performance.now();

    const modelT0 = performance.now();
    const results = await mainSession.run({
      input_ids: T('int64', bIds, [2, C, maxLen]),
      audio_mask: T('bool', bMask, [2, maxLen]),
      attention_mask: T('bool', bAttn, [2, 1, maxLen, maxLen]),
    });
    const logits = results.audio_logits.data; // (2, C, maxLen, V)
    totalModelMs += performance.now() - modelT0;

    const nPos = C * numTargetTokens;
    const pred = step === 0 ? new Int32Array(nPos) : pred_buf;
    const scores = step === 0 ? new Float32Array(nPos) : scores_buf;
    if (step === 0) { pred_buf = pred; scores_buf = scores; }

    const ppT0 = performance.now();
    if (gpuPostProc) {
      try {
        await gpuPostProc.run(logits, {
          C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty
        }, pred, scores);
        // On first step, benchmark CPU too and keep whichever is faster
        if (step === 0) {
          const gpuMs = performance.now() - ppT0;
          const cpuPred = new Int32Array(nPos);
          const cpuScores = new Float32Array(nPos);
          const cpuT0 = performance.now();
          cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, cpuPred, cpuScores);
          const cpuMs = performance.now() - cpuT0;
          if (cpuMs < gpuMs) {
            console.log(`[gpu-postprocess] CPU faster (${cpuMs.toFixed(0)}ms) than GPU (${gpuMs.toFixed(0)}ms), switching to CPU`);
            // Use CPU results for this step
            pred.set(cpuPred);
            scores.set(cpuScores);
            gpuPostProc.destroy();
            gpuPostProc = null;
          } else {
            console.log(`[gpu-postprocess] GPU (${gpuMs.toFixed(0)}ms) faster than CPU (${cpuMs.toFixed(0)}ms), keeping GPU`);
          }
        }
      } catch (e) {
        console.warn('[gpu-postprocess] dispatch failed, falling back to CPU:', e.message);
        gpuPostProc.destroy();
        gpuPostProc = null;
        cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, pred, scores);
      }
    } else {
      cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, pred, scores);
    }
    totalGpuPPMs += performance.now() - ppT0;

    // Gumbel noise + mask already-unmasked (fused)
    const bigMaskId = BigInt(maskId);
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


    // Update batch inputs
    for (let c = 0; c < C; c++)
      for (let t = 0; t < numTargetTokens; t++) {
        const v = tokens[c * numTargetTokens + t];
        bIds[c * maxLen + targetOff + t] = v;
        bIds[(C + c) * maxLen + t] = v;
      }

    const stepMs = performance.now() - stepT0;
    totalInferenceMs += stepMs;
    postMessage({
      type: 'progress', stage: 'generating', jobId: activeJobId,
      step: step + 1, numStep, stepMs: Math.round(stepMs),
      detail: `Step ${step + 1}/${numStep} (${stepMs.toFixed(0)}ms)`,
    });
  }
  const jsMs = totalInferenceMs - totalModelMs;
  const ppLabel = gpuPostProc ? 'GPU-PP' : 'CPU-PP';
  console.log(`[perf] ${numStep} steps in ${totalInferenceMs.toFixed(0)}ms total | model: ${totalModelMs.toFixed(0)}ms (${(totalModelMs/numStep).toFixed(0)}ms/step) | ${ppLabel}: ${totalGpuPPMs.toFixed(0)}ms (${(totalGpuPPMs/numStep).toFixed(0)}ms/step) | JS-other: ${(jsMs - totalGpuPPMs).toFixed(0)}ms`);

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

    // GPU post-processor uses its own GPUDevice. Only useful when ONNX runs
    // on WASM (so the GPU is free). When ONNX uses WebGPU, a second device
    // causes contention that slows model inference by 5-7x.
    // We defer this decision until after we know the actual ONNX backend.

    postMessage({ type: 'progress', stage: 'loading', phase: 'config', detail: 'Loading config...' });
    config = await (await fetch(`${modelBaseUrl}/omnivoice-config.json`)).json();

    postMessage({ type: 'progress', stage: 'loading', phase: 'tokenizer', detail: 'Loading tokenizer (Qwen2 BPE)...' });
    tokenizer = await AutoTokenizer.from_pretrained('Gigsu/vocoloco-onnx');

    // ── Load model data ────────────────────────────────────────────────────
    const dataFiles = await (await fetch(`${modelBaseUrl}/omnivoice-main-manifest.json`)).json();

    // Check if all models are cached
    const cache = await caches.open(CACHE_NAME);
    const allUrls = [
      ...dataFiles.map(f => `${modelBaseUrl}/${f}`),
      `${modelBaseUrl}/omnivoice-decoder.onnx`,
      `${modelBaseUrl}/omnivoice-encoder-fixed.onnx`,
    ];
    const cacheChecks = await Promise.all(allUrls.map(u => cache.match(u)));
    const allCached = cacheChecks.every(Boolean);
    const uncachedCount = cacheChecks.filter(c => !c).length;

    // Byte-accurate download plan: cached sizes come from the stored
    // Content-Length header, uncached sizes from a HEAD pre-pass (Hugging Face
    // exposes Content-Length / X-Linked-Size through CORS). Any failure falls
    // back to file-count progress (totalBytes = null).
    let cachedBytes = 0;
    let totalBytes = null;
    try {
      const sizes = await Promise.all(allUrls.map(async (u, i) => {
        const hit = cacheChecks[i];
        if (hit) {
          const n = parseInt(hit.headers.get('Content-Length') || '0', 10);
          cachedBytes += n;
          return n;
        }
        const head = await fetch(u, { method: 'HEAD' });
        return parseInt(head.headers.get('Content-Length') || head.headers.get('X-Linked-Size') || '0', 10);
      }));
      if (sizes.every(n => n > 0)) totalBytes = sizes.reduce((a, b) => a + b, 0);
    } catch { totalBytes = null; }

    postMessage({
      type: 'plan',
      firstRun: uncachedCount === allUrls.length,
      resuming: uncachedCount > 0 && uncachedCount < allUrls.length,
      totalBytes, cachedBytes,
      fileCount: allUrls.length, filesToDownload: uncachedCount,
    });

    let shardBuffers, decBuf, encBuf;

    if (allCached) {
      // All cached: load in parallel (fast)
      postMessage({ type: 'progress', stage: 'loading', phase: 'cache-load', detail: 'Loading from cache...' });
      const results = await Promise.all(allUrls.map(u => fetchWithProgress(u, null, null)));
      shardBuffers = results.slice(0, dataFiles.length);
      decBuf = results[dataFiles.length];
      encBuf = results[dataFiles.length + 1];
    } else {
      // Not (fully) cached: download sequentially with byte-accurate progress
      let loadedBytes = cachedBytes;
      let lastPost = 0;
      const postDownload = (extra, fileIndex, fname, detail, force = false) => {
        const now = performance.now();
        if (!force && now - lastPost < 150) return;
        lastPost = now;
        postMessage({
          type: 'progress', stage: 'downloading',
          loadedBytes: loadedBytes + extra, totalBytes,
          fileIndex, fileCount: allUrls.length, file: fname,
          detail,
        });
      };
      const results = [];
      for (let i = 0; i < allUrls.length; i++) {
        const url = allUrls[i];
        const wasCached = !!cacheChecks[i];
        const fname = url.split('/').pop();
        const label = i < dataFiles.length
          ? `Shard ${i + 1}/${dataFiles.length}`
          : (i === dataFiles.length ? 'Decoder' : 'Encoder');
        if (!wasCached) postDownload(0, i + 1, fname, `${label}...`, true);
        const buf = await fetchWithProgress(url, (loaded, total) => {
          const lMB = (loaded / 1e6).toFixed(0), tMB = total ? (total / 1e6).toFixed(0) : '?';
          postDownload(loaded, i + 1, fname, `${label}: ${lMB}/${tMB} MB`);
        }, null);
        if (!wasCached) {
          loadedBytes += buf.byteLength;
          postDownload(0, i + 1, fname, `${label} complete`, true);
        }
        results.push(buf);
      }
      shardBuffers = results.slice(0, dataFiles.length);
      decBuf = results[dataFiles.length];
      encBuf = results[dataFiles.length + 1];
    }

    const externalData = dataFiles.map((fname, i) => ({ path: fname, data: shardBuffers[i] }));

    // ── Create ONNX sessions ─────────────────────────────────────────────
    let actualBackend = 'cpu';
    postMessage({ type: 'progress', stage: 'loading', phase: 'session-main', detail: 'Creating model session...' });
    if (hasWorkingGPU) {
      try {
        mainSession = await ort.InferenceSession.create(
          `${modelBaseUrl}/omnivoice-main-split.onnx`,
          { executionProviders: ['webgpu'], externalData, graphOptimizationLevel: 'all', enableCpuMemArena: true }
        );
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

    // Init GPU post-processor only when ONNX is on WASM (GPU is free)
    if (actualBackend === 'cpu' && hasWorkingGPU) {
      try {
        gpuPostProc = new GpuPostProcessor();
        await gpuPostProc.init();
        console.log('[init] GPU post-processor ready (ONNX on WASM, GPU free for post-processing)');
      } catch (e) {
        console.warn('[init] GPU post-processor unavailable, using CPU fallback:', e.message);
        gpuPostProc = null;
      }
    }

    const decEp = actualBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];

    postMessage({ type: 'progress', stage: 'loading', phase: 'session-decoder', detail: 'Creating decoder session...' });
    decoderSession = await ort.InferenceSession.create(decBuf, { executionProviders: decEp });

    // Encoder failure is non-fatal: cloning via cached tokens still works,
    // only encoding NEW reference audio is unavailable.
    postMessage({ type: 'progress', stage: 'loading', phase: 'session-encoder', detail: 'Creating encoder session...' });
    try {
      try {
        encoderSession = await ort.InferenceSession.create(encBuf, { executionProviders: decEp });
      } catch (e) {
        console.warn('Encoder WebGPU failed, falling back to WASM:', e.message);
        encoderSession = await ort.InferenceSession.create(encBuf, { executionProviders: ['wasm'] });
      }
    } catch (e) {
      console.warn('[init] Encoder unavailable on this device:', e.message);
      encoderSession = null;
      postMessage({ type: 'progress', stage: 'warning', detail: 'Voice encoder could not load — creating new cloned voices is limited on this device' });
    }

    // Warm up all sessions with dummy data to compile GPU shaders
    postMessage({ type: 'progress', stage: 'loading', phase: 'warmup', detail: 'Warming up...' });
    try {
      const dummyIds = new BigInt64Array(2 * 8 * 4).fill(1024n);
      const dummyMask = new Uint8Array(2 * 4);
      const dummyAttn = new Uint8Array(2 * 1 * 4 * 4).fill(1);
      await mainSession.run({
        input_ids: new ort.Tensor('int64', dummyIds, [2, 8, 4]),
        audio_mask: new ort.Tensor('bool', dummyMask, [2, 4]),
        attention_mask: new ort.Tensor('bool', dummyAttn, [2, 1, 4, 4]),
      });
      const dummyCodes = new BigInt64Array(8 * 2).fill(0n);
      await decoderSession.run({ audio_codes: new ort.Tensor('int64', dummyCodes, [1, 8, 2]) });
      if (encoderSession) {
        const dummyAudio = new Float32Array(960);
        await encoderSession.run({ input_values: new ort.Tensor('float32', dummyAudio, [1, 1, 960]) });
      }
    } catch (e) { /* warm-up errors are non-fatal */ }

    postMessage({ type: 'ready', backend: actualBackend, encoderAvailable: !!encoderSession });
  } catch (err) {
    postMessage({ type: 'error', message: `Init failed: ${err.message}` });
  }
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
  if (!encoderSession) {
    postMessage({ type: 'encode-error', requestId, code: 'encoder-unavailable', message: 'Voice encoder not available on this device' });
    return;
  }
  try {
    postMessage({ type: 'progress', stage: 'encoding', detail: 'Analyzing voice...' });
    const r = await encodeRefPcm(refAudio);
    postMessage({ type: 'encoded', requestId, tokens: r.tokens, tokenCount: r.tokenCount, numCodebooks: r.numCodebooks, duration: r.duration }, [r.tokens.buffer]);
  } catch (err) {
    postMessage({ type: 'encode-error', requestId, code: 'encode-failed', message: err.message });
  }
}

// ─── Synthesize ─────────────────────────────────────────────────────────────

async function synthesize(params) {
  const {
    jobId = null,
    text, lang = null, refAudio = null, refText = null, refTokens = null,
    instruct = null,
    numStep = 32, guidanceScale = 2.0, tShift = 0.1, speed = 1.0, // OmniVoice defaults
    seed = null,
    returnTokens = false, normalize = true,
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
    } else if (refAudio && encoderSession) {
      postMessage({ type: 'progress', stage: 'encoding', detail: 'Encoding reference audio...' });
      const enc = await encodeRefPcm(refAudio);
      refAudioTokens = expandTokens(enc.tokens, enc.numCodebooks);
      postMessage({ type: 'progress', stage: 'encoding', detail: `Encoded: ${enc.tokenCount} tokens (${enc.duration.toFixed(1)}s)` });
    } else if (refAudio && !encoderSession) {
      postMessage({ type: 'progress', stage: 'warning', detail: 'Voice cloning unavailable on this device (not enough memory for encoder)' });
    }

    // Duration estimation — the (refText, refTokens) pair must stay consistent:
    // mixing a real token count with the default text (or vice versa) skews the
    // estimate ~10x. estimateTargetTokens falls back to its internal defaults
    // whenever either half is missing.
    const estRefText = refAudioTokens ? refText : null;
    const estRefTokens = refAudioTokens ? refAudioTokens[0].length : null;
    let numTargetTokens = Math.min(estimateTargetTokens(text, estRefText, estRefTokens, speed), 700);

    postMessage({ type: 'progress', stage: 'preparing', detail: `Target: ${numTargetTokens} tokens` });

    const inputs = await prepareInferenceInputs(text, numTargetTokens, tokenizer, config, {
      lang, instruct, refText: refAudioTokens ? refText : null, refAudioTokens,
      denoise: true,
    });

    const tokens = await generateIterative(inputs, config, numStep, guidanceScale, tShift);

    if (isCancelRequested()) throw new CancelledError('cancelled');
    const rawPcm = await decodeTokens(tokens, C, numTargetTokens);

    postMessage({ type: 'progress', stage: 'postprocessing', detail: 'Processing audio...' });
    const { pcm, peak } = postProcessAudio(rawPcm, config.sampling_rate, normalize);

    const reply = { type: 'audio', jobId, pcm, sampleRate: config.sampling_rate, peak };
    const transfers = [pcm.buffer];
    if (returnTokens) {
      // Generated tokens are already valid reference-audio tokens — expose them
      // flat so the app can chain them as the voice reference for later chunks.
      const flat = new Int32Array(tokens.length);
      for (let i = 0; i < tokens.length; i++) flat[i] = Number(tokens[i]);
      reply.tokens = flat;
      reply.tokenCount = numTargetTokens;
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
  jobQueue = jobQueue.then(() => handleMessage(msg)).catch((err) => console.error('[worker] job failed:', err));
};
