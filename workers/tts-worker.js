/**
 * TTS Web Worker (ES Module) — runs OmniVoice inference via ONNX Runtime Web.
 * Uses @huggingface/transformers for proper Qwen2 BPE tokenization.
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.mjs';
import { AutoTokenizer, env as tfEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
import { estimateTargetTokens } from '../duration-estimator.js';
import { GpuPostProcessor } from './gpu-postprocess.js';

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

// ─── Time steps (port of _get_time_steps) ───────────────────────────────────

function getTimeSteps(tStart, tEnd, numStep, tShift) {
  const steps = [];
  for (let i = 0; i <= numStep; i++) {
    let t = tStart + (tEnd - tStart) * (i / numStep);
    t = tShift * t / (1 + (tShift - 1) * t);
    steps.push(t);
  }
  return steps;
}

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
  let fullText = refText ? refText.trim() + ' ' + text.trim() : text.trim();
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

  // Unmasking schedule
  // Python passes num_step+1 to _get_time_steps which creates linspace(0,1,num_step+2)
  const timesteps = getTimeSteps(0, 1, numStep + 1, tShift);
  const totalMask = numTargetTokens * C;
  let rem = totalMask;
  const sched = [];
  for (let s = 0; s < numStep; s++) {
    const n = s === numStep - 1 ? rem : Math.min(Math.ceil(totalMask * (timesteps[s + 1] - timesteps[s])), rem);
    sched.push(n); rem -= n;
  }

  if (gpuPostProc) {
    try { gpuPostProc.prepare(C, maxLen, V, numTargetTokens); }
    catch (e) { console.warn('[gpu-postprocess] prepare failed:', e.message); gpuPostProc.destroy(); gpuPostProc = null; }
  }

  let totalInferenceMs = 0, totalModelMs = 0, totalGpuPPMs = 0;
  for (let step = 0; step < numStep; step++) {
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
    postMessage({ type: 'progress', stage: 'generating', detail: `Step ${step + 1}/${numStep} (${stepMs.toFixed(0)}ms)` });
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

function postProcessAudio(pcm, sr) {
  const thresh = 0.005, margin = Math.floor(sr * 0.02);
  let start = 0, end = pcm.length;
  for (let i = 0; i < pcm.length; i++) if (Math.abs(pcm[i]) > thresh) { start = Math.max(0, i - margin); break; }
  for (let i = pcm.length - 1; i >= 0; i--) if (Math.abs(pcm[i]) > thresh) { end = Math.min(pcm.length, i + margin); break; }
  const out = pcm.slice(start, end);
  let peak = 0;
  for (let i = 0; i < out.length; i++) { const a = Math.abs(out[i]); if (a > peak) peak = a; }
  if (peak > 1e-6) { const s = 0.5 / peak; for (let i = 0; i < out.length; i++) out[i] *= s; }
  return out;
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

    postMessage({ type: 'progress', stage: 'loading', detail: 'Loading config...' });
    config = await (await fetch(`${modelBaseUrl}/omnivoice-config.json`)).json();

    postMessage({ type: 'progress', stage: 'loading', detail: 'Loading tokenizer (Qwen2 BPE)...' });
    tokenizer = await AutoTokenizer.from_pretrained('Gigsu/vocoloco-onnx');
    postMessage({ type: 'progress', stage: 'loading', detail: 'Tokenizer loaded.' });

    // ── Load model data ────────────────────────────────────────────────────
    postMessage({ type: 'progress', stage: 'downloading', detail: 'Loading models...' });
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

    let shardBuffers, decBuf, encBuf;

    if (allCached) {
      // All cached: load in parallel (fast)
      postMessage({ type: 'progress', stage: 'loading', detail: 'Loading from cache...' });
      const results = await Promise.all(allUrls.map(u => fetchWithProgress(u, null, null)));
      shardBuffers = results.slice(0, dataFiles.length);
      decBuf = results[dataFiles.length];
      encBuf = results[dataFiles.length + 1];
    } else {
      // Not cached: download sequentially with progress
      shardBuffers = [];
      for (let i = 0; i < dataFiles.length; i++) {
        const fname = dataFiles[i];
        postMessage({ type: 'progress', stage: 'downloading', detail: `Shard ${i + 1}/${dataFiles.length}...` });
        shardBuffers.push(await fetchWithProgress(`${modelBaseUrl}/${fname}`, (loaded, total) => {
          const lMB = (loaded / 1e6).toFixed(0), tMB = total ? (total / 1e6).toFixed(0) : '?';
          postMessage({ type: 'progress', stage: 'downloading', detail: `Shard ${i + 1}/${dataFiles.length}: ${lMB}/${tMB} MB` });
        }));
      }
      postMessage({ type: 'progress', stage: 'downloading', detail: 'Downloading decoder...' });
      decBuf = await fetchWithProgress(`${modelBaseUrl}/omnivoice-decoder.onnx`, (loaded, total) => {
        const lMB = (loaded / 1e6).toFixed(0), tMB = total ? (total / 1e6).toFixed(0) : '83';
        postMessage({ type: 'progress', stage: 'downloading', detail: `Decoder: ${lMB}/${tMB} MB` });
      });
      postMessage({ type: 'progress', stage: 'downloading', detail: 'Downloading encoder...' });
      encBuf = await fetchWithProgress(`${modelBaseUrl}/omnivoice-encoder-fixed.onnx`, (loaded, total) => {
        const lMB = (loaded / 1e6).toFixed(0), tMB = total ? (total / 1e6).toFixed(0) : '654';
        postMessage({ type: 'progress', stage: 'downloading', detail: `Encoder: ${lMB}/${tMB} MB` });
      });
    }

    const externalData = dataFiles.map((fname, i) => ({ path: fname, data: shardBuffers[i] }));

    // ── Create ONNX sessions ─────────────────────────────────────────────
    let actualBackend = 'cpu';
    postMessage({ type: 'progress', stage: 'loading', detail: 'Creating model session...' });
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

    postMessage({ type: 'progress', stage: 'loading', detail: 'Creating decoder session...' });
    decoderSession = await ort.InferenceSession.create(decBuf, { executionProviders: decEp });

    postMessage({ type: 'progress', stage: 'loading', detail: 'Creating encoder session...' });
    try {
      encoderSession = await ort.InferenceSession.create(encBuf, { executionProviders: decEp });
    } catch (e) {
      console.warn('Encoder WebGPU failed, falling back to WASM:', e.message);
      encoderSession = await ort.InferenceSession.create(encBuf, { executionProviders: ['wasm'] });
    }

    // Warm up all sessions with dummy data to compile GPU shaders
    postMessage({ type: 'progress', stage: 'loading', detail: 'Warming up...' });
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
      const dummyAudio = new Float32Array(960);
      await encoderSession.run({ input_values: new ort.Tensor('float32', dummyAudio, [1, 1, 960]) });
    } catch (e) { /* warm-up errors are non-fatal */ }

    postMessage({ type: 'ready', backend: actualBackend });
  } catch (err) {
    postMessage({ type: 'error', message: `Init failed: ${err.message}` });
  }
}

// ─── Synthesize ─────────────────────────────────────────────────────────────

async function synthesize(params) {
  const {
    text, lang = null, refAudio = null, refText = null, instruct = null,
    numStep = 20, guidanceScale = 4.0, tShift = 0.05, speed = 1.0,
    seed = null,
  } = params;

  try {
    // Use seeded PRNG for deterministic output when seed is provided
    rng = seed != null ? mulberry32(seed) : Math.random;
    // Encode reference audio for voice cloning if provided
    let refAudioTokens = null;
    let refDuration = null;
    if (refAudio && !encoderSession) {
      postMessage({ type: 'progress', stage: 'warning', detail: 'Voice cloning unavailable on this device (not enough memory for encoder)' });
    }
    if (refAudio && encoderSession) {
      postMessage({ type: 'progress', stage: 'encoding', detail: 'Encoding reference audio...' });
      const pcmF32 = new Float32Array(refAudio);
      // Clip to hop_length alignment (hop=960 for 24kHz)
      const hopLength = 960;
      const clipLen = pcmF32.length - (pcmF32.length % hopLength);
      const aligned = pcmF32.slice(0, clipLen);
      const inputTensor = new ort.Tensor('float32', aligned, [1, 1, aligned.length]);
      const encResult = await encoderSession.run({ input_values: inputTensor });
      const codesData = encResult.audio_codes.data; // BigInt64Array
      const codeDims = encResult.audio_codes.dims; // [1, 8, T]
      const T = Number(codeDims[2]);
      // Reshape to array of arrays [8][T]
      refAudioTokens = [];
      for (let c = 0; c < 8; c++) {
        const row = [];
        for (let t = 0; t < T; t++) row.push(Number(codesData[c * T + t]));
        refAudioTokens.push(row);
      }
      refDuration = aligned.length / config.sampling_rate;
      postMessage({ type: 'progress', stage: 'encoding', detail: `Encoded: ${T} tokens (${refDuration.toFixed(1)}s)` });
    }

    // Duration estimation
    const estRefText = refText || 'Nice to meet you.';
    const estRefTokens = refAudioTokens ? refAudioTokens[0].length : 25;
    let numTargetTokens = estimateTargetTokens(text, estRefText, estRefTokens, speed);

    postMessage({ type: 'progress', stage: 'preparing', detail: `Target: ${numTargetTokens} tokens` });

    const inputs = await prepareInferenceInputs(text, numTargetTokens, tokenizer, config, {
      lang, instruct, refText, refAudioTokens,
      denoise: true,
    });

    const tokens = await generateIterative(inputs, config, numStep, guidanceScale, tShift);
    const rawPcm = await decodeTokens(tokens, config.num_audio_codebook, numTargetTokens);

    postMessage({ type: 'progress', stage: 'postprocessing', detail: 'Processing audio...' });
    const pcm = postProcessAudio(rawPcm, config.sampling_rate);
    postMessage({ type: 'audio', pcm, sampleRate: config.sampling_rate }, [pcm.buffer]);
  } catch (err) {
    postMessage({ type: 'error', message: `Synthesis failed: ${err.message}\n${err.stack}` });
  }
}

// ─── Message handler ────────────────────────────────────────────────────────

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'init') await init(msg.modelBaseUrl, msg.forceCPU);
  else if (msg.type === 'synthesize') await synthesize(msg);
};
