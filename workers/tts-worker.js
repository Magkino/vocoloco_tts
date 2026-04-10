/**
 * TTS Web Worker (ES Module) — runs OmniVoice inference via ONNX Runtime Web.
 * Uses @huggingface/transformers for proper Qwen2 BPE tokenization.
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.mjs';
import { AutoTokenizer, env as tfEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
import { estimateTargetTokens } from '../duration-estimator.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// Maximize performance
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;

// Configure transformers.js to load tokenizer from our server
tfEnv.allowLocalModels = false;

let mainSession = null;
let decoderSession = null;
let encoderSession = null;
let tokenizer = null;
let config = null;

// ─── Cache API ─────────────────────────────────────────────────────────────

const CACHE_NAME = 'omnivoice-models-v1';

// ─── Fetch with progress + Cache API caching ──────────────────────────────

async function fetchWithProgress(url, onProgress) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    if (onProgress) onProgress(buf.byteLength, buf.byteLength);
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

function topKUnmask(scores, pred, tokens, n, k, bigMaskId) {
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

  let totalInferenceMs = 0, totalModelMs = 0;
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

    const gScale1 = 1 + guidanceScale;
    for (let c = 0; c < C; c++) {
      const layerScore = layerPenalty * c;
      for (let t = 0; t < numTargetTokens; t++) {
        const cOff = (c * maxLen + targetOff + t) * V;
        const uOff = ((C + c) * maxLen + t) * V;

        // Inline log-softmax into pre-allocated buffers
        logSoftmaxInto(logits, cOff, V, _cLP);
        logSoftmaxInto(logits, uOff, V, _uLP);

        // CFG + log-softmax + argmax fused in one pass
        // g[v] = (1+scale)*cLP[v] - scale*uLP[v]
        let mx = -Infinity;
        for (let v = 0; v < V; v++) {
          const gv = gScale1 * _cLP[v] - guidanceScale * _uLP[v];
          _g[v] = gv;
          if (gv > mx) mx = gv;
        }
        let sm = 0;
        for (let v = 0; v < V; v++) sm += Math.exp(_g[v] - mx);
        const lse = mx + Math.log(sm);

        // Find best (skip maskId)
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
    topKUnmask(scores, pred, tokens, nPos, k, bigMaskId);


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
  console.log(`[perf] ${numStep} steps in ${totalInferenceMs.toFixed(0)}ms total | model: ${totalModelMs.toFixed(0)}ms (${(totalModelMs/numStep).toFixed(0)}ms/step) | JS: ${jsMs.toFixed(0)}ms (${(jsMs/numStep).toFixed(0)}ms/step)`);

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

async function init(modelBaseUrl) {
  try {
    // Check for working WebGPU — model is too large (~2.5 GB) for WASM-only execution
    let hasWorkingGPU = false;
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        hasWorkingGPU = !!adapter;
      } catch {}
    }
    if (!hasWorkingGPU) {
      postMessage({ type: 'error', message: 'NO_WEBGPU' });
      return;
    }

    postMessage({ type: 'progress', stage: 'loading', detail: 'Loading config...' });
    config = await (await fetch(`${modelBaseUrl}/omnivoice-config.json`)).json();

    postMessage({ type: 'progress', stage: 'loading', detail: 'Loading tokenizer (Qwen2 BPE)...' });
    tokenizer = await AutoTokenizer.from_pretrained('Gigsu/vocoloco-onnx');
    postMessage({ type: 'progress', stage: 'loading', detail: 'Tokenizer loaded.' });

    const ep = [];
    if (typeof navigator !== 'undefined' && navigator.gpu) ep.push('webgpu');
    ep.push('wasm');
    console.log(`[init] Execution providers: ${ep.join(', ')}, threads: ${ort.env.wasm.numThreads}, GPU available: ${!!navigator.gpu}`);

    // Load main model (FP32, sharded)
    postMessage({ type: 'progress', stage: 'downloading', detail: 'Loading manifest...' });
    const dataFiles = await (await fetch(`${modelBaseUrl}/omnivoice-main-manifest.json`)).json();
    const externalData = [];
    for (let i = 0; i < dataFiles.length; i++) {
      const fname = dataFiles[i];
      postMessage({ type: 'progress', stage: 'downloading', detail: `Shard ${i + 1}/${dataFiles.length}...` });
      const buf = await fetchWithProgress(`${modelBaseUrl}/${fname}`, (loaded, total) => {
        const lMB = (loaded / 1e6).toFixed(0), tMB = total ? (total / 1e6).toFixed(0) : '?';
        postMessage({ type: 'progress', stage: 'downloading', detail: `Shard ${i + 1}/${dataFiles.length}: ${lMB}/${tMB} MB` });
      });
      externalData.push({ path: fname, data: buf });
    }
    postMessage({ type: 'progress', stage: 'loading', detail: 'Creating model session...' });
    mainSession = await ort.InferenceSession.create(
      `${modelBaseUrl}/omnivoice-main-split.onnx`,
      { executionProviders: ep, externalData, graphOptimizationLevel: 'all', enableCpuMemArena: true }
    );

    // Load decoder
    postMessage({ type: 'progress', stage: 'loading', detail: 'Loading decoder (~83 MB)...' });
    const decBuf = await fetchWithProgress(`${modelBaseUrl}/omnivoice-decoder.onnx`, (loaded, total) => {
      const lMB = (loaded / 1e6).toFixed(0), tMB = total ? (total / 1e6).toFixed(0) : '83';
      postMessage({ type: 'progress', stage: 'downloading', detail: `Decoder: ${lMB}/${tMB} MB` });
    });
    decoderSession = await ort.InferenceSession.create(decBuf, { executionProviders: ep });

    // Load encoder (for voice cloning)
    postMessage({ type: 'progress', stage: 'loading', detail: 'Loading encoder (~654 MB)...' });
    const encBuf = await fetchWithProgress(`${modelBaseUrl}/omnivoice-encoder-fixed.onnx`, (loaded, total) => {
      const lMB = (loaded / 1e6).toFixed(0), tMB = total ? (total / 1e6).toFixed(0) : '654';
      postMessage({ type: 'progress', stage: 'downloading', detail: `Encoder: ${lMB}/${tMB} MB` });
    });
    try {
      encoderSession = await ort.InferenceSession.create(encBuf, { executionProviders: ep });
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

    postMessage({ type: 'ready' });
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
  if (msg.type === 'init') await init(msg.modelBaseUrl);
  else if (msg.type === 'synthesize') await synthesize(msg);
};
