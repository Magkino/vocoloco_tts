/**
 * Speech recognition for the voice wizard: Whisper large-v3-turbo (the model
 * OmniVoice itself uses to transcribe references, ~100 languages) via
 * transformers.js on WebGPU. Downloaded on first use, cached like the TTS
 * models (parallel ranged download, see model-files.js). The app only keeps
 * this worker alive while the voice wizard is open.
 *
 * In:  { type: 'load', host }                   host = HF or the local /models/ mirror
 *      { type: 'transcribe', id, pcm }          16 kHz mono Float32Array
 * Out: { type: 'progress', loadedBytes, totalBytes, phase }
 *      { type: 'ready', dtype }
 *      { type: 'result', id, text, lang }
 *      { type: 'error', id, code, message }
 */

import {
  AutoProcessor, WhisperForConditionalGeneration, LogitsProcessor, LogitsProcessorList, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js'; // bundles ONNX Runtime (the .web build needs a bundler)
import { inspectFile, loadFile } from './model-files.js';

const MODEL_ID = 'onnx-community/whisper-large-v3-turbo';
const CACHE_NAME = 'vocoloco-asr-v1';
const HF_HOST = 'https://huggingface.co/';

env.allowLocalModels = false;
env.useBrowserCache = false; // our fetch below caches (a second, whole-file copy would fail for 1.2 GB)

// ─── Downloads with progress ────────────────────────────────────────────────

const progress = { loaded: 0, total: 0, lastPost: 0 };
function postProgress(phase, force = false) {
  const now = performance.now();
  if (!force && now - progress.lastPost < 250) return;
  progress.lastPost = now;
  postMessage({ type: 'progress', phase, loadedBytes: progress.loaded, totalBytes: progress.total || null });
}

let cache = null;
env.fetch = async (url, init = {}) => {
  url = String(url);
  if (init.method && init.method !== 'GET') return fetch(url, init);
  cache ??= await caches.open(CACHE_NAME);
  const info = await inspectFile(cache, url);
  // missing file (404) or unknown size: hand transformers.js the real response
  if (!info.complete && !info.size) return fetch(url, init);
  const buf = await loadFile(cache, url, {
    size: info.size,
    onBytes: (n) => { progress.loaded += n; postProgress('download'); },
  });
  return new Response(buf, { headers: { 'Content-Length': String(buf.byteLength) } });
};

// ─── Model ──────────────────────────────────────────────────────────────────

let processor = null;
let model = null;
let loading = null;

// Prompt: <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>.
// The language token is left to the model, restricted to language tokens:
// that is Whisper's own language detection (transformers.js has none and
// would assume English). The first text token can't be blank / end of text.
class PromptProcessor extends LogitsProcessor {
  constructor(gc) {
    super();
    this.langIds = Object.values(gc.lang_to_id);
    this.forced = [gc.task_to_id.transcribe, gc.no_timestamps_token_id];
    this.beginSuppress = gc.begin_suppress_tokens || [];
  }
  _call(inputIds, logits) {
    for (let i = 0; i < inputIds.length; i++) {
      const n = inputIds[i].length;
      const d = logits[i].data;
      if (n === 1) keepOnly(d, this.langIds);
      else if (n <= 3) keepOnly(d, [this.forced[n - 2]]);
      else if (n === 4) for (const id of this.beginSuppress) d[id] = -Infinity;
    }
    return logits;
  }
}

function keepOnly(d, ids) {
  const keep = ids.map((id) => d[id]);
  d.fill(-Infinity);
  ids.forEach((id, j) => { d[id] = keep[j]; });
}

// 4-bit weights, fp16 compute: 536 MB in total. Without fp16 shader support
// the 4-bit / fp32 files are used (723 MB).
async function pickDtype() {
  const adapter = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
  if (!adapter) return null;
  const dt = adapter.features.has('shader-f16') ? 'q4f16' : 'q4';
  return { encoder_model: dt, decoder_model_merged: dt };
}

function load(host) {
  loading ??= (async () => {
    const dtype = await pickDtype();
    if (!dtype) throw Object.assign(new Error('Automatic transcription needs WebGPU'), { code: 'no-webgpu' });
    const local = host !== HF_HOST;
    env.remoteHost = host;
    env.remotePathTemplate = local ? '{model}/' : '{model}/resolve/{revision}/';

    // Byte total for the progress bar (the two model files are ~all of it)
    cache ??= await caches.open(CACHE_NAME);
    const base = `${host}${local ? MODEL_ID + '/' : MODEL_ID + '/resolve/main/'}onnx/`;
    const files = await Promise.all([
      `${base}encoder_model_${dtype.encoder_model}.onnx`,
      `${base}decoder_model_merged_${dtype.decoder_model_merged}.onnx`,
    ].map((u) => inspectFile(cache, u)));
    progress.total = files.reduce((s, f) => s + f.size, 0);
    progress.loaded = files.reduce((s, f) => s + f.cachedBytes, 0);
    postProgress('download', true);

    processor = await AutoProcessor.from_pretrained(MODEL_ID);
    model = await WhisperForConditionalGeneration.from_pretrained(MODEL_ID, { dtype, device: 'webgpu' });
    postProgress('compile', true);
    // Warm-up: compiles the GPU shaders so the first real request is quick
    await transcribe(new Float32Array(16000));
    return dtype;
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

async function transcribe(pcm) {
  const gc = model.generation_config;
  const { input_features } = await processor(pcm);
  const lp = new LogitsProcessorList();
  lp.push(new PromptProcessor(gc));
  const out = await model.generate({
    inputs: input_features,
    decoder_input_ids: [gc.decoder_start_token_id],
    logits_processor: lp,
    // the config suppresses <|transcribe|> on every step (it's normally part of
    // the given prompt, not generated) — that would void the forced token
    suppress_tokens: (gc.suppress_tokens || []).filter((id) => id !== gc.task_to_id.transcribe),
    max_new_tokens: 200, // a 15 s reference is well under 100 tokens
  });
  const ids = out.tolist()[0].map(Number);
  const langToken = Object.keys(gc.lang_to_id).find((k) => gc.lang_to_id[k] === ids[1]);
  const text = processor.tokenizer.decode(ids, { skip_special_tokens: true }).trim();
  return { text, lang: langToken ? langToken.slice(2, -2) : null };
}

// ─── Messages (one request at a time) ───────────────────────────────────────

let queue = Promise.resolve();

self.onmessage = (e) => {
  const msg = e.data;
  queue = queue.then(async () => {
    try {
      if (msg.type === 'load') {
        const dtype = await load(msg.host);
        postMessage({ type: 'ready', dtype });
      } else if (msg.type === 'transcribe') {
        await load(msg.host);
        postMessage({ type: 'progress', phase: 'transcribe' });
        const r = await transcribe(msg.pcm);
        postMessage({ type: 'result', id: msg.id, ...r });
      }
    } catch (err) {
      console.warn('[asr]', err);
      postMessage({ type: 'error', id: msg.id ?? null, code: err.code || 'failed', message: err.message });
    }
  });
};
