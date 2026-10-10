/**
 * Speech recognition for the voice wizard: Whisper large-v3-turbo (the model
 * OmniVoice itself uses to transcribe references, ~100 languages) on the
 * VocoLoco engine (engine/whisper-engine.js). Downloaded on first use into the
 * app's disk store like the engine's other weights (model-files.js). The app
 * only keeps this worker alive while the voice wizard is open.
 *
 * In:  { type: 'load', base, resident }          base = the models' URL (HF or the local mirror);
 *                                                resident: keep all weights on the GPU (desktop)
 *      { type: 'transcribe', id, pcm, base, resident }   16 kHz mono Float32Array
 * Out: { type: 'progress', loadedBytes, totalBytes, phase }   phase: download | compile | transcribe
 *      { type: 'ready' }
 *      { type: 'result', id, text, lang }
 *      { type: 'error', id, code, message }
 */

import { inspectFile, openDiskFile } from './model-files.js?v=4';
import { httpSource } from './engine/omnivoice-engine.js';
import { WhisperEngine } from './engine/whisper-engine.js';

const FILES = 'engine/whisper-i8'; // .json manifest and .bin weights
const CACHE_NAME = 'vocoloco-asr-v2'; // only consulted where there is no disk store
const OLD_CACHE = 'vocoloco-asr-v1'; // the ONNX files of earlier versions

// ─── Download with progress ─────────────────────────────────────────────────

const progress = { loaded: 0, total: 0, lastPost: 0 };
function postProgress(phase, force = false) {
  const now = performance.now();
  if (!force && now - progress.lastPost < 250) return;
  progress.lastPost = now;
  postMessage({ type: 'progress', phase, loadedBytes: progress.loaded, totalBytes: progress.total || null });
}

// The weights on disk (what an interrupted download left is kept), or read from the network
// where there is no disk store
async function weights(url) {
  const info = await inspectFile(await caches.open(CACHE_NAME), url);
  progress.total = info.size;
  progress.loaded = info.cachedBytes;
  if (!info.complete) postProgress('download', true);
  const onBytes = (n) => { progress.loaded += n; postProgress('download'); };
  return await openDiskFile(url, { size: info.size, onBytes }) ?? httpSource(url);
}

// ─── Model ──────────────────────────────────────────────────────────────────

let engine = null;
let loading = null;

function load(base, resident) {
  loading ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
    if (!adapter?.features.has('shader-f16')) throw Object.assign(new Error('Automatic transcription needs WebGPU'), { code: 'no-webgpu' });
    const url = `${base}/${FILES}`;
    const r = await fetch(`${url}.json`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${FILES}.json: ${r.status}`);
    const manifest = await r.json();
    const source = await weights(`${url}.bin`);
    postProgress('compile', true);
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: ['shader-f16'],
      requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
    });
    device.lost.then((info) => { if (info.reason !== 'destroyed') console.warn('[asr] GPU device lost:', info.message); engine = null; loading = null; });
    engine = await WhisperEngine.load(device, manifest, source, { resident });
    await caches.delete(OLD_CACHE).catch(() => {});
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

// ─── Messages (one request at a time) ───────────────────────────────────────

let queue = Promise.resolve();

self.onmessage = (e) => {
  const msg = e.data;
  queue = queue.then(async () => {
    try {
      await load(msg.base, msg.resident);
      if (msg.type === 'load') {
        postMessage({ type: 'ready' });
      } else if (msg.type === 'transcribe') {
        postMessage({ type: 'progress', phase: 'transcribe' });
        const { text, lang } = await engine.transcribe(msg.pcm);
        postMessage({ type: 'result', id: msg.id, text, lang });
      }
    } catch (err) {
      console.warn('[asr]', err);
      postMessage({ type: 'error', id: msg.id ?? null, code: err.code || 'failed', message: err.message });
    }
  });
};
