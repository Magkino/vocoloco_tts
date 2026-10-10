/**
 * The voice encoder on phones and tablets: turns a reference recording into the audio tokens a
 * cloned voice is made of (engine/audio-encoder.js), in a worker of its own that the app ends
 * right after. WebKit gives a worker's GPU memory back only when the worker ends (freed buffers
 * stay held while bind groups that used them wait for garbage collection), so on a phone the
 * speech model's worker is stopped while this one runs, and the other way round (app.js,
 * parkSpeech). Elsewhere tts-worker.js encodes next to the speech model.
 *
 * In:  { type: 'encode', id, pcm, base }   24 kHz mono Float32Array; base = the models' URL
 * Out: { type: 'progress', stage: 'encoder-download', loadedBytes, totalBytes, detail, done }
 *      { type: 'encoded', id, tokens, tokenCount, numCodebooks, duration }   tokens: Int32Array [C*T]
 *      { type: 'error', id, code, message }
 */

import { inspectFile, openDiskFile } from './model-files.js?v=4';
import { httpSource } from './engine/omnivoice-engine.js';
import { AudioEncoder } from './engine/audio-encoder.js';

const FILES = 'engine/encoder-f32'; // .json manifest and .bin weights
const CACHE_NAME = 'omnivoice-models-v1'; // as tts-worker.js: only consulted where there is no disk store
const HOP = 960; // samples per frame at 24 kHz

// The weights on disk (downloaded first if needed, with progress), or read from the network
// where there is no disk store
async function weights(url) {
  const info = await inspectFile(await caches.open(CACHE_NAME), url);
  let loaded = info.cachedBytes, lastPost = 0;
  const post = (force = false) => {
    const now = performance.now();
    if (!force && now - lastPost < 250) return;
    lastPost = now;
    const amount = info.size ? `${Math.min(100, Math.floor((loaded / info.size) * 100))}%` : `${(loaded / 1e6).toFixed(0)} MB`;
    postMessage({ type: 'progress', stage: 'encoder-download', loadedBytes: loaded, totalBytes: info.size || null, detail: `Downloading voice encoder ${amount}` });
  };
  if (!info.complete) post(true);
  const source = await openDiskFile(url, { size: info.size, onBytes: (n) => { loaded += n; post(); } }) ?? httpSource(url);
  if (!info.complete) postMessage({ type: 'progress', stage: 'encoder-download', done: true, detail: 'Voice encoder ready' });
  return source;
}

let encoder = null;

async function load(base) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (!adapter?.features.has('shader-f16')) throw Object.assign(new Error('The voice encoder needs WebGPU'), { code: 'encoder-unavailable' });
  const url = `${base}/${FILES}`;
  const r = await fetch(`${url}.json`, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${FILES}.json: ${r.status}`);
  const manifest = await r.json();
  const source = await weights(`${url}.bin`);
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16'],
    requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
  });
  // All weights at once: this worker ends after the voice, uploading per stage would hold them anyway
  return AudioEncoder.load(device, manifest, source, { resident: true });
}

let queue = Promise.resolve();

self.onmessage = ({ data: msg }) => {
  if (msg.type !== 'encode') return;
  queue = queue.then(async () => {
    try {
      encoder ??= await load(msg.base);
      const pcm = msg.pcm.subarray(0, msg.pcm.length - (msg.pcm.length % HOP));
      const { codes, T } = await encoder.encode(pcm);
      postMessage({ type: 'encoded', id: msg.id, tokens: codes, tokenCount: T, numCodebooks: codes.length / T, duration: pcm.length / 24000 }, [codes.buffer]);
    } catch (err) {
      console.warn('[encoder]', err);
      postMessage({ type: 'error', id: msg.id, code: err.code || (err.name === 'NotFoundError' ? 'models-gone' : 'encode-failed'), message: err.message });
    }
  });
};
