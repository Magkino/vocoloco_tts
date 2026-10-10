/**
 * Model files into the app's disk store ahead of use, several as one download with one progress
 * (app.js: voice cloning's Whisper and voice encoder). The workers that use them later find them
 * complete and read them from disk. What an interrupted download left is kept.
 *
 * In:  { type: 'inspect', urls }    → { type: 'state', totalBytes, storedBytes }
 *      { type: 'download', urls }   → { type: 'progress', loadedBytes, totalBytes }…, then { type: 'done' }
 * Out on failure: { type: 'error', message }
 */

import { inspectFile, openDiskFile, diskSupported } from './model-files.js?v=4';

const CACHE_NAME = 'omnivoice-models-v1'; // as tts-worker.js: only consulted where there is no disk store

async function inspect(urls) {
  const cache = await caches.open(CACHE_NAME);
  return Promise.all(urls.map((u) => inspectFile(cache, u)));
}

self.onmessage = async ({ data: msg }) => {
  try {
    const files = await inspect(msg.urls);
    const totalBytes = files.reduce((s, f) => s + f.size, 0);
    let loadedBytes = files.reduce((s, f) => s + f.cachedBytes, 0);
    if (msg.type === 'inspect') {
      // no disk store: the workers read these from the network when used, nothing to keep
      postMessage({ type: 'state', totalBytes, storedBytes: diskSupported() ? loadedBytes : totalBytes });
      return;
    }
    let lastPost = 0;
    const post = (force = false) => {
      const now = performance.now();
      if (!force && now - lastPost < 200) return;
      lastPost = now;
      postMessage({ type: 'progress', loadedBytes, totalBytes });
    };
    post(true);
    if (diskSupported()) {
      for (const [i, url] of msg.urls.entries()) {
        if (files[i].complete) continue;
        await openDiskFile(url, { size: files[i].size, onBytes: (n) => { loadedBytes += n; post(); } });
      }
    }
    loadedBytes = totalBytes;
    post(true);
    postMessage({ type: 'done' });
  } catch (err) {
    postMessage({ type: 'error', message: err.message });
  }
};
