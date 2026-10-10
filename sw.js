// VocoLoco offline: the app's own files and the small JSON files that describe the models go to
// the network first, exactly as without this worker, so updates arrive as before; a copy of each
// answer is kept, and without a connection the copy answers. The model weights live on disk
// already (workers/model-files.js) and never pass through here.

const CACHE = 'vocoloco-app-v1';

// The app as shipped, kept at install, so the first visit already works offline later
// (tests/sw.test.js checks that every shipped file is listed)
const APP_FILES = [
  './', 'index.html', 'manifest.json', 'tailwind.css', 'mobile.css',
  'app.js', 'mobile.js', 'player.js', 'ui-dialogs.js', 'audio-postprocess.js', 'trim-editor.js',
  'text-chunker.js', 'sentence-buffer.js', 'duration-estimator.js', 'time-stretch.js',
  'lib/lamejs.min.js', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png',
  'workers/tts-worker.js', 'workers/asr-worker.js', 'workers/encoder-worker.js', 'workers/download-worker.js', 'workers/model-files.js',
  'workers/gpu-postprocess.js', 'workers/unmask-schedule.js', 'workers/voice-pitch.js',
  'workers/engine/omnivoice-engine.js', 'workers/engine/audio-decoder.js', 'workers/engine/audio-encoder.js',
  'workers/engine/whisper-engine.js', 'workers/engine/qwen2-tokenizer.js', 'workers/engine/gelu.js',
];

// The models' descriptions: config and engine manifests (a few KB; weights and tokenizer are
// stored by the workers themselves), from Hugging Face or the local mirror
const MODEL_JSON = /\/(omnivoice-config|engine\/[\w-]+)\.json$/;

self.addEventListener('install', (e) => {
  self.skipWaiting(); // nothing here depends on the version: take over right away
  e.waitUntil(caches.open(CACHE).then((c) => Promise.all(APP_FILES.map((f) => c.add(f).catch(() => {})))));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith('vocoloco-app-') && k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || req.headers.has('range')) return;
  const url = new URL(req.url);
  const appFile = url.origin === location.origin && !url.pathname.includes('/models/');
  const modelJson = (url.origin === location.origin || url.hostname === 'huggingface.co') && MODEL_JSON.test(url.pathname);
  if (appFile || modelJson) e.respondWith(networkFirst(req, url));
});

async function networkFirst(req, url) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    if (res.ok && res.type !== 'opaque') {
      const copy = res.clone();
      // one copy per file: the one under an earlier ?v= goes
      (async () => {
        for (const k of await cache.keys()) {
          const u = new URL(k.url);
          if (u.origin === url.origin && u.pathname === url.pathname && u.search !== url.search) await cache.delete(k);
        }
        await cache.put(req, copy);
      })().catch(() => {});
    }
    return res;
  } catch (err) {
    const hit = await cache.match(req) ?? await cache.match(req, { ignoreSearch: true })
      ?? (req.mode === 'navigate' ? await cache.match('./') ?? await cache.match('index.html') : undefined);
    if (hit) return hit;
    throw err;
  }
}
