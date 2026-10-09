/**
 * Model file loading for the workers: parallel ranged downloads + Cache API.
 *
 * Hugging Face's CDN serves a single connection at a few MB/s, so a file is
 * fetched as 32 MB ranges over several connections. Each range is cached as
 * its own entry as soon as it arrives: an interrupted download resumes where
 * it stopped, and no cache entry gets large enough to fail storing (very
 * large entries can). Files cached whole by earlier versions are read as-is.
 *
 * Cache layout per file URL:
 *   <url>                         whole file (earlier versions)
 *   <url>?vl-size=N&vl-part=i     part i of a file of N bytes
 *   <url>?vl-parts                {"size": N} — written once all parts are stored
 */

export const PART_BYTES = 32 * 1024 * 1024;
const MAX_CONNECTIONS = 8;
const MAX_ATTEMPTS = 5;

const partKey = (url, size, i) => `${url}${url.includes('?') ? '&' : '?'}vl-size=${size}&vl-part=${i}`;
const indexKey = (url) => `${url}${url.includes('?') ? '&' : '?'}vl-parts`;
const partCount = (size) => Math.ceil(size / PART_BYTES);
const partLen = (size, i) => Math.min(PART_BYTES, size - i * PART_BYTES);

// Connection limit shared by every download in this worker
let openConns = 0;
const connQueue = [];
async function withConnection(fn) {
  while (openConns >= MAX_CONNECTIONS) await new Promise((r) => connQueue.push(r));
  openConns++;
  try { return await fn(); } finally {
    openConns--;
    const next = connQueue.shift();
    if (next) next();
  }
}

class NoRangeSupport extends Error {}

/** Size in bytes from a HEAD request (0 if unknown). */
export async function headSize(url) {
  const r = await fetch(url, { method: 'HEAD' });
  if (!r.ok) return 0;
  return parseInt(r.headers.get('Content-Length') || r.headers.get('X-Linked-Size') || '0', 10);
}

/**
 * What the cache holds for `url`: { complete, size, cachedBytes }.
 * For files that aren't complete, `size` comes from a HEAD request and
 * `cachedBytes` counts the parts stored by an interrupted download.
 */
export async function inspectFile(cache, url) {
  const whole = await cache.match(url);
  if (whole) {
    const size = parseInt(whole.headers.get('Content-Length') || '0', 10);
    return { complete: true, size, cachedBytes: size };
  }
  const idx = await cache.match(indexKey(url));
  if (idx) {
    const { size } = await idx.json();
    return { complete: true, size, cachedBytes: size };
  }
  const size = await headSize(url).catch(() => 0);
  if (!size) return { complete: false, size: 0, cachedBytes: 0 };
  const hits = await Promise.all(Array.from({ length: partCount(size) }, (_, i) => cache.match(partKey(url, size, i))));
  const cachedBytes = hits.reduce((s, h, i) => s + (h ? partLen(size, i) : 0), 0);
  return { complete: false, size, cachedBytes };
}

/** The cached file, or null when it isn't (completely) cached. */
async function readCached(cache, url) {
  const whole = await cache.match(url);
  if (whole) return whole.arrayBuffer();
  const idx = await cache.match(indexKey(url));
  if (!idx) return null;
  const { size } = await idx.json();
  const out = new Uint8Array(size);
  try {
    await Promise.all(Array.from({ length: partCount(size) }, async (_, i) => {
      const r = await cache.match(partKey(url, size, i));
      const b = r ? new Uint8Array(await r.arrayBuffer()) : null;
      if (!b || b.byteLength !== partLen(size, i)) throw new Error(`cached part ${i} missing`);
      out.set(b, i * PART_BYTES);
    }));
  } catch (e) {
    console.warn(`[model-files] ${url}: ${e.message}, downloading again`);
    await cache.delete(indexKey(url));
    return null;
  }
  return out.buffer;
}

async function fetchRange(url, off, len, out, onBytes) {
  for (let attempt = 1; ; attempt++) {
    let got = 0;
    try {
      const resp = await fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` } });
      if (resp.status === 200) { resp.body?.cancel(); throw new NoRangeSupport(); }
      if (resp.status !== 206) throw new Error(`HTTP ${resp.status}`);
      const reader = resp.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (got + value.byteLength > len) throw new Error('range response too long');
        out.set(value, off + got);
        got += value.byteLength;
        onBytes(value.byteLength);
      }
      if (got !== len) throw new Error(`range response cut short (${got} of ${len} bytes)`);
      return;
    } catch (e) {
      if (got) onBytes(-got);
      if (e instanceof NoRangeSupport || attempt >= MAX_ATTEMPTS) throw e;
      console.warn(`[model-files] range ${off}+${len} failed (${e.message}), retry ${attempt}/${MAX_ATTEMPTS - 1}`);
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
  }
}

// Plain single-connection download (unknown size, or no range support)
async function downloadWhole(cache, url, onBytes) {
  const resp = await withConnection(() => fetch(url));
  if (!resp.ok) throw new Error(`Fetch failed: ${resp.status} for ${url}`);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onBytes(value.byteLength);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  try {
    await cache.put(url, new Response(out, { headers: { 'Content-Length': String(loaded) } }));
  } catch (e) { console.warn('[model-files] cache store failed:', e.message); }
  return out.buffer;
}

async function downloadParts(cache, url, size, onBytes) {
  const out = new Uint8Array(size);
  let rangeCount = 0;
  try {
    await Promise.all(Array.from({ length: partCount(size) }, async (_, i) => {
      const key = partKey(url, size, i), off = i * PART_BYTES, len = partLen(size, i);
      const hit = await cache.match(key);
      if (hit) {
        const b = new Uint8Array(await hit.arrayBuffer());
        if (b.byteLength === len) { out.set(b, off); return; }
      }
      await withConnection(() => fetchRange(url, off, len, out, onBytes));
      rangeCount++;
      try {
        await cache.put(key, new Response(out.subarray(off, off + len), { headers: { 'Content-Length': String(len) } }));
      } catch (e) { console.warn('[model-files] cache store failed:', e.message); }
    }));
  } catch (e) {
    if (e instanceof NoRangeSupport && rangeCount === 0) return downloadWhole(cache, url, onBytes);
    throw e;
  }
  try {
    await cache.put(indexKey(url), new Response(JSON.stringify({ size }), { headers: { 'Content-Type': 'application/json' } }));
  } catch (e) { console.warn('[model-files] cache store failed:', e.message); }
  return out.buffer;
}

/**
 * The file as an ArrayBuffer — from the cache, or downloaded (and cached).
 * `size` (from inspectFile) enables the parallel download; `onBytes(delta)`
 * reports newly downloaded bytes (negative when a failed range is retried).
 */
export async function loadFile(cache, url, { size = 0, onBytes = () => {} } = {}) {
  const cached = await readCached(cache, url);
  if (cached) return cached;
  if (!size) size = await headSize(url).catch(() => 0);
  return size ? downloadParts(cache, url, size, onBytes) : downloadWhole(cache, url, onBytes);
}
