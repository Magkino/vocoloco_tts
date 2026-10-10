/**
 * Model file loading for the workers: parallel ranged downloads + Cache API,
 * or straight to disk for the engine's weights (openDiskFile, at the end).
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
  const disk = diskSupported() ? await diskState(url) : null; // see openDiskFile
  if (disk) {
    const cachedBytes = disk.done.reduce((s, d, i) => s + (d ? partLen(disk.size, i) : 0), 0);
    return { complete: cachedBytes === disk.size, size: disk.size, cachedBytes };
  }
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
    // One part at a time: reading all at once held the file twice in memory,
    // which on phones is enough to get the tab killed
    for (let i = 0; i < partCount(size); i++) {
      const r = await cache.match(partKey(url, size, i));
      const b = r ? new Uint8Array(await r.arrayBuffer()) : null;
      if (!b || b.byteLength !== partLen(size, i)) throw new Error(`cached part ${i} missing`);
      out.set(b, i * PART_BYTES);
    }
  } catch (e) {
    console.warn(`[model-files] ${url}: ${e.message}, downloading again`);
    await cache.delete(indexKey(url));
    return null;
  }
  return out.buffer;
}

// out: the whole file's buffer, or write(chunk, position) for a file on disk
async function fetchRange(url, off, len, out, onBytes) {
  const write = typeof out === 'function' ? out : (chunk, pos) => out.set(chunk, pos);
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
        // A compressed response (CDN): its HEAD size isn't the file size, so ranges can't work
        if (got + value.byteLength > len) { reader.cancel(); throw new NoRangeSupport(); }
        write(value, off + got);
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

// ─── Files on disk (origin private file system) ─────────────────────────────
// The engine's weight files live in OPFS instead of the Cache API and are read
// one tensor at a time, so no copy of the weights is in memory (phones kill
// tabs that hold the weights twice). Downloads go straight to disk as well.
//
// OPFS directory vl-models/, per file:
//   <name>         the file, written part by part at its offset
//   <name>.parts   one byte per PART_BYTES part, 1 once the part is on disk

const DISK_DIR = 'vl-models';

export function diskSupported() {
  return typeof FileSystemFileHandle !== 'undefined' && 'createSyncAccessHandle' in FileSystemFileHandle.prototype
    && !!globalThis.navigator?.storage?.getDirectory;
}

function diskName(url) {
  let h = 0;
  for (let i = 0; i < url.length; i++) h = (Math.imul(h, 31) + url.charCodeAt(i)) | 0;
  return `${new URL(url).pathname.split('/').pop().replace(/[^\w.-]/g, '_')}-${(h >>> 0).toString(16)}`;
}

const diskDir = async (create = false) => (await navigator.storage.getDirectory()).getDirectoryHandle(DISK_DIR, { create });

// { size, done } of a file on disk (done: a 0/1 byte per part), or null
async function diskState(url) {
  try {
    const dir = await diskDir();
    const name = diskName(url);
    const file = await (await dir.getFileHandle(name)).getFile();
    const done = new Uint8Array(await (await (await dir.getFileHandle(`${name}.parts`)).getFile()).arrayBuffer());
    return done.length === partCount(file.size) ? { size: file.size, done } : null;
  } catch { return null; }
}

// A file on disk as the engine's weights source: read(offset, bytes) → ArrayBuffer.
// Reads go through a File, which needs no lock: other tabs can read it at the same time.
// iOS can stop reading a File it handed out earlier although the file is still there
// (NotFoundError, seen under memory pressure): the file is then opened again from the directory.
class DiskFile {
  constructor(name, file) { this.name = name; this.file = file; this.size = file.size; }
  async read(offset, bytes) {
    let buf;
    try {
      buf = await this.file.slice(offset, offset + bytes).arrayBuffer();
    } catch {
      this.file = await (await (await diskDir()).getFileHandle(this.name)).getFile();
      buf = await this.file.slice(offset, offset + bytes).arrayBuffer();
    }
    if (buf.byteLength !== bytes) throw new Error('short read from disk');
    return buf;
  }
}

/**
 * A file complete on disk, as a weights source (DiskFile): what an interrupted
 * download left is kept, the rest is downloaded. null when OPFS isn't usable
 * here (e.g. another tab is downloading the file): read from the network then.
 */
export async function openDiskFile(url, { size = 0, onBytes = () => {} } = {}) {
  if (!diskSupported()) return null;
  try {
    const dir = await diskDir(true);
    const name = diskName(url);
    const state = await diskState(url);
    if (!state || !state.done.every(Boolean)) await downloadToDisk(dir, name, url, size || state?.size, onBytes);
    return new DiskFile(name, await (await dir.getFileHandle(name)).getFile());
  } catch (e) {
    if (e instanceof DownloadError) throw e.cause;
    console.warn(`[model-files] ${url}: no disk access (${e.message})`);
    return null;
  }
}

class DownloadError extends Error {}

// Writes the missing parts of `url` to dir/name, marking each in name.parts once it is on disk
async function downloadToDisk(dir, name, url, size, onBytes) {
  const data = await (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle();
  let marks = null;
  try {
    marks = await (await dir.getFileHandle(`${name}.parts`, { create: true })).createSyncAccessHandle();
  } catch (e) {
    await data.close();
    throw e;
  }
  try {
    size ||= await headSize(url).catch(() => 0);
    if (!size) throw new Error(`size of ${url} unknown`);
    const n = partCount(size);
    if (await marks.getSize() !== n || await data.getSize() !== size) {
      await marks.truncate(0);
      await marks.truncate(n);
      await data.truncate(size);
    }
    const done = new Uint8Array(n);
    marks.read(done, { at: 0 });
    const markDone = async (i) => {
      await data.flush();
      marks.write(Uint8Array.of(1), { at: i });
      await marks.flush();
      done[i] = 1;
    };
    const write = (chunk, pos) => data.write(chunk, { at: pos });

    // Written to disk as it arrives
    const missing = Array.from({ length: n }, (_, i) => i).filter((i) => !done[i]);
    try {
      await Promise.all(missing.map(async (i) => {
        await withConnection(() => fetchRange(url, i * PART_BYTES, partLen(size, i), write, onBytes));
        await markDone(i);
      }));
    } catch (e) {
      if (!(e instanceof NoRangeSupport)) throw e;
      const resp = await withConnection(() => fetch(url));
      if (!resp.ok) throw new Error(`Fetch failed: ${resp.status} for ${url}`);
      await streamTo(resp, write, size, onBytes);
      for (let i = 0; i < n; i++) if (!done[i]) await markDone(i);
    }
  } catch (e) {
    throw new DownloadError(e.message, { cause: e }); // a failed download is an error, not "no disk"
  } finally {
    await marks.close();
    await data.close();
  }
}

async function streamTo(resp, write, size, onBytes) {
  const reader = resp.body.getReader();
  let pos = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    write(value, pos);
    pos += value.byteLength;
    onBytes(value.byteLength);
  }
  if (pos !== size) throw new Error(`got ${pos} of ${size} bytes`);
}

/** Bytes of model files on disk (for the storage overview). */
export async function diskUsage() {
  let total = 0;
  try {
    for await (const handle of (await diskDir()).values()) {
      if (handle.kind !== 'file') continue;
      try { total += (await handle.getFile()).size; } catch { /* can't be opened right now */ }
    }
  } catch { /* nothing on disk */ }
  return total;
}

/** Whether any model file is on disk: names only, so a file that can't be opened still counts. */
export async function diskHasFiles() {
  try {
    for await (const handle of (await diskDir()).values()) if (handle.kind === 'file') return true;
  } catch { /* nothing on disk */ }
  return false;
}

/** Removes every model file on disk. */
export async function clearDisk() {
  try { await (await navigator.storage.getDirectory()).removeEntry(DISK_DIR, { recursive: true }); } catch { /* nothing there */ }
}
