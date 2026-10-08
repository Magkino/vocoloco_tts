/**
 * audio-postprocess.js — cleanup of decoded audio before chunks are joined.
 * Mirrors OmniVoice's post-processing: edge silence is detected on 10 ms
 * frames at -50 dBFS and trimmed down to a short natural lead/tail (instead
 * of cutting right at the first/last loud sample), then faded across that
 * kept silence. Speech onsets are never attenuated, and every chunk starts
 * and ends at exactly zero, so joins are click-free.
 */

const FRAME_MS = 10;

function frameDb(pcm, start, end) {
  let sum = 0;
  for (let i = start; i < end; i++) sum += pcm[i] * pcm[i];
  const rms = Math.sqrt(sum / Math.max(1, end - start));
  return rms > 0 ? 20 * Math.log10(rms) : -Infinity;
}

/**
 * First/last sample of the region louder than `thresholdDb` (10 ms frames),
 * or null if all silent. With `minRunMs`, only sustained sound counts: short
 * isolated blips at the edges (a mic pop, the click of the stop button) are
 * left out.
 */
export function findSpeechBounds(pcm, sr, thresholdDb = -50, minRunMs = 0) {
  const frame = Math.max(1, Math.round(sr * FRAME_MS / 1000));
  const n = Math.ceil(pcm.length / frame);
  const loud = new Uint8Array(n);
  for (let f = 0; f < n; f++) loud[f] = frameDb(pcm, f * frame, Math.min(pcm.length, (f + 1) * frame)) >= thresholdDb ? 1 : 0;
  const minRun = Math.max(1, Math.round(minRunMs / FRAME_MS));
  let first = -1, last = -1;
  for (let f = 0, run = 0; f < n; f++) {
    run = loud[f] ? run + 1 : 0;
    if (run >= minRun) { first = f - run + 1; break; }
  }
  if (first < 0) return null;
  for (let f = n - 1, run = 0; f >= first; f--) {
    run = loud[f] ? run + 1 : 0;
    if (run >= minRun) { last = f + run - 1; break; }
  }
  return { start: first * frame, end: Math.min(pcm.length, (last + 1) * frame) };
}

// Raised-cosine gain: 0 at i = 0, approaching 1 at i = n.
function fadeGain(i, n) { return 0.5 - 0.5 * Math.cos(Math.PI * i / n); }

/**
 * Trim edge silence, keeping up to `leadMs`/`trailMs` of it, and fade in/out
 * over the kept silence (at least `minFadeMs`, so a chunk whose speech starts
 * at sample 0 still gets a click-free edge).
 * Returns { pcm, speechStart, speechEnd } with bounds relative to the output.
 */
export function trimAndFade(pcm, sr, { thresholdDb = -50, leadMs = 100, trailMs = 100, minFadeMs = 10, minRunMs = 0 } = {}) {
  const b = findSpeechBounds(pcm, sr, thresholdDb, minRunMs) || { start: 0, end: pcm.length };
  const start = Math.max(0, b.start - Math.round(sr * leadMs / 1000));
  const end = Math.min(pcm.length, b.end + Math.round(sr * trailMs / 1000));
  const out = pcm.slice(start, end);
  const n = out.length;
  const minFade = Math.round(sr * minFadeMs / 1000);
  const half = Math.floor(n / 2);
  const fin = Math.min(half, Math.max(minFade, b.start - start));
  const fout = Math.min(half, Math.max(minFade, end - b.end));
  for (let i = 0; i < fin; i++) out[i] *= fadeGain(i, fin);
  for (let i = 0; i < fout; i++) out[n - 1 - i] *= fadeGain(i, fout);
  return { pcm: out, speechStart: b.start - start, speechEnd: b.end - start };
}

/** Sample at the centre of the quietest 10 ms frame in [from, to) — a safe place to cut. */
export function quietestPoint(pcm, sr, from, to) {
  const frame = Math.max(1, Math.round(sr * FRAME_MS / 1000));
  from = Math.max(0, Math.floor(from));
  to = Math.min(pcm.length, Math.ceil(to));
  if (to - from <= frame) return Math.round((from + to) / 2);
  let best = from, bestDb = Infinity;
  for (let i = from; i + frame <= to; i += frame) {
    const db = frameDb(pcm, i, i + frame);
    if (db < bestDb) { bestDb = db; best = i; }
  }
  return best + (frame >> 1);
}

/**
 * Default voice-reference selection in a longer clip: skip leading silence,
 * keep the whole speech if it fits in `maxS`, otherwise end at the quietest
 * point in the last few seconds before `maxS` (like OmniVoice's
 * trim_long_audio, which splits at a pause). Returns { start, end } in samples.
 */
export function defaultSelection(pcm, sr, { minS = 3, maxS = 15 } = {}) {
  const b = findSpeechBounds(pcm, sr, -50, 100) || { start: 0, end: pcm.length };
  const start = Math.max(0, b.start - Math.round(sr * 0.1));
  const max = Math.round(sr * maxS);
  const speechEnd = Math.min(pcm.length, b.end + Math.round(sr * 0.2));
  if (speechEnd - start <= max) return { start, end: speechEnd };
  const searchFrom = start + Math.max(Math.round(sr * minS), max - Math.round(sr * 4));
  return { start, end: quietestPoint(pcm, sr, searchFrom, start + max) };
}

/**
 * Prepare a voice-cloning reference like OmniVoice does: quiet recordings are
 * raised to 0.1 RMS, edge silence is trimmed to 100 ms lead / 200 ms tail and
 * faded, and the tail is padded so the clip always ends in >= 200 ms of
 * silence. Every generated chunk continues from the end of the reference, so
 * an abrupt cut there gets reproduced as a small sound at the start of each
 * chunk. The length is padded to a multiple of `hop` so the encoder's frame
 * alignment never cuts into the audio.
 */
export const REF_PREP_VERSION = 2;

export function prepareReference(pcm, sr, { hop = 960 } = {}) {
  let src = pcm;
  let sum = 0, peak = 0;
  for (let i = 0; i < src.length; i++) { sum += src[i] * src[i]; peak = Math.max(peak, Math.abs(src[i])); }
  const rms = Math.sqrt(sum / Math.max(1, src.length));
  if (rms > 0 && rms < 0.1) {
    const g = Math.min(0.1 / rms, 0.99 / Math.max(peak, 1e-9));
    src = src.map((v) => v * g);
  }
  const { pcm: trimmed, speechEnd } = trimAndFade(src, sr, { leadMs: 100, trailMs: 200, minRunMs: 100 });
  const tail = Math.round(sr * 0.2);
  let len = Math.max(trimmed.length, speechEnd + tail);
  len = Math.ceil(len / hop) * hop;
  const out = new Float32Array(len); // zero-padded
  out.set(trimmed);
  return out;
}

export function peakAbs(pcm) {
  let peak = 0;
  for (let i = 0; i < pcm.length; i++) { const a = Math.abs(pcm[i]); if (a > peak) peak = a; }
  return peak;
}
