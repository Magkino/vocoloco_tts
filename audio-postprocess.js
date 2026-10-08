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

/** First/last sample of the region louder than `thresholdDb` (10 ms frames), or null if all silent. */
export function findSpeechBounds(pcm, sr, thresholdDb = -50) {
  const frame = Math.max(1, Math.round(sr * FRAME_MS / 1000));
  let start = -1;
  for (let i = 0; i < pcm.length; i += frame) {
    if (frameDb(pcm, i, Math.min(pcm.length, i + frame)) >= thresholdDb) { start = i; break; }
  }
  if (start < 0) return null;
  let end = pcm.length;
  for (let j = pcm.length; j > start; j -= frame) {
    if (frameDb(pcm, Math.max(start, j - frame), j) >= thresholdDb) { end = j; break; }
  }
  return { start, end };
}

// Raised-cosine gain: 0 at i = 0, approaching 1 at i = n.
function fadeGain(i, n) { return 0.5 - 0.5 * Math.cos(Math.PI * i / n); }

/**
 * Trim edge silence, keeping up to `leadMs`/`trailMs` of it, and fade in/out
 * over the kept silence (at least `minFadeMs`, so a chunk whose speech starts
 * at sample 0 still gets a click-free edge).
 * Returns { pcm, speechStart, speechEnd } with bounds relative to the output.
 */
export function trimAndFade(pcm, sr, { thresholdDb = -50, leadMs = 100, trailMs = 100, minFadeMs = 10 } = {}) {
  const b = findSpeechBounds(pcm, sr, thresholdDb) || { start: 0, end: pcm.length };
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

export function peakAbs(pcm) {
  let peak = 0;
  for (let i = 0; i < pcm.length; i++) { const a = Math.abs(pcm[i]); if (a > peak) peak = a; }
  return peak;
}
