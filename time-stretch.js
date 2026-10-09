/**
 * time-stretch.js — pitch-preserving tempo change for the speed slider.
 * OmniVoice's own speed parameter changes how much time the model gets per
 * sentence, and with cloned voices that shifted pitch and timbre as well.
 * Stretching the finished audio leaves the voice untouched.
 *
 * WSOLA (Verhelst & Roelands 1993): 40 ms Hann frames are overlap-added at a
 * 20 ms hop and read from the input at 20 ms × tempo. Each frame is shifted
 * by up to ±10 ms so its waveform lines up with how the previous frame would
 * have continued: no phase jumps, so the pitch stays the same.
 */

const FRAME_MS = 40;
const TOLERANCE_MS = 10;

// Cross-correlation of x[a ..] with x[b ..] over `len` samples, every `stride`-th
function xcorr(x, a, b, len, stride) {
  let s = 0;
  for (let i = 0; i < len; i += stride) s += x[a + i] * x[b + i];
  return s;
}

/** Returns `pcm` played `tempo` times as fast (0.85 → ~18% longer), same pitch. */
export function timeStretch(pcm, sr, tempo) {
  if (!(tempo > 0) || Math.abs(tempo - 1) < 1e-3 || pcm.length === 0) return pcm;
  const hs = Math.round(sr * FRAME_MS / 2000); // synthesis hop, half a frame
  const n = 2 * hs;
  const ha = hs * tempo;                        // analysis hop
  const tol = Math.round(sr * TOLERANCE_MS / 1000);
  const outLen = Math.round(pcm.length / tempo);
  const frames = Math.ceil(outLen / hs) + 1;

  // Zero padding: frame 0 is centred on sample 0, and every frame, search
  // window and natural continuation stays inside x
  const pad = hs + tol;
  const x = new Float32Array(pcm.length + 2 * pad + 2 * n + Math.ceil(2 * ha));
  x.set(pcm, pad);

  const win = new Float32Array(n); // periodic Hann: overlaps at n/2 sum to 1
  for (let i = 0; i < n; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);

  const y = new Float32Array((frames - 1) * hs + n);
  let start = tol; // frame 0: nominal position, no shift
  for (let m = 0; m < frames; m++) {
    const o = m * hs;
    for (let i = 0; i < n; i++) y[o + i] += win[i] * x[start + i];
    if (m + 1 === frames) break;

    // Next frame: the shift whose waveform best matches the natural
    // continuation of this one (coarse search, then refined)
    const natural = start + hs;
    const nominal = pad + Math.round((m + 1) * ha) - hs;
    let best = 0, bestScore = -Infinity;
    for (let d = -tol; d <= tol; d += 2) {
      const s = xcorr(x, natural, nominal + d, n, 2);
      if (s > bestScore) { bestScore = s; best = d; }
    }
    const coarse = best;
    bestScore = -Infinity;
    for (let d = Math.max(-tol, coarse - 1); d <= Math.min(tol, coarse + 1); d++) {
      const s = xcorr(x, natural, nominal + d, n, 1);
      if (s > bestScore) { bestScore = s; best = d; }
    }
    start = nominal + best;
  }
  return y.slice(hs, hs + outLen);
}
