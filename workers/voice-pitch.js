/**
 * voice-pitch.js — a reference voice's pitch, and the gender instruct it implies.
 *
 * With a reference voice and no instruct, OmniVoice sometimes switched to
 * another voice or recording (9 of 20 seeds for one Greek voice). Naming one
 * attribute that fits the reference stops that: "male" for that voice, 0 of 60.
 */

/**
 * The gender instruct for a reference voice (tts-worker.js, autoGender) from its median
 * pitch: "male" below 165 Hz, "female" up to 255 Hz (adult women speak at ~165–255 Hz),
 * none above. Higher, the pitch no longer tells: a man giving a speech in 1935 measured
 * 286–300 Hz and got "female", which pulled the clones away from his voice and made
 * them sound warped.
 */
export const GENDER_PITCH_HZ = 165;
export const GENDER_MAX_PITCH_HZ = 255;
export function genderInstruct(pitch) {
  if (pitch == null || pitch > GENDER_MAX_PITCH_HZ) return null;
  return pitch < GENDER_PITCH_HZ ? 'male' : 'female';
}

/**
 * Median pitch (Hz) of the voiced frames, or null without voiced speech.
 * Normalized autocorrelation on a 2× decimated signal, 60–400 Hz.
 */
export function voicePitch(pcm, sr = 24000) {
  const x = new Float32Array(pcm.length >> 1);
  for (let i = 0; i < x.length; i++) x[i] = 0.5 * (pcm[2 * i] + pcm[2 * i + 1]);
  const fs = sr / 2, W = 512, hop = 240;
  const minLag = Math.floor(fs / 400), maxLag = Math.ceil(fs / 60);
  let peak = 0;
  for (const v of x) peak = Math.max(peak, Math.abs(v));
  const f0s = [];
  const r = new Float32Array(maxLag + 2);
  for (let s = 0; s + W + maxLag + 1 <= x.length; s += hop) {
    let e = 0;
    for (let i = 0; i < W; i++) e += x[s + i] * x[s + i];
    if (Math.sqrt(e / W) < 0.1 * peak) continue; // quiet frame
    let best = 0;
    for (let lag = minLag; lag <= maxLag + 1; lag++) {
      let num = 0, e2 = 0;
      for (let i = 0; i < W; i++) { const b = x[s + i + lag]; num += x[s + i] * b; e2 += b * b; }
      r[lag] = num / Math.sqrt(e * e2 + 1e-12);
      if (lag <= maxLag && r[lag] > best) best = r[lag];
    }
    if (best <= 0.6) continue; // unvoiced
    // The shortest period that correlates nearly as well as the best: multiples
    // of the period correlate as well, and picking one halves the pitch
    for (let lag = minLag + 1; lag <= maxLag; lag++) {
      if (r[lag] >= 0.9 * best && r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1]) { f0s.push(fs / lag); break; }
    }
  }
  if (f0s.length < 5) return null;
  f0s.sort((a, b) => a - b);
  return f0s[f0s.length >> 1];
}

// Identifies a reference by its tokens (flat [C][T]) and the version of the measurement above:
// its pitch is kept with a saved voice under this key (app.js, persistRefInfo), so the
// reference is decoded once per voice, not once per session
export const REF_INFO_VERSION = 1;
export function refInfoKey(tokens, C) {
  let h = 0;
  for (let i = 0; i < tokens.length; i++) h = (Math.imul(h, 31) + tokens[i]) | 0;
  return `${REF_INFO_VERSION}:${tokens.length / C}:${h}`;
}
