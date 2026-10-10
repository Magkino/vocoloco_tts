import { test } from 'node:test';
import assert from 'node:assert/strict';
import { voicePitch, refInfoKey, genderInstruct, GENDER_PITCH_HZ, GENDER_MAX_PITCH_HZ } from '../workers/voice-pitch.js';

const SR = 24000;

// A "voice": harmonics of f0 with a formant-like spectral tilt, syllable-like bursts
function voice(f0, { seconds = 2, tilt = 1, noise = 0.002, seed = 1 } = {}) {
  let s = seed;
  const rand = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32) - 0.5;
  const pcm = new Float32Array(Math.round(seconds * SR));
  for (let i = 0; i < pcm.length; i++) {
    const env = 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * i / SR);
    let v = 0;
    for (let h = 1; h * f0 < 8000; h++) v += Math.sin(2 * Math.PI * f0 * h * i / SR + h) / h ** tilt;
    pcm[i] = 0.2 * env * v + noise * rand();
  }
  return pcm;
}

test('pitch: the fundamental of low and high voices, nothing for silence', () => {
  for (const f0 of [95, 140, 210, 270]) {
    const p = voicePitch(voice(f0, { tilt: 1.5 }));
    assert.ok(Math.abs(p - f0) / f0 < 0.03, `${f0} Hz measured ${p}`);
  }
  assert.equal(voicePitch(new Float32Array(SR)), null);
});

test('refInfoKey: tokens and frame count, as the worker measured them per [C][T]', () => {
  const C = 8, T = 5;
  const tokens = Int32Array.from({ length: C * T }, (_, i) => (i * 37) % 1025);
  const key = refInfoKey(tokens, C);
  assert.match(key, /^\d+:5:-?\d+$/);
  assert.equal(refInfoKey(Array.from(tokens), C), key); // stored voices may hold plain arrays
  const other = tokens.slice(); other[C * T - 1] ^= 1;
  assert.notEqual(refInfoKey(other, C), key);
});

test('gender instruct only where the pitch tells it', () => {
  assert.equal(genderInstruct(null), null);
  assert.equal(genderInstruct(110), 'male');
  assert.equal(genderInstruct(GENDER_PITCH_HZ - 1), 'male');
  assert.equal(genderInstruct(210), 'female');
  assert.equal(genderInstruct(GENDER_MAX_PITCH_HZ), 'female');
  // a man giving a speech in 1935, measured by voicePitch: 286–300 Hz
  assert.equal(genderInstruct(286), null);
  assert.equal(genderInstruct(300), null);
});
