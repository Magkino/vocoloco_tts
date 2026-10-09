import { test } from 'node:test';
import assert from 'node:assert/strict';
import { timeStretch } from '../time-stretch.js';
import { findSpeechBounds } from '../audio-postprocess.js';

const SR = 24000;

// Voiced-speech stand-in: f0 plus 5 decaying harmonics
function voiced(f0, seconds, amp = 0.3) {
  const pcm = new Float32Array(Math.round(seconds * SR));
  for (let i = 0; i < pcm.length; i++) {
    let v = 0;
    for (let h = 1; h <= 6; h++) v += Math.sin(2 * Math.PI * f0 * h * i / SR + h) / h;
    pcm[i] = amp * v / 2;
  }
  return pcm;
}

// f0 from the first autocorrelation peak (60–400 Hz) within 90% of the
// highest one (multiples of the period score the same), parabolic interpolation
function pitch(pcm, from, len) {
  const r = (lag) => { let s = 0; for (let i = 0; i < len; i++) s += pcm[from + i] * pcm[from + i + lag]; return s; };
  const lo = Math.floor(SR / 400), hi = Math.ceil(SR / 60);
  const rs = [];
  for (let lag = lo - 1; lag <= hi + 1; lag++) rs[lag] = r(lag);
  const max = Math.max(...rs.slice(lo, hi + 1));
  let best = lo;
  while (best < hi && !(rs[best] >= 0.9 * max && rs[best] >= rs[best - 1] && rs[best] >= rs[best + 1])) best++;
  const a = rs[best - 1], b = rs[best], c = rs[best + 1];
  return SR / (best + 0.5 * (a - c) / (a - 2 * b + c));
}

function rms(pcm, from, len) {
  let s = 0;
  for (let i = 0; i < len; i++) s += pcm[from + i] ** 2;
  return Math.sqrt(s / len);
}

test('tempo 1 returns the audio unchanged', () => {
  const pcm = voiced(120, 0.5);
  assert.equal(timeStretch(pcm, SR, 1), pcm);
});

test('length scales with 1 / tempo', () => {
  const pcm = voiced(120, 2);
  for (const tempo of [0.85, 0.95, 1.1]) {
    assert.equal(timeStretch(pcm, SR, tempo).length, Math.round(pcm.length / tempo));
  }
});

test('pitch stays the same at every slider position', () => {
  for (const f0 of [85, 120, 210]) {
    const pcm = voiced(f0, 2);
    for (const tempo of [0.85, 0.9, 0.95, 1.05, 1.1]) {
      const out = timeStretch(pcm, SR, tempo);
      const f = pitch(out, Math.round(out.length / 2) - SR / 4, SR / 4);
      assert.ok(Math.abs(f - f0) / f0 < 0.005, `${f0} Hz at ${tempo}: ${f.toFixed(2)} Hz`);
    }
  }
});

test('frames join without jumps: level and sample-to-sample steps match the input', () => {
  const pcm = voiced(120, 2);
  const maxStep = (a) => { let m = 0; for (let i = 1; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - a[i - 1])); return m; };
  const inStep = maxStep(pcm.subarray(SR / 10, pcm.length - SR / 10));
  for (const tempo of [0.85, 1.1]) {
    const out = timeStretch(pcm, SR, tempo);
    const body = out.subarray(SR / 10, out.length - SR / 10);
    assert.ok(Math.abs(rms(body, 0, body.length) / rms(pcm, SR / 10, pcm.length - SR / 5) - 1) < 0.03, `level at ${tempo}`);
    assert.ok(maxStep(body) < inStep * 1.1, `steps at ${tempo}: ${maxStep(body)} vs ${inStep}`);
  }
});

test('pauses and words move with the tempo', () => {
  // 0.5 s silence, 1 s voice, 0.5 s silence
  const pcm = new Float32Array(2 * SR);
  pcm.set(voiced(150, 1), SR / 2);
  for (const tempo of [0.85, 1.1]) {
    const b = findSpeechBounds(timeStretch(pcm, SR, tempo), SR);
    assert.ok(Math.abs(b.start - SR / 2 / tempo) <= SR * 0.03, `onset at ${tempo}: ${b.start}`);
    assert.ok(Math.abs(b.end - b.start - SR / tempo) <= SR * 0.06, `voice length at ${tempo}: ${b.end - b.start}`);
  }
});
