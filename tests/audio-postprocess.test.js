import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findSpeechBounds, trimAndFade } from '../audio-postprocess.js';
import { addEndPunctuation } from '../sentence-buffer.js';

const SR = 24000;

// `silence` s of low noise, `speech` s of a 0.3-amplitude tone, `silence` s of noise
function makeChunk({ lead = 0.5, speech = 1.0, trail = 0.5, noise = 0.001 } = {}) {
  const n = Math.round((lead + speech + trail) * SR);
  const pcm = new Float32Array(n);
  const s0 = Math.round(lead * SR), s1 = Math.round((lead + speech) * SR);
  for (let i = 0; i < n; i++) {
    pcm[i] = (i >= s0 && i < s1) ? 0.3 * Math.sin(2 * Math.PI * 220 * i / SR) : noise * Math.sin(i * 1.7);
  }
  return { pcm, s0, s1 };
}

test('speech bounds land on the loud region (10 ms frames)', () => {
  const { pcm, s0, s1 } = makeChunk();
  const b = findSpeechBounds(pcm, SR);
  assert.ok(Math.abs(b.start - s0) <= 240);
  assert.ok(Math.abs(b.end - s1) <= 240);
  assert.equal(findSpeechBounds(new Float32Array(SR), SR), null);
});

test('trim keeps ~100 ms of natural lead/tail and ends exactly at zero', () => {
  const { pcm } = makeChunk();
  const r = trimAndFade(pcm, SR);
  assert.ok(Math.abs(r.speechStart - 2400) <= 240, `lead ${r.speechStart}`);
  assert.ok(Math.abs(r.pcm.length - r.speechEnd - 2400) <= 240, `tail ${r.pcm.length - r.speechEnd}`);
  assert.equal(Math.abs(r.pcm[0]), 0);
  assert.equal(Math.abs(r.pcm[r.pcm.length - 1]), 0);
});

test('fades only cover the kept silence, never the speech onset', () => {
  const { pcm } = makeChunk();
  const r = trimAndFade(pcm, SR);
  // a full period of the tone right after the onset keeps its amplitude
  let peak = 0;
  for (let i = r.speechStart; i < r.speechStart + 240; i++) peak = Math.max(peak, Math.abs(r.pcm[i]));
  assert.ok(peak > 0.29, `onset peak ${peak}`);
});

test('speech at sample 0 still gets a short click-free fade', () => {
  const { pcm } = makeChunk({ lead: 0, trail: 0 });
  const r = trimAndFade(pcm, SR);
  assert.equal(Math.abs(r.pcm[0]), 0);
  assert.equal(Math.abs(r.pcm[r.pcm.length - 1]), 0);
  // the 10 ms fade is over: full amplitude right after it
  let peak = 0;
  for (let i = 240; i < 480; i++) peak = Math.max(peak, Math.abs(r.pcm[i]));
  assert.ok(peak > 0.29, `peak after fade ${peak}`);
});

test('reference transcripts get end punctuation like upstream', () => {
  assert.equal(addEndPunctuation('Hello there'), 'Hello there.');
  assert.equal(addEndPunctuation('Hello there!'), 'Hello there!');
  assert.equal(addEndPunctuation('  quoted"  '), 'quoted"');
  assert.equal(addEndPunctuation('你好'), '你好。');
  assert.equal(addEndPunctuation(''), '');
});
