import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findSpeechBounds, trimAndFade, quietestPoint, defaultSelection, prepareReference } from '../audio-postprocess.js';
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

// `n` tone bursts of `burst` s separated by `pause` s of near-silence
function makeSpeechLike({ n = 10, burst = 1.6, pause = 0.4, lead = 1.0 } = {}) {
  const total = Math.round((lead + n * (burst + pause)) * SR);
  const pcm = new Float32Array(total).map((_, i) => 0.0005 * Math.sin(i * 1.3));
  const pauses = [];
  for (let k = 0; k < n; k++) {
    const s0 = Math.round((lead + k * (burst + pause)) * SR), s1 = s0 + Math.round(burst * SR);
    for (let i = s0; i < s1; i++) pcm[i] = 0.3 * Math.sin(2 * Math.PI * 180 * i / SR);
    pauses.push([s1, s1 + Math.round(pause * SR)]);
  }
  return { pcm, pauses };
}

test('quietest point lands inside a pause', () => {
  const { pcm, pauses } = makeSpeechLike();
  const [p0, p1] = pauses[3];
  const q = quietestPoint(pcm, SR, p0 - SR * 0.3, p1 + SR * 0.3);
  assert.ok(q >= p0 && q < p1, `${q} not in [${p0}, ${p1})`);
});

test('default selection keeps short clips whole, minus leading silence', () => {
  const { pcm } = makeSpeechLike({ n: 4 }); // ~9 s incl. 1 s lead
  const sel = defaultSelection(pcm, SR);
  assert.ok(Math.abs(sel.start - (SR - SR * 0.1)) <= 240, `start ${sel.start}`);
  assert.ok(sel.end <= pcm.length && sel.end > pcm.length - SR);
});

test('default selection of a long clip is <= 15 s and ends in a pause', () => {
  const { pcm, pauses } = makeSpeechLike({ n: 30 }); // ~61 s
  const sel = defaultSelection(pcm, SR, { maxS: 15 });
  const len = (sel.end - sel.start) / SR;
  assert.ok(len <= 15 && len >= 11, `length ${len}`);
  assert.ok(pauses.some(([a, b]) => sel.end >= a && sel.end < b), 'end is not inside a pause');
});

test('reference prep: always ends in >= 200 ms of silence, hop-aligned, click-free', () => {
  // speech runs right to the end of the clip (cut mid-word)
  const { pcm } = makeChunk({ lead: 0.3, speech: 2.0, trail: 0 });
  const out = prepareReference(pcm, SR);
  assert.equal(out.length % 960, 0);
  const b = findSpeechBounds(out, SR);
  assert.ok(out.length - b.end >= 0.2 * SR - 240, `tail ${(out.length - b.end) / SR}s`);
  assert.equal(Math.abs(out[out.length - 1]), 0);
  assert.equal(Math.abs(out[0]), 0);
});

test('reference prep drops a stop-button click after the speech', () => {
  // 2 s speech, 400 ms pause, 20 ms click, 100 ms silence
  const { pcm } = makeChunk({ lead: 0.3, speech: 2.0, trail: 0.52 });
  const clickAt = Math.round((0.3 + 2.0 + 0.4) * SR);
  for (let i = clickAt; i < clickAt + 480; i++) pcm[i] = (i % 2 ? 0.4 : -0.4);
  const out = prepareReference(pcm, SR);
  const speechEnd = findSpeechBounds(out, SR).end;
  // nothing louder than -50 dBFS after the speech: the click is gone
  let tailPeak = 0;
  for (let i = speechEnd + 240; i < out.length; i++) tailPeak = Math.max(tailPeak, Math.abs(out[i]));
  assert.ok(tailPeak < 0.01, `tail peak ${tailPeak}`);
  assert.ok(out.length / SR < 0.3 + 2.0 + 0.45, `length ${out.length / SR}s still contains the click`);
  // and the default selection stops before the click as well
  const sel = defaultSelection(pcm, SR);
  assert.ok(sel.end < clickAt, `selection end ${sel.end} >= click ${clickAt}`);
});

test('reference prep raises quiet recordings to ~0.1 RMS', () => {
  const pcm = new Float32Array(SR * 3).map((_, i) => 0.02 * Math.sin(2 * Math.PI * 200 * i / SR));
  const out = prepareReference(pcm, SR);
  let s = 0, n = 0;
  for (let i = 0; i < out.length; i++) if (Math.abs(out[i]) > 0) { s += out[i] * out[i]; n++; }
  const rms = Math.sqrt(s / n);
  assert.ok(rms > 0.09 && rms < 0.11, `rms ${rms}`);
});

test('reference transcripts get end punctuation like upstream', () => {
  assert.equal(addEndPunctuation('Hello there'), 'Hello there.');
  assert.equal(addEndPunctuation('Hello there!'), 'Hello there!');
  assert.equal(addEndPunctuation('  quoted"  '), 'quoted"');
  assert.equal(addEndPunctuation('你好'), '你好。');
  assert.equal(addEndPunctuation(''), '');
});
