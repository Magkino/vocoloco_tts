import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chunkText, CHUNK_TOKEN_BUDGET, MIN_TAIL_TOKENS, DEFAULT_PAUSE_S } from '../text-chunker.js';
import { estimateTargetTokens } from '../duration-estimator.js';

const GREEK = 'Γεια σου, Μιχάλη! Τι κάνεις? Θα ήθελα να παραιτηθώ από τη θέση μου με άμεση ισχύ. '
  + 'Τη θέση μου θα αναλάβει ο Γιώργος Ντρίτσος. Σε ευχαριστώ πολύ για τη συνεργασία.';

test('a short last sentence is merged instead of becoming its own chunk', () => {
  const chunks = chunkText(GREEK);
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].text, GREEK);
});

test('a long remainder still becomes its own chunk', () => {
  const s = 'This sentence is long enough to fill a good part of the first chunk on its own. ';
  const chunks = chunkText(s.repeat(8));
  assert.ok(chunks.length >= 2);
  assert.ok(chunks[0].estTokens <= CHUNK_TOKEN_BUDGET);
  assert.ok(chunks.at(-1).estTokens >= MIN_TAIL_TOKENS || chunks.length === 1);
});

test('a short tail after a full chunk is merged (it stays within 1.3x of the budget)', () => {
  let head = 'The first sentence goes on';
  while (estimateTargetTokens(head + ' and on.') <= CHUNK_TOKEN_BUDGET) head += ' and on';
  head += '. ';
  let tail = 'And the tail goes on';
  while (estimateTargetTokens(tail + ' and on') < MIN_TAIL_TOKENS - 1) tail += ' and on';
  tail += '.';
  assert.ok(estimateTargetTokens(head + tail) <= CHUNK_TOKEN_BUDGET * 1.3);
  assert.equal(chunkText(head + tail).length, 1);
});

test('no chunk exceeds 1.3x the budget', () => {
  const text = 'A sentence of moderate length that keeps going for a while. '.repeat(40);
  chunkText(text).forEach((c) => assert.ok(c.estTokens <= CHUNK_TOKEN_BUDGET * 1.3));
});

test('every word survives chunking', () => {
  const text = GREEK + ' ' + 'And one more sentence in English to make the text longer than the first budget allows. '.repeat(3);
  const words = (t) => t.split(/\s+/).filter(Boolean);
  assert.deepEqual(chunkText(text).flatMap((c) => words(c.text)), words(text.trim()));
});

test('pause tags split the text and carry their length', () => {
  const chunks = chunkText('Hello there. [pause] And now [pause 2s] the rest [pause 500ms]');
  assert.deepEqual(chunks.map((c) => c.text), ['Hello there.', 'And now', 'the rest']);
  assert.deepEqual(chunks.map((c) => c.pauseAfter), [DEFAULT_PAUSE_S, 2, 0.5]);
});

test('pause tags: leading, repeated, and next to bare punctuation', () => {
  const chunks = chunkText('[pause 1s] Start. [pause] [pause 1,5s]. End.');
  assert.deepEqual(chunks.map((c) => c.text), ['Start.', 'End.']);
  assert.equal(chunks[0].pauseBefore, 1);
  assert.equal(chunks[0].pauseAfter, DEFAULT_PAUSE_S + 1.5);
  assert.equal(chunks[1].pauseAfter, undefined);
  assert.deepEqual(chunkText('[pause] [pause 3s]'), []);
});

test('other bracket tags stay in the text', () => {
  assert.equal(chunkText('[laughter] You really got me.')[0].text, '[laughter] You really got me.');
});

// A slow cloned voice: a 1935 speech, 26 s reference (657 tokens) and its Whisper transcript
const SLOW = {
  refText: 'Ob du meine Arbeit für richtig hältst, ob du glaubst, dass ich fleißig gewesen bin, dass ich gearbeitet habe, dass ich mich in diesen Jahren für dich eingesetzt habe, dass ich anständig meine Zeit verwendet habe im Dienste meines Volkes, gib du jetzt deine Stimme ab. Wenn ja, dann tritt für mich ein, so wie ich für dich eingetreten bin.',
  refTokens: 657,
};
// four sentences filling a chunk at the default pace, then a short last sentence that gets merged
let TAIL_DE = 'Am Ufer saßen ein paar Angler';
for (const w of 'und warteten geduldig auf ihr Glück am stillen Wasser des alten Flusses unter den Bäumen'.split(' ')) {
  if (estimateTargetTokens(`${TAIL_DE} ${w}.`) < MIN_TAIL_TOKENS) TAIL_DE += ` ${w}`;
}
const LONG_DE = 'Der Morgenzug hatte wieder Verspätung, also gingen wir am Fluss entlang. '.repeat(4) + TAIL_DE + '.';

test('a cloned voice\'s pace sizes the chunks: none passes the 700 tokens the worker allows', () => {
  const inVoice = (t) => estimateTargetTokens(t, SLOW.refText, SLOW.refTokens);
  // at the default pace the merged last chunk is too long for this voice
  assert.ok(Math.max(...chunkText(LONG_DE).map((c) => inVoice(c.text))) > 700);
  const chunks = chunkText(LONG_DE, SLOW);
  chunks.forEach((c) => {
    assert.equal(c.estTokens, inVoice(c.text));
    assert.ok(c.estTokens <= CHUNK_TOKEN_BUDGET * 1.3);
  });
  const words = (t) => t.split(/\s+/).filter(Boolean);
  assert.deepEqual(chunks.flatMap((c) => words(c.text)), words(LONG_DE));
  // without a transcript or token count: the default pace
  assert.deepEqual(chunkText(LONG_DE, { refText: SLOW.refText }), chunkText(LONG_DE));
});
