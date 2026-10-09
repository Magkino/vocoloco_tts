import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chunkText, CHUNK1_TOKEN_BUDGET, CHUNK_TOKEN_BUDGET, MIN_TAIL_TOKENS } from '../text-chunker.js';
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
  assert.ok(chunks[0].estTokens <= CHUNK1_TOKEN_BUDGET);
  assert.ok(chunks.at(-1).estTokens >= MIN_TAIL_TOKENS || chunks.length === 1);
});

test('a short tail stays separate when merging would stretch the chunk too far', () => {
  // Fill chunk 1 to just under its budget, then add a tail just under MIN_TAIL_TOKENS
  let head = 'The first sentence goes on';
  while (estimateTargetTokens(head + ' and on.') <= CHUNK1_TOKEN_BUDGET) head += ' and on';
  head += '. ';
  let tail = 'And the tail goes on';
  while (estimateTargetTokens(tail + ' and on') < MIN_TAIL_TOKENS - 1) tail += ' and on';
  tail += '.';
  assert.ok(estimateTargetTokens(head + tail) > CHUNK1_TOKEN_BUDGET * 1.3);
  const chunks = chunkText(head + tail);
  assert.equal(chunks.length, 2);
  assert.equal(chunks[1].text, tail);
});

test('no chunk exceeds 1.3x its budget', () => {
  const text = 'A sentence of moderate length that keeps going for a while. '.repeat(40);
  chunkText(text).forEach((c, i) => assert.ok(c.estTokens <= (i === 0 ? CHUNK1_TOKEN_BUDGET : CHUNK_TOKEN_BUDGET) * 1.3));
});

test('every word survives chunking', () => {
  const text = GREEK + ' ' + 'And one more sentence in English to make the text longer than the first budget allows. '.repeat(3);
  const words = (t) => t.split(/\s+/).filter(Boolean);
  assert.deepEqual(chunkText(text).flatMap((c) => words(c.text)), words(text.trim()));
});
