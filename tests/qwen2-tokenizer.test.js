import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Qwen2Tokenizer } from '../workers/engine/qwen2-tokenizer.js';

// GPT-2's byte characters (printable bytes stand for themselves, the rest move to U+0100…)
const BC = [];
for (let b = 0, k = 0; b < 256; b++) BC[b] = String.fromCharCode((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 ? b : 256 + k++);
const bytes = (s) => Array.from(new TextEncoder().encode(s));
const chars = (s) => bytes(s).map((b) => BC[b]).join('');

// A tiny tokenizer.json: byte b has id b, then the merges' results in order. `vocab` on the
// result: the token ids, for the expected values
function tokenizer(merges, added = []) {
  const vocab = Object.fromEntries(BC.map((c, b) => [c, b]));
  for (const [a, b] of merges) vocab[a + b] ??= Object.keys(vocab).length;
  const t = new Qwen2Tokenizer({
    normalizer: { type: 'NFC' },
    model: { type: 'BPE', vocab, merges },
    added_tokens: added.map((content, i) => ({ id: 1000 + i, content, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true })),
  });
  return Object.assign(t, { vocab: new Map(Object.entries(vocab)) });
}

test('merges go by rank, the leftmost first on ties', () => {
  const t = tokenizer([['h', 'e'], ['t', 'he'], ['a', 'a']]);
  assert.deepEqual(t.encode('the'), [t.vocab.get('the')]);
  assert.deepEqual(t.encode('aaa'), [t.vocab.get('aa'), bytes('a')[0]]);
});

// As Hugging Face tokenizers (Oniguruma) splits: checked against the Rust library
test("contractions are split off case-insensitively, ſ folding to s", () => {
  const t = tokenizer([['t', "'"], ["'", 's'], ['S', 'x']]);
  assert.deepEqual(t.encode("it's"), [...bytes('it'), t.vocab.get("'s")]); // no t' merge across the split
  assert.deepEqual(t.encode("'Sx"), bytes("'Sx")); // pieces 'S | x: no Sx merge
  const s = chars('ſ');
  const u = tokenizer([[s[1], 'x']]); // ſ is two bytes; merging its second with x needs one piece
  assert.deepEqual(u.encode("'ſx"), bytes("'ſx"));
});

test('\\s is Unicode White_Space: U+0085 separates punctuation, U+FEFF does not', () => {
  const t = tokenizer([['!', chars('\u0085')[0]], ['!', chars('﻿')[0]]]);
  assert.deepEqual(t.encode('!\u0085!'), bytes('!\u0085!'));
  assert.equal(t.encode('!﻿!')[0], t.vocab.get('!' + chars('﻿')[0]));
});

test('added tokens are taken as written, the longest first, and text is NFC', () => {
  const t = tokenizer([], ['<|text_start|>', '<|text_end|>', '<|text|>']);
  assert.deepEqual(t.encode('<|text_start|>é<|text_end|>'), [1000, ...bytes('é'), 1001]);
  assert.deepEqual(t.encode('<|text|><|text_'), [1002, ...bytes('<|text_')]);
});
