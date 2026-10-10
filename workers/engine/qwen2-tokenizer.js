// VocoLoco engine: the Qwen2 byte-level BPE tokenizer OmniVoice reads its text with, from the
// model repo's tokenizer.json (Hugging Face tokenizers format). Encoding only, without special
// tokens added, which is all the prompt needs:
//
//   added tokens (<|text_start|> …) split off as written ─ NFC ─ Qwen2's split pattern
//   ─ UTF-8 bytes as GPT-2's byte characters ─ merges by rank ─ ids
//
// Matches the Rust tokenizers library id for id (checked on edge cases, prompts, 20 languages and
// random strings).

// Qwen2's pattern. tokenizers' regex engine (Oniguruma) reads \s as Unicode White_Space and folds
// case in (?i:…); JavaScript's \s is a different set and it has no inline flags, so both are
// spelled out (ſ folds to s).
const SPLIT = /'(?:[sSſ]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\p{White_Space}\p{L}\p{N}]+[\r\n]*|\p{White_Space}*[\r\n]+|\p{White_Space}+(?!\P{White_Space})|\p{White_Space}+/gu;

// GPT-2's byte-level alphabet: printable bytes stand for themselves, the rest move to U+0100…
const BYTE_CHAR = (() => {
  const t = new Array(256);
  let k = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    t[b] = String.fromCharCode(printable ? b : 256 + k++);
  }
  return t;
})();

const CACHE_LIMIT = 20000; // encoded words kept (a long text repeats most of its words)
const PAIR = 1 << 18;      // merge key: left id · PAIR + right id (ids < 2^18)

export class Qwen2Tokenizer {
  // json: the parsed tokenizer.json. Only ids are kept: the bytes' ids and, per merge, its rank
  // and result (a vocabulary of strings would hold ~50 MB more, on phones next to the model)
  constructor(json) {
    const m = json.model, vocab = m.vocab;
    if (m.type !== 'BPE' || json.normalizer?.type !== 'NFC') throw new Error('tokenizer: not a Qwen2 BPE tokenizer.json');
    this.byteIds = BYTE_CHAR.map((c) => {
      if (vocab[c] === undefined) throw new Error('tokenizer: a byte is missing from the vocabulary');
      return vocab[c];
    });
    // merges as [a, b] or "a b" (older files; byte characters never include a space). A pair
    // listed twice keeps its last rank, as in the tokenizers library
    this.ranks = new Map();
    this.merged = new Int32Array(m.merges.length);
    m.merges.forEach((p, r) => {
      const [a, b] = Array.isArray(p) ? p : p.split(' ');
      const id = vocab[a + b];
      if (vocab[a] === undefined || vocab[b] === undefined || id === undefined) throw new Error(`tokenizer: merge ${a} ${b} is outside the vocabulary`);
      this.ranks.set(vocab[a] * PAIR + vocab[b], r);
      this.merged[r] = id;
    });
    this.added = new Map(json.added_tokens.map((a) => [a.content, a.id]));
    for (const a of json.added_tokens) {
      if (a.normalized || a.lstrip || a.rstrip || a.single_word) throw new Error(`tokenizer: added token ${a.content} needs options this tokenizer lacks`);
    }
    // longest first: at one position the longest added token wins
    const esc = [...this.added.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    this.addedRe = new RegExp(esc.join('|') || '(?!)', 'g'); // (?!): none, matches nothing
    this.cache = new Map();
    this.utf8 = new TextEncoder();
  }

  // text -> token ids
  encode(text) {
    const ids = [];
    let last = 0;
    for (const m of text.matchAll(this.addedRe)) {
      this.encodeText(text.slice(last, m.index), ids);
      ids.push(this.added.get(m[0]));
      last = m.index + m[0].length;
    }
    this.encodeText(text.slice(last), ids);
    return ids;
  }

  encodeText(text, ids) {
    if (!text) return;
    const s = text.normalize('NFC');
    let last = 0;
    for (const m of s.matchAll(SPLIT)) {
      if (m.index > last) this.word(s.slice(last, m.index), ids); // (the pattern leaves no gaps)
      this.word(m[0], ids);
      last = m.index + m[0].length;
    }
    if (last < s.length) this.word(s.slice(last), ids);
  }

  // one piece of the split: its bytes merged by rank, the lowest first (leftmost on ties)
  word(piece, ids) {
    let out = this.cache.get(piece);
    if (!out) {
      const parts = Array.from(this.utf8.encode(piece), (b) => this.byteIds[b]);
      const rank = (i) => this.ranks.get(parts[i] * PAIR + parts[i + 1]) ?? Infinity;
      const ranks = parts.slice(1).map((_, i) => rank(i)); // ranks[i]: the pair parts[i], parts[i + 1]
      for (;;) {
        let at = -1, best = Infinity;
        for (let i = 0; i < ranks.length; i++) if (ranks[i] < best) { best = ranks[i]; at = i; }
        if (at < 0) break;
        parts.splice(at, 2, this.merged[best]);
        ranks.splice(at, 1);
        if (at > 0) ranks[at - 1] = rank(at - 1);
        if (at < ranks.length) ranks[at] = rank(at);
      }
      out = parts;
      if (this.cache.size >= CACHE_LIMIT) this.cache.clear();
      this.cache.set(piece, out);
    }
    for (const id of out) ids.push(id);
  }
}
