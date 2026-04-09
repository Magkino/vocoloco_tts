/**
 * SentenceBuffer — port of assistant/sentence_buffer.py
 * Splits streamed text into complete sentences for TTS.
 */

const ABBREVS = new Set([
  'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st',
  'vs', 'etc', 'approx', 'dept', 'est', 'govt',
  'i.e', 'e.g', 'al', 'fig', 'vol', 'no',
  // German
  'z.b', 'd.h', 'usw', 'bzw', 'ca', 'evtl', 'ggf', 'inkl',
  'nr', 'tel', 'str',
]);

const SENTENCE_END = /([.!?;:])\s/g;

const MD_PATTERNS = [
  [/\*\*(.+?)\*\*/g, '$1'],
  [/\*(.+?)\*/g, '$1'],
  [/^#+\s+/gm, ''],
  [/^[-*]\s+/gm, ''],
  [/`(.+?)`/g, '$1'],
];

function cleanForTts(text) {
  for (const [pat, repl] of MD_PATTERNS) {
    text = text.replace(pat, repl);
  }
  return text.trim();
}

function isAbbreviation(textBeforeDot) {
  const parts = textBeforeDot.trimEnd().split(/\s+/);
  const word = (parts[parts.length - 1] || '').replace(/\.+$/, '');
  return ABBREVS.has(word.toLowerCase());
}

function isNumberContext(text, dotPos) {
  if (dotPos > 0 && dotPos < text.length - 1) {
    return /\d/.test(text[dotPos - 1]) && /\d/.test(text[dotPos + 1]);
  }
  return false;
}

export class SentenceBuffer {
  constructor({ minChars = 20, maxChars = 250, onSentence } = {}) {
    this.minChars = minChars;
    this.maxChars = maxChars;
    this.onSentence = onSentence || (() => {});
    this.buf = '';
  }

  addText(text) {
    this.buf += text;
    this._extract();
  }

  flush() {
    const cleaned = cleanForTts(this.buf);
    if (cleaned) this.onSentence(cleaned);
    this.buf = '';
  }

  _extract() {
    let searchStart = 0;
    while (true) {
      SENTENCE_END.lastIndex = searchStart;
      const match = SENTENCE_END.exec(this.buf);
      if (!match) break;

      const pos = match.index + match[0].length;
      const dotPos = match.index;

      if (match[1] === '.') {
        if (isAbbreviation(this.buf.slice(0, dotPos))) { searchStart = pos; continue; }
        if (isNumberContext(this.buf, dotPos)) { searchStart = pos; continue; }
      }

      const candidate = this.buf.slice(0, pos).trim();
      const cleaned = cleanForTts(candidate);

      if (cleaned.length >= this.minChars) {
        this.onSentence(cleaned);
        this.buf = this.buf.slice(pos);
        searchStart = 0;
      } else {
        searchStart = pos;
      }
    }

    // Force-split long buffers
    if (this.buf.length >= this.maxChars) {
      let splitPos = -1;
      for (const delim of ['. ', '! ', '? ', '; ', ', ', ' - ']) {
        const idx = this.buf.lastIndexOf(delim, this.maxChars);
        if (idx >= this.minChars) {
          splitPos = idx + delim.length;
          break;
        }
      }
      if (splitPos < 0) {
        const idx = this.buf.lastIndexOf(' ', this.maxChars);
        splitPos = idx > this.minChars ? idx + 1 : this.maxChars;
      }

      const cleaned = cleanForTts(this.buf.slice(0, splitPos));
      if (cleaned) this.onSentence(cleaned);
      this.buf = this.buf.slice(splitPos);
    }
  }
}
