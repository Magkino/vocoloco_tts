/**
 * text-chunker.js — splits input text into sentence-aligned chunks for
 * streamed generation. Chunk 1 is kept small so first audio arrives fast and
 * its generated tokens stay short enough to serve as the voice reference for
 * the remaining chunks (token chaining).
 */

import { SentenceBuffer } from './sentence-buffer.js';
import { estimateTargetTokens } from './duration-estimator.js';

// Audio token budgets (25 tokens ≈ 1 s of audio).
export const CHUNK1_TOKEN_BUDGET = 220; // ~9 s — fast first audio + valid chain reference
export const CHUNK_TOKEN_BUDGET = 450;  // ~18 s — amortizes per-chunk overhead

const est = (text) => estimateTargetTokens(text);

// Split a single oversized unit into budget-sized pieces. Binary-searches the
// largest prefix that fits, prefers cutting at a space (falls back to a raw
// cut for unspaced scripts like CJK).
function splitOversized(unit, budget) {
  const parts = [];
  let rest = unit.trim();
  while (rest && est(rest) > budget) {
    let lo = 1, hi = rest.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (est(rest.slice(0, mid)) <= budget) lo = mid; else hi = mid - 1;
    }
    let cut = Math.max(1, lo);
    const sp = rest.lastIndexOf(' ', cut);
    if (sp >= Math.floor(cut * 0.5)) cut = sp + 1;
    parts.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) parts.push(rest);
  return parts.filter(Boolean);
}

/**
 * chunkText(fullText) → Array<{ text: string, estTokens: number }>
 * Returns [] when the text contains nothing speakable.
 */
export function chunkText(fullText) {
  // Normalize newlines to spaces BEFORE the worker would glue words together
  // (the worker strips \r\n without inserting a space).
  const norm = String(fullText || '').replace(/\s*[\r\n]+\s*/g, ' ').trim();
  if (!norm) return [];

  // Sentence collection (abbreviation-aware, markdown-cleaned). The trailing
  // space lets a terminal "…end." match the sentence-end regex; flush() picks
  // up any unterminated remainder.
  const units = [];
  const sb = new SentenceBuffer({ minChars: 20, maxChars: 280, onSentence: (s) => units.push(s) });
  sb.addText(norm + ' ');
  sb.flush();
  if (units.length === 0) return [];

  // Greedy packing: chunk 1 against the small budget, the rest against the
  // large one. A single unit over the current budget is split down to fit.
  const chunks = [];
  const queue = [...units];
  let current = '';
  const budgetFor = () => (chunks.length === 0 ? CHUNK1_TOKEN_BUDGET : CHUNK_TOKEN_BUDGET);
  while (queue.length) {
    const unit = queue[0];
    const candidate = current ? current + ' ' + unit : unit;
    if (est(candidate) <= budgetFor()) {
      current = candidate;
      queue.shift();
    } else if (!current) {
      queue.shift();
      queue.unshift(...splitOversized(unit, budgetFor()));
    } else {
      chunks.push(current);
      current = '';
    }
  }
  if (current) chunks.push(current);

  return chunks
    .map((text) => text.trim())
    .filter(Boolean)
    .map((text) => ({ text, estTokens: est(text) }));
}
