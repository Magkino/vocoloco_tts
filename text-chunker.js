/**
 * text-chunker.js — splits input text into sentence-aligned chunks of up to
 * CHUNK_TOKEN_BUDGET (the model generates at most 700 tokens at once). The app
 * generates every chunk before playing; chunk 1 also serves as the voice
 * reference for the remaining chunks in design mode (token chaining).
 */

import { SentenceBuffer, addEndPunctuation } from './sentence-buffer.js?v=2'; // ?v: a stale cached copy lacks addEndPunctuation
import { estimateTargetTokens } from './duration-estimator.js?v=2';

// Audio token budget (25 tokens ≈ 1 s of audio). One budget for every chunk: a small
// first chunk got audio playing sooner but cost nearly as much as a full one (the
// reference runs through every step), and its short part followed by a long wait felt choppy.
export const CHUNK_TOKEN_BUDGET = 450;  // ~18 s
// A short last chunk gets a too tight duration estimate (the slowdown at the
// end of a sentence doesn't scale with text length) and its last word can be
// cut off. It's merged into the previous chunk while that stays within
// MERGE_STRETCH of its budget — OmniVoice itself doesn't split short texts.
export const MIN_TAIL_TOKENS = 100;     // ~4 s
const MERGE_STRETCH = 1.3;

const defaultEst = (text) => estimateTargetTokens(text);

// Split a single oversized unit into budget-sized pieces. Binary-searches the
// largest prefix that fits, prefers cutting at a space (falls back to a raw
// cut for unspaced scripts like CJK).
function splitOversized(unit, budget, est) {
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

// [pause], [pause 2s], [pause 1.5], [pause 500ms]: exactly that much silence. OmniVoice has
// no pause token, so the text is split there and the app inserts the silence.
export const PAUSE_TAG = /\[pause(?:\s+(\d+(?:[.,]\d+)?)\s*(ms|s)?)?\]/gi;
export const DEFAULT_PAUSE_S = 0.8;
const MAX_PAUSE_S = 10;

function pauseSeconds(num, unit) {
  if (!num) return DEFAULT_PAUSE_S;
  const v = parseFloat(num.replace(',', '.')) / (unit && unit.toLowerCase() === 'ms' ? 1000 : 1);
  return Math.min(MAX_PAUSE_S, Math.max(0, v));
}

/**
 * chunkText(fullText, { refText?, refTokens? }) → Array<{ text, estTokens, pauseAfter?, pauseBefore? }>
 * refText/refTokens: a cloned voice, whose pace sizes the chunks (estTokens in its tokens).
 * pauseAfter / pauseBefore: seconds of silence a [pause] tag asks for after a chunk
 * (or before the first one); without it the app picks a natural pause.
 * Returns [] when the text contains nothing speakable.
 */
export function chunkText(fullText, { refText = null, refTokens = null } = {}) {
  // A cloned voice's own pace (its transcript and token count, as the worker estimates):
  // at the default pace a slow voice's merged last part went past the 700 tokens the
  // worker allows and was spoken faster (a 1935 speech, 26 s reference: up to 759 tokens, cut to 700)
  const est = refText && refTokens ? (t) => estimateTargetTokens(t, addEndPunctuation(refText), refTokens) : defaultEst;
  // Normalize newlines to spaces BEFORE the worker would glue words together
  // (the worker strips \r\n without inserting a space).
  const norm = String(fullText || '').replace(/\s*[\r\n]+\s*/g, ' ').trim();
  if (!norm) return [];

  // Text between pause tags is chunked on its own; a tag's pause follows the chunk before it
  const out = [];
  let lead = 0, from = 0;
  const segments = [];
  for (const m of norm.matchAll(PAUSE_TAG)) {
    segments.push([norm.slice(from, m.index), pauseSeconds(m[1], m[2])]);
    from = m.index + m[0].length;
  }
  segments.push([norm.slice(from), 0]);
  for (const [text, pause] of segments) {
    const chunks = chunkSegment(text.replace(/^[\s.,;:!?…]+/, '').trim(), est); // a tag set before "." leaves it here
    out.push(...chunks);
    if (!pause) continue;
    if (out.length) out[out.length - 1].pauseAfter = (chunks.length ? 0 : out[out.length - 1].pauseAfter || 0) + pause;
    else lead += pause;
  }
  if (out.length && lead) out[0].pauseBefore = lead;
  return out;
}

function chunkSegment(norm, est) {
  if (!/[\p{L}\p{N}]/u.test(norm)) return []; // e.g. the "." after a pause tag

  // Sentence collection (abbreviation-aware, markdown-cleaned). The trailing
  // space lets a terminal "…end." match the sentence-end regex; flush() picks
  // up any unterminated remainder.
  const units = [];
  const sb = new SentenceBuffer({ minChars: 20, maxChars: 280, onSentence: (s) => units.push(s) });
  sb.addText(norm + ' ');
  sb.flush();
  if (units.length === 0) return [];

  // Greedy packing against the budget. A single unit over it is split down to fit.
  const chunks = [];
  const queue = [...units];
  let current = '';
  const budgetFor = () => CHUNK_TOKEN_BUDGET;
  while (queue.length) {
    const unit = queue[0];
    const candidate = current ? current + ' ' + unit : unit;
    if (est(candidate) <= budgetFor()) {
      current = candidate;
      queue.shift();
    } else if (!current) {
      queue.shift();
      queue.unshift(...splitOversized(unit, budgetFor(), est));
    } else {
      chunks.push(current);
      current = '';
    }
  }
  if (current) chunks.push(current);

  if (chunks.length > 1) {
    const last = chunks[chunks.length - 1], prev = chunks[chunks.length - 2];
    const prevBudget = CHUNK_TOKEN_BUDGET;
    const merged = prev + ' ' + last;
    if (est(last) < MIN_TAIL_TOKENS && est(merged) <= prevBudget * MERGE_STRETCH) chunks.splice(-2, 2, merged);
  }

  return chunks
    .map((text) => text.trim())
    .filter(Boolean)
    .map((text) => ({ text, estTokens: est(text) }));
}
