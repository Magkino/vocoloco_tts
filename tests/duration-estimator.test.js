import { test } from 'node:test';
import assert from 'node:assert/strict';
import { calculateTotalWeight, estimateTargetTokens } from '../duration-estimator.js';

// Weights and token counts from upstream omnivoice/utils/duration.py (RuleDurationEstimator,
// _estimate_target_tokens) for the same strings, with the first as a 310-token reference
const REF = 'Ob du meine Arbeit für richtig hältst, ob du glaubst, dass ich pleistig gewesen bin, dass ich gearbeitet habe, dass ich mich in diesen Jahren für dich eingesetzt habe.';
const CASES = [
  [REF, 142.1, 310],
  ['Der Morgenzug hatte wieder Verspätung, also gingen wir am Fluss entlang. Das Wasser war ruhig, und die Luft roch nach Regen. Am Ufer saßen ein paar Angler und warteten geduldig auf ihr Glück.', 162.9, 355],
  ['„Straße“ – 1935 – Äpfel, Öl & Übermut… ‚ja‘ — so?', 45.5, 99],
  ['να έχεις έτοιμη τη μελέτη, να μην σε κοστίσει πολλά η μελέτη και να μπορέσεις να προχωρήσεις και να υλοποιηθεί το οποιοδήποτε έργο προκύψη.', 119.6, 260],
  ['Wait, what? You did what? Oh my goodness, that is incredible! I honestly never thought you would pull it off. Seriously though, how on earth did you manage that?', 134.6, 293],
  ['Hi 😀  non-breaking thin　ideo · ½ © ™ € £ ¿Qué? ¡Sí! 日本語 한국어 हिन्दी', 60.1, 131],
  ['Nice to meet you.', 14.1, 42],
];

test('weights and token counts match upstream', () => {
  for (const [text, weight, tokens] of CASES) {
    assert.equal(Math.round(calculateTotalWeight(text) * 10) / 10, weight, text);
    assert.equal(estimateTargetTokens(text, REF, 310), tokens, text);
  }
});

test('without a reference: upstream\'s "Nice to meet you." at 25 tokens (short estimates boosted)', () => {
  assert.equal(estimateTargetTokens('Nice to meet you.'), 39); // 50 * (25 / 50) ** (1 / 3)
  assert.equal(estimateTargetTokens('Nice to meet you.', 'only text', null), 39);
});
