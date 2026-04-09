/**
 * RuleDurationEstimator — port of omnivoice/utils/duration.py
 * Estimates audio token count from text using character phonetic weights.
 */

const WEIGHTS = {
  cjk: 3.0, hangul: 2.5, kana: 2.2, ethiopic: 3.0, yi: 3.0,
  indic: 1.8, thai_lao: 1.5, khmer_myanmar: 1.8,
  arabic: 1.5, hebrew: 1.5,
  latin: 1.0, cyrillic: 1.0, greek: 1.0, armenian: 1.0, georgian: 1.0,
  punctuation: 0.5, space: 0.2, digit: 3.5, mark: 0.0, default: 1.0,
};

// (end_codepoint, type_key) — binary search table
const RANGES = [
  [0x02AF, 'latin'], [0x03FF, 'greek'], [0x052F, 'cyrillic'],
  [0x058F, 'armenian'], [0x05FF, 'hebrew'],
  [0x077F, 'arabic'], [0x089F, 'arabic'], [0x08FF, 'arabic'],
  [0x097F, 'indic'], [0x09FF, 'indic'], [0x0A7F, 'indic'],
  [0x0AFF, 'indic'], [0x0B7F, 'indic'], [0x0BFF, 'indic'],
  [0x0C7F, 'indic'], [0x0CFF, 'indic'], [0x0D7F, 'indic'],
  [0x0DFF, 'indic'], [0x0EFF, 'thai_lao'], [0x0FFF, 'indic'],
  [0x109F, 'khmer_myanmar'], [0x10FF, 'georgian'],
  [0x11FF, 'hangul'], [0x137F, 'ethiopic'], [0x139F, 'ethiopic'],
  [0x13FF, 'default'], [0x167F, 'default'], [0x169F, 'default'],
  [0x16FF, 'default'], [0x171F, 'default'], [0x173F, 'default'],
  [0x175F, 'default'], [0x177F, 'default'], [0x17FF, 'khmer_myanmar'],
  [0x18AF, 'default'], [0x18FF, 'default'],
  [0x194F, 'indic'], [0x19DF, 'indic'], [0x19FF, 'khmer_myanmar'],
  [0x1A1F, 'indic'], [0x1AAF, 'indic'], [0x1B7F, 'indic'],
  [0x1BBF, 'indic'], [0x1BFF, 'indic'], [0x1C4F, 'indic'],
  [0x1C7F, 'indic'], [0x1C8F, 'cyrillic'], [0x1CBF, 'georgian'],
  [0x1CCF, 'indic'], [0x1CFF, 'indic'], [0x1D7F, 'latin'],
  [0x1DBF, 'latin'], [0x1DFF, 'default'], [0x1EFF, 'latin'],
  [0x309F, 'kana'], [0x30FF, 'kana'], [0x312F, 'cjk'],
  [0x318F, 'hangul'], [0x9FFF, 'cjk'], [0xA4CF, 'yi'],
  [0xA4FF, 'default'], [0xA63F, 'default'], [0xA69F, 'cyrillic'],
  [0xA6FF, 'default'], [0xA7FF, 'latin'], [0xA82F, 'indic'],
  [0xA87F, 'default'], [0xA8DF, 'indic'], [0xA8FF, 'indic'],
  [0xA92F, 'indic'], [0xA95F, 'indic'], [0xA97F, 'hangul'],
  [0xA9DF, 'indic'], [0xA9FF, 'khmer_myanmar'], [0xAA5F, 'indic'],
  [0xAA7F, 'khmer_myanmar'], [0xAADF, 'indic'], [0xAAFF, 'indic'],
  [0xAB2F, 'ethiopic'], [0xAB6F, 'latin'], [0xABBF, 'default'],
  [0xABFF, 'indic'], [0xD7AF, 'hangul'], [0xFAFF, 'cjk'],
  [0xFDFF, 'arabic'], [0xFE6F, 'default'], [0xFEFF, 'arabic'],
  [0xFFEF, 'latin'],
];

const BREAKPOINTS = RANGES.map(r => r[0]);

function bisectLeft(arr, val) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < val) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// Unicode general category detection (simplified for common cases)
function getCharCategory(code) {
  // Combining marks (Mn, Mc, Me)
  if ((code >= 0x0300 && code <= 0x036F) || // Combining Diacritical Marks
      (code >= 0x0483 && code <= 0x0489) ||
      (code >= 0x0591 && code <= 0x05BD) ||
      (code >= 0x064B && code <= 0x065F) ||
      (code >= 0x0900 && code <= 0x0903) ||
      (code >= 0x093A && code <= 0x094F) ||
      (code >= 0x0951 && code <= 0x0957) ||
      (code >= 0x0962 && code <= 0x0963) ||
      (code >= 0xFE20 && code <= 0xFE2F)) return 'M';
  // Punctuation
  if ((code >= 0x0021 && code <= 0x002F) ||
      (code >= 0x003A && code <= 0x0040) ||
      (code >= 0x005B && code <= 0x0060) ||
      (code >= 0x007B && code <= 0x007E) ||
      (code >= 0x2000 && code <= 0x206F) ||
      (code >= 0x3000 && code <= 0x303F)) return 'P';
  // Digits
  if (code >= 0x0030 && code <= 0x0039) return 'N';
  // Symbols
  if ((code >= 0x00A0 && code <= 0x00BF) ||
      (code >= 0x2100 && code <= 0x27FF)) return 'S';
  // Space separators
  if (code === 0x00A0 || code === 0x2000 || code === 0x2001 ||
      code === 0x2002 || code === 0x2003 || code === 0x3000) return 'Z';
  return 'L'; // Letter (default)
}

function getCharWeight(char) {
  const code = char.codePointAt(0);
  if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) return WEIGHTS.latin;
  if (code === 32) return WEIGHTS.space;
  if (code === 0x0640) return WEIGHTS.mark; // Arabic Tatweel

  const cat = getCharCategory(code);
  if (cat === 'M') return WEIGHTS.mark;
  if (cat === 'P' || cat === 'S') return WEIGHTS.punctuation;
  if (cat === 'Z') return WEIGHTS.space;
  if (cat === 'N') return WEIGHTS.digit;

  const idx = bisectLeft(BREAKPOINTS, code);
  if (idx < RANGES.length) {
    return WEIGHTS[RANGES[idx][1]] || WEIGHTS.default;
  }
  if (code > 0x20000) return WEIGHTS.cjk;
  return WEIGHTS.default;
}

export function calculateTotalWeight(text) {
  let total = 0;
  for (const char of text) {
    total += getCharWeight(char);
  }
  return total;
}

export function estimateDuration(targetText, refText, refDuration, lowThreshold = 50, boostStrength = 3) {
  if (refDuration <= 0 || !refText) return 0;
  const refWeight = calculateTotalWeight(refText);
  if (refWeight === 0) return 0;
  const speedFactor = refWeight / refDuration;
  const targetWeight = calculateTotalWeight(targetText);
  const estimated = targetWeight / speedFactor;
  if (lowThreshold !== null && estimated < lowThreshold) {
    const alpha = 1.0 / boostStrength;
    return lowThreshold * Math.pow(estimated / lowThreshold, alpha);
  }
  return estimated;
}

export function estimateTargetTokens(text, refText = null, numRefAudioTokens = null, speed = 1.0) {
  let rText = refText;
  let rTokens = numRefAudioTokens;
  if (rTokens === null || rText === null || rText.length === 0) {
    rText = 'Nice to meet you.';
    rTokens = 25;
  }
  let est = estimateDuration(text, rText, rTokens);
  if (speed > 0 && speed !== 1.0) est = est / speed;
  return Math.max(1, Math.round(est));
}
