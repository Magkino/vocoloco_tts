// VocoLoco engine: the audio encoder for voice cloning (HiggsAudioV2 tokenizer: HuBERT + semantic
// conv encoder, DAC acoustic encoder, fc, residual VQ) on WebGPU. Weights: a JSON manifest and one
// binary file.
//
// f32 throughout: the codes are argmaxes and should match upstream's. Activations are time-major
// [T][C] and every convolution is a GEMM over (tap, input channel), like the decoder's, here with
// strides, groups and windows: a conv computes output rows [olo, olo + M) of a signal whose rows
// are held from global row `base` in a buffer, reading zeros outside the signal. The DAC encoder
// (at 24 kHz) and HuBERT's conv stack (at 16 kHz) run in windows of output frames; the input rows
// each window needs are worked out layer by layer (need()), so the result does not depend on the
// window size. Weights are uploaded per stage and dropped after it unless `resident`.
//
//   pcm 24 kHz ─ resample (CPU, torchaudio's kernel) ─ HuBERT ─ mean of 13 hidden states, every
//   2nd frame ─ semantic encoder ─┐
//   pcm 24 kHz ─ DAC encoder ─────┴─ concat ─ fc ─ 8 × (project_in, nearest code, residual) ─ codes

import { compile, gemmWGSL, parseCfg, dims } from './omnivoice-engine.js';
import { SNAKE } from './audio-decoder.js';
import { GELU } from './gelu.js';

const UNI = 256;
const DAC_WINDOW = 64;     // DAC frames per window of its first blocks (2.56 s; bounds memory at 24 kHz × 64 channels)
const DAC_SPLIT = 3;       // DAC blocks run in windows; the rest over the whole clip
const HUBERT_WINDOW = 256; // HuBERT frames per window of its conv stack (5.1 s)
const HEAD_GROUP = 4;      // attention heads per pass (bounds the score buffer)
const UPLOAD_BATCH = 64 << 20; // weight bytes written before waiting (upload)
const ATT_SCORE = '16x16r4v1k16', ATT_PV = '16x16r4v1k16';

// input rows a conv needs for output rows [lo, hi), within the input signal [0, len)
const need = ([lo, hi], { k, s = 1, d = 1, p = 0 }, len) => [Math.max(0, lo * s - p), Math.min(len, (hi - 1) * s - p + (k - 1) * d + 1)];
const span = ([lo, hi]) => hi - lo;

const ACT = `${GELU}
fn expm1_(x: f32) -> f32 { // Kahan: accurate near 0
  let u = exp(x);
  if (u == 1.0) { return x; }
  let um = u - 1.0;
  if (um == -1.0) { return -1.0; }
  return um * x / log(u);
}
fn elu(y: vec4<f32>) -> vec4<f32> {
  return select(vec4<f32>(expm1_(y.x), expm1_(y.y), expm1_(y.z), expm1_(y.w)), y, y > vec4<f32>(0.0));
}
${SNAKE}`;

// Epilogues: OUT = f(y) (or y + OUT for rmw), and optionally SN = act(new OUT) next to it.
const EPI = {
  raw: { add: false, out: 'y' }, gelu: { add: false, out: 'gelu(y)' }, elu: { add: false, out: 'elu(y)' },
  snake: { add: false, out: 'snake(y, AL[bc])' }, rmw: { add: true, out: 'y' },
  rawElu: { add: false, out: 'y', sn: 'elu(y)' }, rmwElu: { add: true, out: 'y', sn: 'elu(y)' },
  rawSnake: { add: false, out: 'y', sn: 'snake(y, AL[bc])' }, rmwSnake: { add: true, out: 'y', sn: 'snake(y, AL[bc])' },
};

// OUT[olo + m − obase][ocol + z·N + n] = B[z·N + n] + Σ_(j, ci) W[z][j·cin + ci][n] · X[(olo + m)·stride + j·dil − pad − xbase][z·cin + ci]
// (X rows outside [0, xlen) read as 0). Grid (N / NT, ⌈M / MT⌉, groups); KT divides cin.
// Variants as the engine's gemmWGSL (same FMAs in the same order, so bit-identical): 'wreg' reads
// the weights straight from memory, 'reg' both operands (no shared memory, no barriers).
// wT 'f16' (Whisper): f16 weights converted to f32 as they are read, so the math is the f32 path's;
// 'i8': u32 = 4 int8 along N, scales S [K/32][N/4] vec4<f16> at binding 5, q · s exact in f32 (one
// weight matrix: wz = 0). epi: an epilogue like EPI's (Whisper's stem adds the positions after the GELU).
function convWGSL({ TM, TN, RM, VN, KT, v: variant = '' }, epiName, { wT = 'f32', epi = EPI[epiName] } = {}) {
  // weight row k, vec4 column col, as vec4<f32>
  const w = (k, col) => {
    const i = `wB + (${k}) * N4 + ${col}`;
    if (wT === 'i8') return `wq(${i}, ((${k}) / 32u) * N4 + ${col})`;
    return wT === 'f32' ? `W[${i}]` : `vec4<f32>(W[${i}])`;
  };
  const MT = TM * RM, NT = TN * VN * 4, THREADS = TM * TN, KT4 = KT / 4, NT4 = NT / 4;
  const reg = variant === 'reg', wreg = reg || variant === 'wreg';
  let b = 5;
  const decls = [];
  if (wT === 'i8') decls.push(`@group(0) @binding(${b++}) var<storage, read> S: array<vec4<f16>>;
fn wq(i: u32, s: u32) -> vec4<f32> {
  let p = W[i];
  let q = vec4<i32>(bitcast<i32>(p << 24u) >> 24u, bitcast<i32>(p << 16u) >> 24u, bitcast<i32>(p << 8u) >> 24u, bitcast<i32>(p) >> 24u);
  return vec4<f32>(q) * vec4<f32>(S[s]);
}`);
  if (epi.sn) decls.push(`@group(0) @binding(${b++}) var<storage, read_write> SN: array<vec4<f32>>;`);
  if ((epi.out + (epi.sn || '')).includes('AL')) decls.push(`@group(0) @binding(${b++}) var<storage, read> AL: array<vec4<f32>>;`);
  let inner = '', rows = '';
  for (let i = 0; i < RM; i++) {
    if (reg) {
      rows += `
    let g${i} = (u.olo + i32(m0 + ly + ${i * TM}u)) * i32(u.stride) + shift;
    let ok${i} = m0 + ly + ${i * TM}u < u.M && g${i} >= 0 && g${i} < i32(u.xlen);
    let xb${i} = select(0u, u32(g${i} - u.xbase) * u.xs4 + xc + ci4, ok${i});`;
      inner += `\n      let a${i} = select(vec4<f32>(0.0), X[xb${i} + k4], ok${i});`;
    } else inner += `\n      let a${i} = As[(ly + ${i * TM}u) * ${KT4}u + k4];`;
  }
  for (let kk = 0; kk < 4; kk++) {
    const ch = 'xyzw'[kk];
    for (let v = 0; v < VN; v++) {
      inner += wreg
        ? `\n      let b${kk}_${v} = ${w(`k0 + k4 * 4u + ${kk}u`, `n0 / 4u + ${v * TN}u + lx`)};`
        : `\n      let b${kk}_${v} = Ws[(k4 * 4u + ${kk}u) * ${NT4}u + ${v * TN}u + lx];`;
    }
    for (let i = 0; i < RM; i++) for (let v = 0; v < VN; v++) inner += `\n      c${i}_${v} = fma(vec4<f32>(a${i}.${ch}), b${kk}_${v}, c${i}_${v});`;
  }
  let decl = '', store = '';
  for (let i = 0; i < RM; i++) for (let v = 0; v < VN; v++) {
    decl += `\n  var c${i}_${v} = vec4<f32>(0.0);`;
    store += `
  {
    let m = m0 + ly + ${i * TM}u;
    if (m < u.M) {
      let bc = wg.z * N4 + n0 / 4u + ${v * TN}u + lx;
      let o = u32(u.olo + i32(m) - u.obase) * u.os4 + u.ocol4 + bc;
      var y = c${i}_${v} + B[bc];${epi.add ? '\n      y += OUT[o];' : ''}
      OUT[o] = ${epi.out};${epi.sn ? `\n      SN[o] = ${epi.sn};` : ''}
    }
  }`;
  }
  const loadA = reg ? '' : `
    for (var e = li; e < ${MT * KT4}u; e += ${THREADS}u) {
      let r = e / ${KT4}u; let c = e % ${KT4}u;
      let g = (u.olo + i32(m0 + r)) * i32(u.stride) + shift;
      var v = vec4<f32>(0.0);
      if (m0 + r < u.M && g >= 0 && g < i32(u.xlen)) { v = X[u32(g - u.xbase) * u.xs4 + xc + ci4 + c]; }
      As[e] = v;
    }`;
  const loadW = wreg ? '' : `
    for (var e = li; e < ${KT * NT4}u; e += ${THREADS}u) {
      let r = e / ${NT4}u; let c = e % ${NT4}u;
      Ws[e] = ${w('k0 + r', 'n0 / 4u + c')};
    }`;
  const sync = reg ? '' : '\n    workgroupBarrier();';
  return `${wT === 'f32' ? '' : 'enable f16;\n'}struct U { M: u32, N: u32, K: u32, cin: u32, xs4: u32, stride: u32, dil: u32, pad: u32,
  olo: i32, xbase: i32, xlen: u32, obase: i32, os4: u32, ocol4: u32, wz: u32, xz4: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: ${wT === 'i8' ? 'array<u32>' : `array<vec4<${wT}>>`};
@group(0) @binding(3) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> OUT: array<vec4<f32>>;
${decls.join('\n')}
${ACT}
${reg ? '' : `var<workgroup> As: array<vec4<f32>, ${MT * KT4}>;`}
${wreg ? '' : `var<workgroup> Ws: array<vec4<f32>, ${KT * NT4}>;`}
@compute @workgroup_size(${TN}, ${TM})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * ${MT}u; let n0 = wg.x * ${NT}u; let lx = lid.x; let ly = lid.y;
  let N4 = u.N / 4u; let wB = wg.z * u.wz; let xc = wg.z * u.xz4;${decl}
  for (var k0 = 0u; k0 < u.K; k0 += ${KT}u) {
    let shift = i32((k0 / u.cin) * u.dil) - i32(u.pad); let ci4 = (k0 % u.cin) / 4u;${rows}${loadA}${loadW}${sync}
    for (var k4 = 0u; k4 < ${KT4}u; k4++) {${inner}
    }${sync}
  }${store}
}`;
}
// Conv tiles by output width: 'w128' for N a multiple of its NT (most layers), 'w64' otherwise,
// 'g48' for the positional conv's 48-channel groups (KT must divide cin = 48). load({ tiles })
// overrides them.
const CONV_TILES = { w128: '16x16r4v1k16', w64: '16x16r4v1k16', g48: '16x4r4v1k16' };
// timed per device (tuneTiles): Macs pick the shared-memory tiles, the iPhone 'reg' (−16 % there)
const TILE_CANDIDATES = { w128: ['16x16r4v1k16', '8x16r4v2k32:reg'], w64: ['16x16r4v1k16', '8x16r4v1k32:reg'] };

// First convolutions (one input channel): OUT[olo + m − obase][n] = epi(B[n] + Σ_j W[j][n] · X[(olo + m)·stride + j − pad])
// 'rawSnake' (DAC: OUT = y, SN = snake(y)) or 'gn' (HuBERT: OUT = gelu(GroupNorm(y)), per-channel
// (mean, 1/σ) in ST, affine GG, GB). One thread per (row, 4 channels).
const CONV1 = (epi) => `struct U { M: u32, N: u32, taps: u32, stride: u32, pad: u32, olo: i32, xlen: u32, obase: i32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> OUT: array<vec4<f32>>;
${epi === 'gn' ? `@group(0) @binding(5) var<storage, read> ST: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> GG: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> GB: array<vec4<f32>>;` : `@group(0) @binding(5) var<storage, read_write> SN: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> AL: array<vec4<f32>>;`}
${ACT}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N4 = u.N / 4u; let i = gid.y * 65535u * 64u + gid.x;
  if (i >= u.M * N4) { return; }
  let m = i / N4; let bc = i % N4;
  var acc = vec4<f32>(0.0);
  for (var j = 0u; j < u.taps; j++) {
    let g = (u.olo + i32(m)) * i32(u.stride) + i32(j) - i32(u.pad);
    if (g >= 0 && g < i32(u.xlen)) { acc = fma(vec4<f32>(X[g]), W[j * N4 + bc], acc); }
  }
  let y = acc + B[bc]; let o = u32(u.olo + i32(m) - u.obase) * N4 + bc;
  ${epi === 'gn' ? `let s0 = ST[bc * 4u]; let s1 = ST[bc * 4u + 1u]; let s2 = ST[bc * 4u + 2u]; let s3 = ST[bc * 4u + 3u];
  let n = (y - vec4<f32>(s0.x, s1.x, s2.x, s3.x)) * vec4<f32>(s0.y, s1.y, s2.y, s3.y);
  OUT[o] = gelu(n * GG[bc] + GB[bc]);` : 'OUT[o] = y; SN[o] = snake(y, AL[bc]);'}
}`;

// GroupNorm statistics of HuBERT's first conv (one group per channel, over all F frames, conv
// recomputed): pass 0 writes the mean, pass 1 1/sqrt(var + eps) (centred, biased). Grid (channels).
const GNSTAT = (taps, stride, N, eps) => `struct U { F: u32, ph: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> ST: array<vec2<f32>>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let c = wg.x; let mean = ST[c].x;
  var acc = 0.0;
  for (var f = t; f < u.F; f += 256u) {
    var y = 0.0;
    for (var j = 0u; j < ${taps}u; j++) { y = fma(X[f * ${stride}u + j], W[j * ${N}u + c], y); }
    if (u.ph == 0u) { acc += y; } else { let d = y - mean; acc += d * d; }
  }
  red[t] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] += red[t + s]; } workgroupBarrier(); }
  if (t == 0u) {
    if (u.ph == 0u) { ST[c] = vec2<f32>(red[0] / f32(u.F), 0.0); } else { ST[c].y = inverseSqrt(red[0] / f32(u.F) + ${eps}); }
  }
}`;

// LayerNorm over rows of W in place: X = LN(X [+ A]) · g + b, then optionally HS = X ('set') or
// HS += X ('acc'): the running sum of HuBERT's hidden states. One workgroup (W / 4 threads) per row.
const LN = (W, eps, { add = false, hs = null } = {}) => `struct U { rows: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> G: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Bb: array<vec4<f32>>;
${add ? '@group(0) @binding(4) var<storage, read> A: array<vec4<f32>>;' : ''}
${hs ? `@group(0) @binding(${add ? 5 : 4}) var<storage, read_write> HS: array<vec4<f32>>;` : ''}
var<workgroup> red: array<f32, ${W / 4}>;
fn total(t: u32) -> f32 {
  for (var s = ${2 ** Math.floor(Math.log2(W / 4 - 1))}u; s > 0u; s >>= 1u) {
    if (t < s && t + s < ${W / 4}u) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(${W / 4})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  if (wg.x >= u.rows) { return; }
  let i = wg.x * ${W / 4}u + t;
  let x = X[i]${add ? ' + A[i]' : ''};
  red[t] = x.x + x.y + x.z + x.w;
  workgroupBarrier();
  let mean = total(t) / ${W}.0;
  let d = x - vec4<f32>(mean);
  red[t] = dot(d, d);
  workgroupBarrier();
  let v = total(t) / ${W}.0;
  let y = d * inverseSqrt(v + ${eps}) * G[t] + Bb[t];
  X[i] = y;${hs === 'set' ? '\n  HS[i] = y;' : hs === 'acc' ? '\n  HS[i] += y;' : ''}
}`;

// [M][3·NH·HD] (q | k | v) -> Q [NH][Mp][HD], Kt [NH][HD][Mp], V [NH][Mp][HD]. Grid (M, 3·NH), HD / 4 threads.
const QKVSPLIT = (NH, HD) => `struct U { Mp: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Kt: array<f32>;
@group(0) @binding(4) var<storage, read_write> V: array<vec4<f32>>;
@compute @workgroup_size(${HD / 4})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let m = wg.x; let s = wg.y % ${NH}u; let part = wg.y / ${NH}u;
  let x = qkv[m * ${3 * NH * HD / 4}u + part * ${NH * HD / 4}u + s * ${HD / 4}u + t];
  if (part == 0u) { Q[(s * u.Mp + m) * ${HD / 4}u + t] = x; }
  else if (part == 2u) { V[(s * u.Mp + m) * ${HD / 4}u + t] = x; }
  else { for (var j = 0u; j < 4u; j++) { Kt[(s * ${HD}u + 4u * t + j) * u.Mp + m] = x[j]; } }
}`;

// semantic input: every 2nd row of the hidden-state sum, / 13 (torch's mean). Grid (T), W / 4 threads.
const DOWNMEAN = (W, n) => `struct U { T: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> HS: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;
@compute @workgroup_size(${W / 4})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  if (wg.x >= u.T) { return; }
  Y[wg.x * ${W / 4}u + t] = HS[2u * wg.x * ${W / 4}u + t] / ${n}.0;
}`;

// One residual-VQ stage per frame: x = R·Win + bin; code = argmax −(|x|² − 2 x·e + |e|²) (lowest
// index on ties); R −= E[code]·Wout + bout. R [T][H] in place, codes [C][T]. Grid (T), 256 threads.
const RVQ = (H, D, V) => `struct U { T: u32, c: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> R: array<f32>;
@group(0) @binding(2) var<storage, read> Win: array<f32>;
@group(0) @binding(3) var<storage, read> bin: array<f32>;
@group(0) @binding(4) var<storage, read> E: array<f32>;
@group(0) @binding(5) var<storage, read> esq: array<f32>;
@group(0) @binding(6) var<storage, read> Wout: array<f32>;
@group(0) @binding(7) var<storage, read> bout: array<f32>;
@group(0) @binding(8) var<storage, read_write> codes: array<i32>;
var<workgroup> r: array<f32, ${H}>;
var<workgroup> part: array<f32, 256>;
var<workgroup> x: array<f32, ${D}>;
var<workgroup> bd: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x;
  for (var i = t; i < ${H}u; i += 256u) { r[i] = R[row * ${H}u + i]; }
  workgroupBarrier();
  // project_in: output k = t % D, input slice t / D of ${256 / D} slices
  let k = t % ${D}u; let sl = t / ${D}u;
  var acc = 0.0;
  for (var i = sl * ${H / (256 / D)}u; i < (sl + 1u) * ${H / (256 / D)}u; i++) { acc = fma(r[i], Win[i * ${D}u + k], acc); }
  part[t] = acc;
  workgroupBarrier();
  if (t < ${D}u) {
    var s = 0.0;
    for (var j = 0u; j < ${256 / D}u; j++) { s += part[j * ${D}u + t]; }
    x[t] = s + bin[t];
  }
  workgroupBarrier();
  var sq = 0.0;
  for (var j = 0u; j < ${D}u; j++) { sq = fma(x[j], x[j], sq); }
  var best = -3.0e38; var bidx = 0u;
  for (var e = t; e < ${V}u; e += 256u) {
    var dt = 0.0;
    for (var j = 0u; j < ${D}u; j++) { dt = fma(x[j], E[e * ${D}u + j], dt); }
    let dist = -((sq - 2.0 * dt) + esq[e]);
    if (dist > best) { best = dist; bidx = e; }
  }
  bd[t] = best; bi[t] = bidx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s && (bd[t + s] > bd[t] || (bd[t + s] == bd[t] && bi[t + s] < bi[t]))) { bd[t] = bd[t + s]; bi[t] = bi[t + s]; }
    workgroupBarrier();
  }
  let code = bi[0];
  for (var i = t; i < ${H}u; i += 256u) {
    var q = 0.0;
    for (var j = 0u; j < ${D}u; j++) { q = fma(E[code * ${D}u + j], Wout[j * ${H}u + i], q); }
    R[row * ${H}u + i] = r[i] - (q + bout[i]);
  }
  if (t == 0u) { codes[u.c * u.T + row] = i32(code); }
}`;

export class AudioEncoder {
  // manifest: parsed encoder-f32.json; source: { read(offset, bytes) } over its .bin. resident: keep
  // every weight on the GPU between encodes (desktop); otherwise each stage uploads and drops its own.
  // tiles: conv tiles per class (CONV_TILES); classes not given are timed on this device (tuneTiles).
  static async load(device, manifest, source, { resident = true, tiles = {} } = {}) {
    if (manifest.version !== 1) throw new Error(`encoder: weights in format ${manifest.version}, this code reads format 1`);
    const e = new AudioEncoder(device, manifest, source, resident);
    e.tuned = await e.tuneTiles(tiles);
    e.tiles = Object.fromEntries(Object.entries(e.tuned).map(([k, t]) => [k, parseCfg(t)]));
    await e.compile();
    if (resident) await e.upload(Object.keys(manifest.tensors));
    else e.rs = new Float32Array(await source.read(manifest.tensors['rs.kernel'].offset, manifest.tensors['rs.kernel'].bytes));
    return e;
  }

  constructor(device, manifest, source, resident) {
    Object.assign(this, { device, man: manifest, cfg: manifest.config, source, resident, w: {}, p: {} });
    this.zeros = device.createBuffer({ size: 4 * 4096, usage: GPUBufferUsage.STORAGE });
  }

  // The faster conv tile per class on this device, on a convolution like the encoder's (wall
  // clock: the candidates differ by ~15–40 %). All candidates give bit-identical results.
  async tuneTiles(given) {
    const d = this.device, out = { ...CONV_TILES, ...given };
    const shapes = { w128: { M: 512, N: 768, cin: 768, taps: 3 }, w64: { M: 8192, N: 64, cin: 64, taps: 7, dil: 3 } };
    const todo = Object.keys(TILE_CANDIDATES).filter((k) => !given[k]);
    if (!todo.length) return out;
    const lim = d.limits, bufs = [];
    const mk = (bytes) => { const b = d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE }); bufs.push(b); return b; };
    try {
      for (const k of todo) {
        const s = shapes[k], cands = TILE_CANDIDATES[k].filter((t) => {
          const c = parseCfg(t), { MT, NT } = dims(c), shared = c.v === 'reg' ? 0 : (MT * c.KT + c.KT * NT) * 4;
          return c.TM * c.TN <= lim.maxComputeInvocationsPerWorkgroup && shared <= lim.maxComputeWorkgroupStorageSize && s.N % NT === 0 && s.cin % c.KT === 0;
        });
        if (cands.length < 2) { if (cands.length) out[k] = cands[0]; continue; }
        const pipes = await Promise.all(cands.map((t) => compile(d, convWGSL(parseCfg(t), 'raw'), `enc tune ${t}`)));
        const X = mk(s.M * s.cin * 4), W = mk(s.taps * s.cin * s.N * 4), Y = mk(s.M * s.N * 4);
        const ms = cands.map(() => []);
        for (let rep = 0; rep < 3; rep++) { // interleaved; the first round warms up
          for (let i = 0; i < cands.length; i++) {
            const P = this.prog(), { MT, NT } = dims(parseCfg(cands[i])), dil = s.dil || 1, pad = (s.taps - 1) / 2 * dil;
            for (let n = 0; n < 4; n++) {
              P.add(pipes[i], [X, W, this.zeros, Y], [s.M, s.N, s.taps * s.cin, s.cin, s.cin / 4, 1, dil, pad, 0, 0, s.M, 0, s.N / 4, 0, 0, 0],
                [s.N / NT, Math.ceil(s.M / MT), 1]);
            }
            const t0 = performance.now();
            await this.submit(P);
            if (rep) ms[i].push(performance.now() - t0);
          }
        }
        const best = ms.map((m) => Math.min(...m));
        out[k] = cands[best.indexOf(Math.min(...best))];
      }
    } finally { for (const b of bufs) b.destroy(); }
    return out;
  }

  async compile() {
    const { hubert: hb, resample } = this.cfg;
    const d = this.device;
    const jobs = [['conv1.snake', CONV1('rawSnake')], ['conv1.gn', CONV1('gn')],
      ['gnstat', GNSTAT(hb.conv_kernel[0], hb.conv_stride[0], hb.conv_dim[0], hb.gn_eps.toExponential())],
      ['ln512', LN(512, hb.eps.toExponential())], ['ln768', LN(768, hb.eps.toExponential())],
      ['ln768add', LN(768, hb.eps.toExponential(), { add: true, hs: 'set' })], ['ln768acc', LN(768, hb.eps.toExponential(), { hs: 'acc' })],
      ['qkv', QKVSPLIT(hb.heads, hb.hidden / hb.heads)], ['downmean', DOWNMEAN(hb.hidden, hb.layers + 1)],
      ['rvq', RVQ(this.cfg.hidden, this.cfg.codebook_dim, this.cfg.codebook_size)],
      ['score', gemmWGSL('f32', 'f32', 'f32', 'f32', parseCfg(ATT_SCORE), 'score')], ['pv', gemmWGSL('f32', 'f32', 'f32', 'f32', parseCfg(ATT_PV), 'pv')]];
    for (const epi of Object.keys(EPI)) for (const [tk, cfg] of Object.entries(this.tiles)) jobs.push([`conv.${epi}.${tk}`, convWGSL(cfg, epi)]);
    const pipes = await Promise.all(jobs.map(([k, code]) => compile(d, code, `enc ${k}`)));
    jobs.forEach(([k], i) => { this.p[k] = pipes[i]; });
    this.resample = resample;
  }

  // Writes are waited for every UPLOAD_BATCH bytes: Safari copies each write for its GPU process,
  // and a whole stage's copies at once would sit next to the weights
  async upload(names) {
    let pending = 0;
    for (const name of names) {
      if (this.w[name]) continue;
      const t = this.man.tensors[name];
      const buf = this.device.createBuffer({ size: Math.ceil(t.bytes / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `enc ${name}` });
      const data = await this.source.read(t.offset, t.bytes);
      this.device.queue.writeBuffer(buf, 0, data);
      if (name === 'rs.kernel') this.rs = new Float32Array(data.slice(0));
      this.w[name] = buf;
      if ((pending += t.bytes) >= UPLOAD_BATCH) { await this.device.queue.onSubmittedWorkDone(); pending = 0; }
    }
  }

  // ms: wall time of the weight upload (<name>Load) and of the stage (<name>)
  async stage(name, prefix, fn, ms) {
    let t0 = performance.now();
    const names = Object.keys(this.man.tensors).filter((n) => prefix.some((p) => n.startsWith(p)));
    await this.upload(names);
    if (!this.resident) { await this.device.queue.onSubmittedWorkDone(); ms[`${name}Load`] = Math.round(performance.now() - t0); }
    t0 = performance.now();
    try { return await fn(); } finally {
      ms[name] = Math.round(performance.now() - t0);
      if (!this.resident) {
        await this.device.queue.onSubmittedWorkDone();
        for (const n of names) { this.w[n].destroy(); delete this.w[n]; }
      }
    }
  }

  // 24 kHz pcm (length a multiple of the hop) -> { codes: Int32Array [C][T], T }. debug: names of
  // intermediates to read back (feat, h0, hmean, esem, eac, emb), returned as Float32Arrays.
  async encode(pcm, { debug = [] } = {}) {
    const { hop, codebooks: C, hidden } = this.cfg;
    const T = pcm.length / hop;
    if (!Number.isInteger(T) || T < 1) throw new Error(`encoder: ${pcm.length} samples is not a whole number of ${hop}-sample frames`);
    const d = this.device;
    const keep = [];
    const mk = (bytes, label) => { const b = d.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label }); keep.push(b); return b; };
    const out = { T, stageMs: {} };
    const tap = async (name, buf, floats) => { if (debug.includes(name)) out[name] = await this.read(buf, floats); };
    try {
      const cat = mk(T * hidden * 4, 'enc cat'); // [T][256 acoustic | 768 semantic]
      await this.stage('hubert', ['rs.', 'h.'], () => this.hubert(pcm, T, mk, tap), out.stageMs);
      await this.stage('semantic', ['s.'], () => this.semantic(T, cat, mk, tap), out.stageMs);
      await this.stage('acoustic', ['a.'], () => this.acoustic(pcm, T, cat, mk, tap), out.stageMs);
      await this.stage('quant', ['fc.', 'q'], async () => {
        const emb = mk(T * hidden * 4, 'enc emb'), codes = mk(C * T * 4, 'enc codes');
        const P = this.prog();
        this.conv(P, 'raw', cat, 'fc.w', 'fc.b', emb, { M: T, N: hidden, cin: hidden });
        await this.submit(P);
        await tap('emb', emb, T * hidden);
        const Q = this.prog();
        for (let c = 0; c < C; c++) {
          Q.add(this.p.rvq, [emb, ...['in.w', 'in.b', 'embed', 'esq', 'out.w', 'out.b'].map((n) => this.w[`q${c}.${n}`]), codes], [T, c], [T, 1, 1]);
        }
        await this.submit(Q);
        out.codes = new Int32Array((await this.read(codes, C * T)).buffer);
      }, out.stageMs);
      return out;
    } finally {
      await d.queue.onSubmittedWorkDone();
      for (const b of keep) b.destroy();
    }
  }

  // HuBERT: resample, conv stack (windows), projection, positional conv, 12 layers; leaves the
  // semantic input [T][768] (mean of the 13 hidden states at every 2nd frame) in this.semIn.
  async hubert(pcm, T, mk, tap) {
    const { resample: rs, semantic_pad: spad, hubert: hb } = this.cfg;
    const d = this.device;
    // resample with torchaudio's kernel: y[i·new + j] = Σ_k K[j][k] · x[i·orig + k − width], then pad
    const n16 = Math.ceil(rs.new * pcm.length / rs.orig);
    const wav = new Float32Array(n16 + 2 * spad);
    for (let i = 0; i * rs.new < n16; i++) {
      for (let j = 0; j < rs.new && i * rs.new + j < n16; j++) {
        let acc = 0;
        for (let k = 0; k < rs.taps; k++) {
          const src = i * rs.orig + k - rs.width;
          if (src >= 0 && src < pcm.length) acc += this.rs[j * rs.taps + k] * pcm[src];
        }
        wav[spad + i * rs.new + j] = acc;
      }
    }
    const X = mk(wav.length * 4, 'enc wav16');
    d.queue.writeBuffer(X, 0, wav);
    // conv stack lengths: level l = output of conv l
    const K = hb.conv_kernel, S = hb.conv_stride, Cd = hb.conv_dim[0];
    const lens = [];
    let L = wav.length;
    for (let l = 0; l < K.length; l++) { L = Math.floor((L - K[l]) / S[l]) + 1; lens.push(L); }
    const F = lens.at(-1); // = 2T
    if (F !== 2 * T) throw new Error(`encoder: HuBERT gives ${F} frames for ${T}`);
    // GroupNorm statistics over all frames of conv 0
    const st = mk(Cd * 8, 'enc gn');
    {
      const P = this.prog();
      for (const pass of [0, 1]) P.add(this.p.gnstat, [X, this.w['h.c0.w'], st], [lens[0], pass], [Cd, 1, 1]);
      await this.submit(P);
    }
    // windows of final frames; ranges per level, back to front
    const wins = [];
    for (let a = 0; a < F; a += HUBERT_WINDOW) {
      const r = [[a, Math.min(F, a + HUBERT_WINDOW)]];
      for (let l = K.length - 1; l > 0; l--) r.unshift(need(r[0], { k: K[l], s: S[l] }, lens[l - 1]));
      wins.push(r); // r[l] = rows of level l
    }
    const maxRows = Math.max(...wins.flatMap((r) => r.slice(0, -1).map(span)));
    const feat = mk(F * Cd * 4, 'enc feat'), bufs = [mk(maxRows * Cd * 4, 'enc hc A'), mk(maxRows * Cd * 4, 'enc hc B')];
    {
      const P = this.prog();
      for (const r of wins) {
        P.add(this.p['conv1.gn'], [X, this.w['h.c0.w'], this.zeros, bufs[0], st, this.w['h.gn.g'], this.w['h.gn.b']],
          [span(r[0]), Cd, K[0], S[0], 0, r[0][0], wav.length, r[0][0]], grid1(span(r[0]) * Cd / 4));
        for (let l = 1; l < K.length; l++) {
          const last = l === K.length - 1, src = bufs[(l - 1) % 2];
          this.conv(P, 'gelu', src, `h.c${l}.w`, null, last ? feat : bufs[l % 2],
            { M: span(r[l]), N: Cd, cin: Cd, taps: K[l], stride: S[l], olo: r[l][0], xbase: r[l - 1][0], xlen: lens[l - 1], obase: last ? 0 : r[l][0] });
        }
      }
      await this.submit(P);
    }
    await tap('feat', feat, F * Cd);
    // projection, positional conv, layers
    const H = hb.hidden, NH = hb.heads, HD = H / NH, Mp = Math.ceil(F / 64) * 64;
    const h = mk(F * H * 4, 'enc h'), pos = mk(F * H * 4, 'enc pos'), hs = mk(F * H * 4, 'enc hs');
    const qkv = mk(F * 3 * H * 4, 'enc qkv'), ff = mk(F * hb.ff * 4, 'enc ff'), att = mk(Mp * H * 4, 'enc att');
    const q = mk(NH * Mp * HD * 4, 'enc q'), kt = mk(NH * HD * Mp * 4, 'enc kt'), v = mk(NH * Mp * HD * 4, 'enc v');
    const G = Math.min(HEAD_GROUP, NH), sc = mk(G * Mp * Mp * 4, 'enc scores'), stt = mk(G * Mp * (Mp / 64) * 8, 'enc score stats');
    const P0 = this.prog();
    P0.add(this.p.ln512, [feat, this.w['h.fp.ln.g'], this.w['h.fp.ln.b']], [F], [F, 1, 1]);
    this.conv(P0, 'raw', feat, 'h.fp.w', 'h.fp.b', h, { M: F, N: H, cin: Cd });
    const g = hb.pos_groups, cg = H / g; // grouped conv, kernel 128, pad 64; the extra last frame is not computed
    this.conv(P0, 'gelu', h, 'h.pos.w', 'h.pos.b', pos, { M: F, N: cg, cin: cg, xs4: H / 4, taps: hb.pos_kernel, pad: hb.pos_kernel / 2, groups: g, os4: H / 4 });
    P0.add(this.p.ln768add, [h, this.w['h.ln.g'], this.w['h.ln.b'], pos, hs], [F], [F, 1, 1]);
    await this.submit(P0);
    await tap('h0', hs, F * H);
    const P = this.prog();
    const scale = new Uint32Array(new Float32Array([1 / Math.sqrt(HD)]).buffer)[0];
    const att0 = dims(parseCfg(ATT_SCORE)), pv0 = dims(parseCfg(ATT_PV));
    for (let i = 0; i < hb.layers; i++) {
      const Lw = (n) => `h.L${i}.${n}`;
      this.conv(P, 'raw', h, Lw('qkv.w'), Lw('qkv.b'), qkv, { M: F, N: 3 * H, cin: H });
      P.add(this.p.qkv, [qkv, q, kt, v], [Mp], [F, 3 * NH, 1]);
      for (let g0 = 0; g0 < NH; g0 += G) {
        const tiles = Mp / att0.NT;
        P.add(this.p.score, [q, kt, sc, stt], [Mp, Mp, HD, HD / 4, Mp / 4, Mp / 4, g0 * Mp * HD / 4, g0 * HD * Mp / 4, 0, Mp * HD / 4, HD * Mp / 4, Mp * Mp / 4, 1, F, scale, tiles],
          [tiles, Mp / att0.MT, G]);
        P.add(this.p.pv, [sc, v, att, stt], [Mp, HD, Mp, Mp / 4, HD / 4, H / 4, 0, g0 * Mp * HD / 4, g0 * HD / 4, Mp * Mp / 4, Mp * HD / 4, HD / 4, 1, F, 0, tiles],
          [HD / pv0.NT, Mp / pv0.MT, G]);
      }
      this.conv(P, 'rmw', att, Lw('o.w'), Lw('o.b'), h, { M: F, N: H, cin: H });
      P.add(this.p.ln768, [h, this.w[Lw('ln1.g')], this.w[Lw('ln1.b')]], [F], [F, 1, 1]);
      this.conv(P, 'gelu', h, Lw('ff1.w'), Lw('ff1.b'), ff, { M: F, N: hb.ff, cin: H });
      this.conv(P, 'rmw', ff, Lw('ff2.w'), Lw('ff2.b'), h, { M: F, N: H, cin: hb.ff });
      P.add(this.p.ln768acc, [h, this.w[Lw('ln2.g')], this.w[Lw('ln2.b')], hs], [F], [F, 1, 1]);
    }
    this.semIn = mk(T * H * 4, 'enc sem in');
    P.add(this.p.downmean, [hs, this.semIn], [T], [T, 1, 1]);
    await this.submit(P);
    await tap('hmean', hs, F * H); // the sum; the page divides by 13
  }

  // semantic encoder on [T][768]: conv, then blocks of ELU residual units and a conv; the result
  // goes into cat's columns 256..1023
  async semantic(T, cat, mk, tap) {
    const { semantic: sc, hubert: hb, acoustic } = this.cfg, H = hb.hidden;
    if (sc.block_stride !== 1) throw new Error('encoder: semantic blocks with stride ≠ 1');
    const xs = [mk(T * H * 4, 'enc sx A'), mk(T * H * 4, 'enc sx B')], sn = mk(T * H * 4, 'enc ssn'), hbuf = mk(T * H * 4, 'enc sh');
    const P = this.prog();
    const k = sc.kernel, p = (kk) => (kk - 1) / 2;
    let x = xs[0];
    this.conv(P, 'rawElu', this.semIn, 's.conv.w', null, x, { M: T, N: H, cin: H, taps: k, pad: p(k), sn });
    for (let b = 0; b < sc.blocks; b++) {
      sc.units[b].forEach((dil, u) => { // x += conv2(elu(conv1(elu(x)))); sn = elu(x)
        const uk = sc.unit_kernel;
        this.conv(P, 'elu', sn, `s.b${b}.u${u}.w1`, null, hbuf, { M: T, N: H, cin: H, taps: uk, dil, pad: p(uk) * dil });
        this.conv(P, 'rmwElu', hbuf, `s.b${b}.u${u}.w2`, null, x, { M: T, N: H, cin: H, sn });
      });
      const bk = sc.block_kernel, o = { M: T, N: H, cin: H, taps: bk, pad: p(bk) };
      if (b === sc.blocks - 1) {
        this.conv(P, 'raw', x, `s.b${b}.w`, `s.b${b}.b`, cat, { ...o, os4: this.cfg.hidden / 4, ocol4: acoustic.hidden / 4 });
      } else {
        const y = xs[(b + 1) % 2];
        this.conv(P, 'rawElu', x, `s.b${b}.w`, `s.b${b}.b`, y, { ...o, sn });
        x = y;
      }
    }
    await this.submit(P);
    await tap('esem', cat, T * this.cfg.hidden);
  }

  // DAC encoder at 24 kHz -> cat's columns 0..255. Blocks before DAC_SPLIT (where most rows are)
  // run in windows; their receptive field is a few samples per frame, so windows recompute little.
  // The rest runs over the whole clip at once (its levels are small; in windows of 64 frames the
  // deep layers' receptive field made the encoder compute 53 % more).
  async acoustic(pcm, T, cat, mk, tap) {
    const { acoustic: ac, hidden } = this.cfg;
    const d = this.device;
    const X = mk(pcm.length * 4, 'enc pcm');
    d.queue.writeBuffer(X, 0, pcm);
    const blocks = ac.blocks, nb = blocks.length, S = Math.min(DAC_SPLIT, nb);
    const lens = [pcm.length];
    for (const b of blocks) lens.push(Math.floor((lens.at(-1) + 2 * b.pad - b.kernel) / b.stride) + 1);
    if (lens.at(-1) !== T) throw new Error(`encoder: DAC gives ${lens.at(-1)} frames for ${T}`);
    const chans = [ac.channels, ...blocks.map((b) => b.cout)];
    // per block i0 ≤ i < i1, the rows its units and down conv need for rows r of level i1
    const levels = (i0, i1, r) => {
      const lv = [];
      for (let i = i1 - 1; i >= i0; i--) {
        const b = blocks[i], unitOut = [];
        let x = need(r, { k: b.kernel, s: b.stride, p: b.pad }, lens[i]);
        for (let u = b.dilations.length - 1; u >= 0; u--) { unitOut.unshift(x); x = need(x, { k: 7, d: b.dilations[u], p: 3 * b.dilations[u] }, lens[i]); }
        lv.unshift({ entry: x, unitOut, down: r });
        r = x;
      }
      return lv;
    };
    // blocks i0.. over the ranges lv; io(i) = { x, sn: snake(x), base: first row held } of level i
    const runBlocks = (P, i0, lv, io, hb) => lv.forEach((l, j) => {
      const i = i0 + j, b = blocks[i], C = b.cin, src = io(i), dst = io(i + 1);
      b.dilations.forEach((dil, u) => { // x += conv2(snake(conv1(snake(x)))); sn = snake(x) with the next alpha
        const ro = l.unitOut[u], p = `a.b${i}.u${u}`;
        const next = u < b.dilations.length - 1 ? `a.b${i}.u${u + 1}.a1` : `a.b${i}.a`;
        const o = { M: span(ro), N: C, cin: C, olo: ro[0], xbase: src.base, xlen: lens[i], obase: src.base };
        this.conv(P, 'snake', src.sn, `${p}.w1`, `${p}.b1`, hb, { ...o, taps: 7, dil, pad: 3 * dil, alpha: `${p}.a2` });
        this.conv(P, 'rmwSnake', hb, `${p}.w2`, `${p}.b2`, src.x, { ...o, sn: src.sn, alpha: next });
      });
      this.conv(P, 'rawSnake', src.sn, `a.b${i}.w`, `a.b${i}.b`, dst.x, { M: span(l.down), N: b.cout, cin: C, taps: b.kernel, stride: b.stride,
        pad: b.pad, olo: l.down[0], xbase: src.base, xlen: lens[i], obase: dst.base, sn: dst.sn, alpha: i < nb - 1 ? `a.b${i + 1}.u0.a1` : 'a.out.a' });
    });
    const full = {}; // levels S.. over the whole clip
    for (let i = S; i <= nb; i++) full[i] = { x: mk(lens[i] * chans[i] * 4, `enc a${i}`), sn: mk(lens[i] * chans[i] * 4, `enc asn${i}`), base: 0 };
    const P = this.prog();
    if (S > 0) { // windows of level-S rows, ~DAC_WINDOW frames each
      const WS = Math.max(1, Math.round(DAC_WINDOW * lens[S] / T)), wins = [];
      for (let a = 0; a < lens[S]; a += WS) wins.push(levels(0, S, [a, Math.min(lens[S], a + WS)]));
      const maxF = Math.max(...wins.flatMap((lv) => lv.map((l, i) => span(l.entry) * chans[i])));
      const sets = [0, 1].map((k) => ({ x: mk(maxF * 4, `enc ax ${k}`), sn: mk(maxF * 4, `enc asn ${k}`) })), hb = mk(maxF * 4, 'enc ah');
      for (const lv of wins) {
        const e0 = lv[0].entry;
        P.add(this.p['conv1.snake'], [X, this.w['a.conv1.w'], this.w['a.conv1.b'], sets[0].x, sets[0].sn, this.w['a.b0.u0.a1']],
          [span(e0), ac.channels, 7, 1, 3, e0[0], pcm.length, e0[0]], grid1(span(e0) * ac.channels / 4));
        runBlocks(P, 0, lv, (i) => (i === S ? full[S] : { ...sets[i % 2], base: lv[i].entry[0] }), hb);
      }
    } else {
      P.add(this.p['conv1.snake'], [X, this.w['a.conv1.w'], this.w['a.conv1.b'], full[0].x, full[0].sn, this.w['a.b0.u0.a1']],
        [lens[0], ac.channels, 7, 1, 3, 0, pcm.length, 0], grid1(lens[0] * ac.channels / 4));
    }
    const lv = levels(S, nb, [0, T]);
    if (lv.length) runBlocks(P, S, lv, (i) => full[i], mk(Math.max(...lv.map((l, j) => span(l.entry) * chans[S + j])) * 4, 'enc ah full'));
    this.conv(P, 'raw', full[nb].sn, 'a.out.w', 'a.out.b', cat, { M: T, N: ac.hidden, cin: chans[nb], taps: 3, pad: 1, xlen: T, os4: hidden / 4 });
    await this.submit(P);
    await tap('eac', cat, T * hidden);
  }

  // conv step (see convWGSL); defaults: one tap, stride 1, input/output rows from 0, row strides = channels
  conv(P, epi, X, wname, bname, OUT, o) {
    const { M, N, cin, taps = 1, stride = 1, dil = 1, pad = 0, olo = 0, xbase = 0, xlen = o.M * stride, obase = 0, groups = 1,
      xs4 = cin / 4, os4 = N / 4, ocol4 = 0, sn = null, alpha = null } = o;
    const tk = N === 48 ? 'g48' : N % dims(this.tiles.w128).NT === 0 ? 'w128' : 'w64', { MT, NT } = dims(this.tiles[tk]);
    const bufs = [X, this.w[wname], bname ? this.w[bname] : this.zeros, OUT];
    if (sn) bufs.push(sn);
    if (alpha) bufs.push(this.w[alpha]);
    P.add(this.p[`conv.${epi}.${tk}`], bufs,
      [M, N, taps * cin, cin, xs4, stride, dil, pad, olo, xbase, xlen ?? M, obase, os4, ocol4, groups > 1 ? taps * cin * N / 4 : 0, groups > 1 ? cin / 4 : 0],
      [N / NT, Math.ceil(M / MT), groups]);
  }

  prog() {
    const steps = [];
    return { steps, add: (pipe, buffers, u, grid) => steps.push({ pipe, buffers, u, grid }) };
  }

  async submit(P) {
    const d = this.device;
    const ub = d.createBuffer({ size: Math.max(1, P.steps.length) * UNI, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const all = new Int32Array(P.steps.length * UNI / 4);
    P.steps.forEach((s, k) => all.set(s.u.map((x) => x | 0), k * UNI / 4));
    d.queue.writeBuffer(ub, 0, all);
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    P.steps.forEach((s, k) => {
      const bg = d.createBindGroup({ layout: s.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ub, offset: k * UNI, size: 64 } },
        ...s.buffers.map((buffer, i) => ({ binding: i + 1, resource: { buffer } })),
      ] });
      pass.setPipeline(s.pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...s.grid);
    });
    pass.end();
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    ub.destroy();
  }

  async read(buf, floats) {
    const d = this.device, rb = d.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, floats * 4); d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    return out;
  }

  destroy() { for (const b of Object.values(this.w)) b.destroy(); this.w = {}; this.zeros.destroy(); }
}

// 1-D grid of 64-thread workgroups over n items (x up to 65535)
function grid1(n) { const g = Math.ceil(n / 64); return [Math.min(g, 65535), Math.ceil(g / 65535), 1]; }

export { convWGSL, QKVSPLIT, grid1, ATT_SCORE, ATT_PV }; // for the Whisper engine
