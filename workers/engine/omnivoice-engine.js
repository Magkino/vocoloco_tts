// VocoLoco engine: the OmniVoice main model on plain WebGPU.
//
// One forward pass runs the cond rows and the uncond rows as one batch, packed (layout()).
// No kernel checks edges: linear layers run the real rows rounded up to their tile; attention
// runs per segment on rows and keys rounded up to 64, keys past the segment masked, which makes
// it exactly block-diagonal (rows past a segment are overwritten by the next one or unused).
// The audio head runs on target rows only.
//
//   rows:  [cond 0..L)[pad to Ls][uncond 0..T)[zero rows to M]    Mr = Ls + T real rows
//   head:  rows [0, T) cond targets, [Th, Th + T) uncond, columns c·1025 + v (8320 wide)
//
// Weights: a JSON manifest and one binary file (engine/ on Hugging Face). Matmul weights
// f32 / f16 / int8 (+ f16 scales per 32 along K). The text embedding never goes to the GPU:
// the rows a generation needs are read on the CPU (source.read) and uploaded per plan.
//
// Precision: residual stream, norms, softmax, attention and all accumulators in f32.
// opts.act = 'f16' stores the matmul inputs (normed rows, SiLU·up, attention output) in f16
// and stages tiles in f16 (needs shader-f16); 'f32' keeps everything in f32.

const PAD = 64;
const ceil = (x, m) => Math.ceil(x / m) * m;
const UNI = 256; // uniform slot size (minUniformBufferOffsetAlignment)
// Weight bytes written before waiting for the GPU: Safari copies each write for its GPU process,
// and a whole model's copies at once froze iPhones for seconds and pushed them to their memory limit
const UPLOAD_BATCH = 64 << 20;

// Row layout of a pass with L cond and T uncond rows: the uncond segment starts at Ls (keys are
// read in vec4s along rows), attention segments are n × n with n = valid rounded up to 64, and
// M rows are allocated (the row stride) so that every tile overhang stays inside the buffers.
function layout(L, T) {
  const Ls = ceil(L, 4), Mr = Ls + T, Th = ceil(T, 32);
  const segs = [[0, L], [Ls, T]].map(([s0, valid]) => ({ s0, valid, n: ceil(valid, PAD) }));
  const M = Math.max(ceil(Mr, PAD), ...segs.map((s) => s.s0 + s.n));
  return { Ls, Mr, Th, segs, M };
}

// tiles, per device (tune()): linear layers per layer kind, attention scores and P·V
const dims = (c) => ({ MT: c.TM * c.RM, NT: c.TN * c.VN * 4 });
// linear-layer tiles tried by tune() ("TMxTN r RM v VN k KT[:variant]", variants see gemmWGSL;
// all give bit-identical results); the first is the default. The variants won on the iPhone
// (−11 %) and Mac Chrome (−5 %), unrolled loops lost badly on the iPhone.
const LIN_CANDIDATES = ['16x16r4v2k32', '16x8r4v2k32', '16x16r4v1k32', '16x8r4v1k32', '16x8r4v2k32:pf',
  '16x16r4v1k32:pf', '16x8r4v2k32:pfw', '8x16r4v2k32:pfw', '8x16r4v2k32:wreg', '16x16r4v2k32:wreg',
  '8x16r4v2k32:reg', '16x8r4v2k32:reg', '16x16r4v2k32:reg'];
// attention tiles: MT divides 64 (segments are padded to 64), the P·V NT divides HD. Score tiles
// stay 16 × 1 vec4 wide (NT 64): a score tile's width sets the order of the softmax sums, so
// other widths would differ in the last bits and the tuner could change results. Attention is
// f32 throughout, which the iPhone stages slowly; there 'reg' scores (−35 %) and 'pfw' P·V
// (−20 %) won (all checked against a float64 reference).
const ATT_CANDIDATES = {
  score: ['16x16r4v1k16', '16x16r4v1k16:pfw', '16x16r4v1k16:reg', '8x16r4v1k16:pfw', '8x16r4v1k16:reg',
    '8x16r8v1k16:reg'],
  pv: ['16x16r4v2k16', '16x16r4v2k16:pf', '16x16r4v2k16:wreg', '8x16r4v2k16:pfw', '8x16r4v2k32:pfw',
    '16x8r4v2k16:pfw', '8x16r8v1k16:pfw'],
};
const ATT_KINDS = ['score', 'pv'];
const parseCfg = (s) => {
  const [tile, v = ''] = s.split(':');
  const [, TM, TN, RM, VN, KT] = /^(\d+)x(\d+)r(\d+)v(\d+)k(\d+)$/.exec(tile).map(Number);
  return { TM, TN, RM, VN, KT, v };
};
const LIN_KINDS = ['qkv', 'o', 'gate_up', 'down', 'head'];
const LIN_MODES = { gate_up: 'swiglu' }; // fused epilogues (gemmWGSL)
// tiles a layer kind can use (the SiLU·up epilogue of gate_up needs 16 × 2 vec4)
const linFits = (kind, name) => !LIN_MODES[kind] || (parseCfg(name).TN === 16 && parseCfg(name).VN === 2);
// whether a device can run a gemmWGSL tile: shared memory and threads per workgroup (WebGPU's
// defaults are 16 KB and 256; many Android GPUs stop there). sT: staging type.
function gemmFits(limits, sT, { TM, TN, RM, VN, KT, v }, mode = null) {
  const MT = TM * RM, NT = TN * VN * 4, vec = sT === 'f16' ? 8 : 16;
  const reg = v === 'reg', wreg = reg || v === 'wreg' || v === 'pfw';
  const shared = (reg ? 0 : MT * KT / 4 * vec) + (wreg ? 0 : KT * NT / 4 * vec) + (mode === 'score' ? MT * TN * 8 : mode === 'pv' ? MT * 8 : 0);
  return shared <= limits.maxComputeWorkgroupStorageSize && TM * TN <= limits.maxComputeInvocationsPerWorkgroup
    && TN <= limits.maxComputeWorkgroupSizeX && TM <= limits.maxComputeWorkgroupSizeY;
}

// ---------------------------------------------------------------- kernels

// C[z] = A[z] · W[z / wDiv] with strides; offsets and leading dims in vec4 units (for i8 weights:
// u32 = 4 values along N, scales S [K/32][N/4] vec4<f16>). W is [K][N]. Grid (N / NT, M / MT, batch).
// wT: weight storage f32 | f16 | i8; aT: A storage; sT: staging (shared memory) type; oT: C storage.
//
// Modes (epilogues and prologues). Attention without a separate softmax pass (uniforms p0 = valid
// keys, p1 = scale bits, p2 = score tiles per row):
//   'score' stores the scaled scores and, per row and NT-column tile, the max m and
//           Σ exp(s − m) over the valid columns into ST [z][row][tile];
//   'pv'    combines a row's tile stats (M = max m, L = Σ l·exp(m − M)) and loads the scores as
//           exp(s − M) / L, 0 for padding keys: the exact softmax, summed in another order;
//   'swiglu' (gate_up, 16 × 2 vec4 tiles: columns interleaved per 64 by interleaveGateUp) writes
//           SiLU(gate) · up into C [M][N / 2], the expression of the separate pass it replaces.
//
// Variants (cfg.v, f32/f16 weights; same operations in the same order):
//   'pf'   the next K tile is loaded into registers while the current one is computed;
//   'wreg' weights are read straight from memory in the inner loop (no shared-memory copy);
//   'pfw'  both (only the activations go through registers and shared memory);
//   'reg'  both operands straight from memory, no shared memory and no barriers.
function gemmWGSL(wT, aT, sT, oT, { TM, TN, RM, VN, KT, v: variant = '' }, mode = null) {
  const MT = TM * RM, NT = TN * VN * 4, THREADS = TM * TN, KT4 = KT / 4, NT4 = NT / 4;
  const f16 = [wT, aT, sT, oT].includes('f16') || wT === 'i8';
  const pf = variant === 'pf' || variant === 'pfw', reg = variant === 'reg', wreg = reg || variant === 'wreg' || variant === 'pfw';
  if ((pf || wreg) && wT === 'i8') throw new Error(`variant ${variant}: f32/f16 weights only`);
  const att = mode === 'score' || mode === 'pv' ? mode : null;
  if (att && THREADS < MT) throw new Error('attention tiles need a thread per tile row');
  if (mode === 'swiglu' && (TN !== 16 || VN !== 2)) throw new Error('swiglu needs 16 × 2 vec4 tiles');
  const loadW = wT === 'i8'
    ? `let u = W[wB + (k0 + r) * g.ldw + n0 / 4u + c];
      let q = vec4<i32>(bitcast<i32>(u << 24u) >> 24u, bitcast<i32>(u << 16u) >> 24u, bitcast<i32>(u << 8u) >> 24u, bitcast<i32>(u) >> 24u);
      Ws[e] = vec4<${sT}>(q) * vec4<${sT}>(S[((k0 + r) / 32u) * g.ldw + n0 / 4u + c]);`
    : `Ws[e] = vec4<${sT}>(W[wB + (k0 + r) * g.ldw + n0 / 4u + c]);`;
  // staged A value: tile row r, vec4 column c of the K tile starting at k
  const ldA = (r, c, k) => {
    const a = `A[aB + (m0 + ${r}) * g.lda + (${k}) / 4u + ${c}]`;
    return att === 'pv' ? `vec4<${sT}>(pload(${r}, (${k}) + 4u * ${c}, vec4<f32>(${a})))` : `vec4<${sT}>(${a})`;
  };
  let inner = '';
  for (let i = 0; i < RM; i++) {
    inner += reg
      ? `\n      let a${i} = vec4<f32>(${ldA(`(ly + ${i * TM}u)`, 'k4', 'k0')});`
      : `\n      let a${i} = vec4<f32>(As[(ly + ${i * TM}u) * ${KT4}u + k4]);`;
  }
  for (let kk = 0; kk < 4; kk++) {
    const ch = 'xyzw'[kk];
    for (let v = 0; v < VN; v++) {
      inner += wreg
        ? `\n      let b${kk}_${v} = vec4<f32>(W[wB + (k0 + k4 * 4u + ${kk}u) * g.ldw + n0 / 4u + ${v * TN}u + lx]);`
        : `\n      let b${kk}_${v} = vec4<f32>(Ws[(k4 * 4u + ${kk}u) * ${NT4}u + ${v * TN}u + lx]);`;
    }
    for (let i = 0; i < RM; i++) for (let v = 0; v < VN; v++) inner += `\n      c${i}_${v} = fma(vec4<f32>(a${i}.${ch}), b${kk}_${v}, c${i}_${v});`;
  }
  let decl = '', store = '';
  for (let i = 0; i < RM; i++) for (let v = 0; v < VN; v++) {
    decl += `\n  var c${i}_${v} = vec4<f32>(0.0);`;
    store += `\n  C[cB + (m0 + ly + ${i * TM}u) * g.ldc + n0 / 4u + ${v * TN}u + lx] = vec4<${oT}>(c${i}_${v});`;
  }
  if (mode === 'swiglu') { // v 0 holds 4 gate columns, v 1 the matching up columns
    store = '';
    for (let i = 0; i < RM; i++) {
      store += `\n  { let gt = c${i}_0; C[cB + (m0 + ly + ${i * TM}u) * g.ldc + n0 / 8u + lx] = vec4<${oT}>(gt / (vec4<f32>(1.0) + exp(-gt)) * c${i}_1); }`;
    }
  }
  const loadA = `As[e] = ${ldA('r', 'c', 'k0')};`;
  let attDecl = '', prologue = '';
  if (att === 'score') {
    attDecl = `@group(0) @binding(4) var<storage, read_write> ST: array<vec2<f32>>;
var<workgroup> red: array<vec2<f32>, ${MT * TN}>;`;
    store = `\n  let sc = bitcast<f32>(g.p1);`;
    for (let i = 0; i < RM; i++) {
      store += '\n  {\n    var m = -3.0e38;';
      for (let v = 0; v < VN; v++) {
        store += `
    let s${v} = c${i}_${v} * sc;
    C[cB + (m0 + ly + ${i * TM}u) * g.ldc + n0 / 4u + ${v * TN}u + lx] = vec4<${oT}>(s${v});
    let ok${v} = vec4<u32>(n0 + 4u * (${v * TN}u + lx)) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(g.p0);
    let sm${v} = select(vec4<f32>(-3.0e38), s${v}, ok${v});
    m = max(m, max(max(sm${v}.x, sm${v}.y), max(sm${v}.z, sm${v}.w)));`;
      }
      store += '\n    var l = 0.0;';
      for (let v = 0; v < VN; v++) store += `\n    l += dot(select(vec4<f32>(0.0), exp(s${v} - vec4<f32>(m)), ok${v}), vec4<f32>(1.0));`;
      store += `\n    red[(ly + ${i * TM}u) * ${TN}u + lx] = vec2<f32>(m, l);\n  }`;
    }
    store += `
  workgroupBarrier();
  if (li < ${MT}u) {
    var M = -3.0e38;
    for (var t = 0u; t < ${TN}u; t++) { M = max(M, red[li * ${TN}u + t].x); }
    var L = 0.0;
    for (var t = 0u; t < ${TN}u; t++) { let p = red[li * ${TN}u + t]; L += p.y * exp(p.x - M); }
    ST[(wg.z * g.M + m0 + li) * g.p2 + wg.x] = vec2<f32>(M, L);
  }`;
  } else if (att === 'pv') {
    attDecl = `@group(0) @binding(4) var<storage, read> ST: array<vec2<f32>>;
var<workgroup> rowM: array<f32, ${MT}>;
var<workgroup> rowL: array<f32, ${MT}>;
// softmax probabilities of tile row r, keys k..k+3 (0 for padding keys)
fn pload(r: u32, k: u32, s: vec4<f32>) -> vec4<f32> {
  let ok = vec4<u32>(k) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(g.p0);
  return select(vec4<f32>(0.0), exp(s - vec4<f32>(rowM[r])) / rowL[r], ok);
}`;
    prologue = `
  if (li < ${MT}u) {
    let base = (wg.z * g.M + m0 + li) * g.p2;
    var M = -3.0e38;
    for (var t = 0u; t < g.p2; t++) { M = max(M, ST[base + t].x); }
    var L = 0.0;
    for (var t = 0u; t < g.p2; t++) { let p = ST[base + t]; L += p.y * exp(p.x - M); }
    rowM[li] = M; rowL[li] = L;
  }
  workgroupBarrier();`;
  }
  const compute = `\n    for (var k4 = 0u; k4 < ${KT4}u; k4++) {${inner}\n    }`;
  let body;
  if (reg) {
    body = `
  for (var k0 = 0u; k0 < g.K; k0 += ${KT}u) {${compute}
  }`;
  } else if (pf) {
    const nA = MT * KT4 / THREADS, nW = wreg ? 0 : KT * NT4 / THREADS;
    if (!Number.isInteger(nA) || !Number.isInteger(nW)) throw new Error('pf: tiles must divide by the thread count');
    const load = (k) => {
      let s = '';
      for (let j = 0; j < nA; j++) s += `\n      { let e = li + ${j * THREADS}u; ra${j} = ${ldA(`(e / ${KT4}u)`, `(e % ${KT4}u)`, k)}; }`;
      for (let j = 0; j < nW; j++) s += `\n      { let e = li + ${j * THREADS}u; rw${j} = vec4<${sT}>(W[wB + ((${k}) + e / ${NT4}u) * g.ldw + n0 / 4u + e % ${NT4}u]); }`;
      return s;
    };
    let regs = '', put = '';
    for (let j = 0; j < nA; j++) { regs += `\n  var ra${j}: vec4<${sT}>;`; put += `\n    As[li + ${j * THREADS}u] = ra${j};`; }
    for (let j = 0; j < nW; j++) { regs += `\n  var rw${j}: vec4<${sT}>;`; put += `\n    Ws[li + ${j * THREADS}u] = rw${j};`; }
    body = `${regs}${load('0u')}
  for (var k0 = 0u; k0 < g.K; k0 += ${KT}u) {${put}
    workgroupBarrier();
    if (k0 + ${KT}u < g.K) {${load(`k0 + ${KT}u`)}
    }${compute}
    workgroupBarrier();
  }`;
  } else {
    body = `
  for (var k0 = 0u; k0 < g.K; k0 += ${KT}u) {
    for (var e = li; e < ${MT * KT4}u; e += ${THREADS}u) {
      let r = e / ${KT4}u; let c = e % ${KT4}u;
      ${loadA}
    }${wreg ? '' : `
    for (var e = li; e < ${KT * NT4}u; e += ${THREADS}u) {
      let r = e / ${NT4}u; let c = e % ${NT4}u;
      ${loadW}
    }`}
    workgroupBarrier();${compute}
    workgroupBarrier();
  }`;
  }
  return `${f16 ? 'enable f16;\n' : ''}struct G { M: u32, N: u32, K: u32, lda: u32, ldw: u32, ldc: u32, aOff: u32, wOff: u32,
  cOff: u32, aZ: u32, wZ: u32, cZ: u32, wDiv: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var<storage, read> A: array<vec4<${aT}>>;
@group(0) @binding(2) var<storage, read> W: ${wT === 'i8' ? 'array<u32>' : `array<vec4<${wT}>>`};
@group(0) @binding(3) var<storage, read_write> C: array<vec4<${oT}>>;
${wT === 'i8' ? '@group(0) @binding(4) var<storage, read> S: array<vec4<f16>>;\n' : ''}${attDecl}
${reg ? '' : `var<workgroup> As: array<vec4<${sT}>, ${MT * KT4}>;`}
${wreg ? '' : `var<workgroup> Ws: array<vec4<${sT}>, ${KT * NT4}>;`}
@compute @workgroup_size(${TN}, ${TM})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * ${MT}u; let n0 = wg.x * ${NT}u; let lx = lid.x; let ly = lid.y;
  let aB = g.aOff + wg.z * g.aZ; let wB = g.wOff + (wg.z / g.wDiv) * g.wZ; let cB = g.cOff + wg.z * g.cZ;${decl}${prologue}${body}${store}
}`;
}

// x[m] = Σ_c audio_emb[c·V + ids[c][m]] at audio rows, the uploaded text row otherwise, 0 at padding.
// rowinfo: bit 0 audio position, bit 1 real (not padding), bits 8.. RoPE position
const EMBED = (C, V, H, eT) => `${eT === 'f16' ? 'enable f16;\n' : ''}struct U { M: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> ids: array<i32>;
@group(0) @binding(2) var<storage, read> rowinfo: array<u32>;
@group(0) @binding(3) var<storage, read> trows: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> aemb: array<vec4<${eT}>>;
@group(0) @binding(5) var<storage, read_write> x: array<vec4<f32>>;
@compute @workgroup_size(${H / 4})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let m = wg.x; let f = rowinfo[m];
  var v = vec4<f32>(0.0);
  if ((f & 3u) == 3u) {
    for (var c = 0u; c < ${C}u; c++) { v += vec4<f32>(aemb[(c * ${V}u + u32(ids[c * u.M + m])) * ${H / 4}u + t]); }
  } else if ((f & 2u) != 0u) {
    v = trows[m * ${H / 4}u + t];
  }
  x[m * ${H / 4}u + t] = v;
}`;

// h = RMSNorm(resid) · w, optionally resid += d first. One workgroup per row (H / 4 threads).
const NORM = (H, EPS, add, hT) => `${hT === 'f16' ? 'enable f16;\n' : ''}@group(0) @binding(0) var<storage, read_write> resid: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> h: array<vec4<${hT}>>;
${add ? '@group(0) @binding(3) var<storage, read> d: array<vec4<f32>>;' : ''}
var<workgroup> red: array<f32, ${H / 4}>;
@compute @workgroup_size(${H / 4})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = wg.x * ${H / 4}u + t;
  ${add ? 'let r = resid[i] + d[i]; resid[i] = r;' : 'let r = resid[i];'}
  red[t] = dot(r, r);
  workgroupBarrier();
  for (var s = ${H / 8}u; s > 0u; s >>= 1u) { if (t < s) { red[t] += red[t + s]; } workgroupBarrier(); }
  h[i] = vec4<${hT}>(w[t] * (r * inverseSqrt(red[0] / ${H}.0 + ${EPS})));
}`;

// Final: rows of the head input, hT[o] = RMSNorm(resid[src] + d[src]) · w for src = rowmap[o]
// (0xffffffff = padding row -> zeros).
const FINAL = (H, EPS, hT) => `${hT === 'f16' ? 'enable f16;\n' : ''}@group(0) @binding(0) var<storage, read> resid: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> d: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rowmap: array<u32>;
@group(0) @binding(4) var<storage, read_write> h: array<vec4<${hT}>>;
var<workgroup> red: array<f32, ${H / 4}>;
@compute @workgroup_size(${H / 4})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let ok = rowmap[wg.x] != 0xffffffffu; let src = select(0u, rowmap[wg.x], ok); // no early return: barriers follow
  let r = resid[src * ${H / 4}u + t] + d[src * ${H / 4}u + t];
  red[t] = dot(r, r);
  workgroupBarrier();
  for (var s = ${H / 8}u; s > 0u; s >>= 1u) { if (t < s) { red[t] += red[t + s]; } workgroupBarrier(); }
  h[wg.x * ${H / 4}u + t] = vec4<${hT}>(select(vec4<f32>(0.0), w[t] * (r * inverseSqrt(red[0] / ${H}.0 + ${EPS})), ok));
}`;

// q/k: per-head RMSNorm + RoPE (HF rotate_half), written head-major: Q [NH][M][HD],
// Kt [NKV][HD][M] (transposed: the GEMM's [K][N] operand), V [NKV][M][HD]. Grid (M, NH + 2·NKV).
const QKROPE = (NH, NKV, HD, EPS) => {
  const QKV = (NH + 2 * NKV) * HD, H2 = HD / 2;
  return `struct U { M: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> rowinfo: array<u32>;
@group(0) @binding(3) var<storage, read> qn: array<f32>;
@group(0) @binding(4) var<storage, read> kn: array<f32>;
@group(0) @binding(5) var<storage, read> cs: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> Q: array<f32>;
@group(0) @binding(7) var<storage, read_write> Kt: array<f32>;
@group(0) @binding(8) var<storage, read_write> Vh: array<f32>;
var<workgroup> red: array<f32, ${H2}>;
@compute @workgroup_size(${H2})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let m = wg.x; let s = wg.y;
  if (s >= ${NH + NKV}u) {
    let hv = s - ${NH + NKV}u; let src = m * ${QKV}u + ${(NH + NKV) * HD}u + hv * ${HD}u; let dst = (hv * u.M + m) * ${HD}u;
    Vh[dst + t] = qkv[src + t]; Vh[dst + t + ${H2}u] = qkv[src + t + ${H2}u];
    return;
  }
  let isQ = s < ${NH}u; let hh = select(s - ${NH}u, s, isQ);
  let src = m * ${QKV}u + select(${NH * HD}u + hh * ${HD}u, hh * ${HD}u, isQ);
  let x1 = qkv[src + t]; let x2 = qkv[src + t + ${H2}u];
  red[t] = x1 * x1 + x2 * x2;
  workgroupBarrier();
  for (var k = ${H2 / 2}u; k > 0u; k >>= 1u) { if (t < k) { red[t] += red[t + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / ${HD}.0 + ${EPS});
  let y1 = select(kn[t], qn[t], isQ) * (x1 * inv);
  let y2 = select(kn[t + ${H2}u], qn[t + ${H2}u], isQ) * (x2 * inv);
  let c = cs[(rowinfo[m] >> 8u) * ${H2}u + t];
  let o1 = y1 * c.x - y2 * c.y; let o2 = y2 * c.x + y1 * c.y;
  if (isQ) {
    let dst = (hh * u.M + m) * ${HD}u; Q[dst + t] = o1; Q[dst + t + ${H2}u] = o2;
  } else {
    Kt[(hh * ${HD}u + t) * u.M + m] = o1; Kt[(hh * ${HD}u + t + ${H2}u) * u.M + m] = o2;
  }
}`;
};

// CFG post-processing per (target position, codebook), the math of cpuPostProcess in
// tts-worker.js: guided = (1 + g)·logsoftmax(cond) − g·logsoftmax(uncond), pred = argmax of
// logsoftmax(guided) without the mask token (lowest index on ties), score = its log-prob − penalty·c.
// Grid (T, C), one workgroup of 256 per row of V logits (V ≤ 1280).
const POST = `struct U { T: u32, Th: u32, N: u32, V: u32, mask: u32, g: f32, penalty: f32, p0: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> pred: array<u32>;
@group(0) @binding(3) var<storage, read_write> score: array<f32>;
var<workgroup> r2: array<vec2<f32>, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let tt = wg.x; let c = wg.y;
  let cb = tt * u.N + c * u.V; let ub = (u.Th + tt) * u.N + c * u.V;
  var cv: array<f32, 5>; var uv: array<f32, 5>;
  var m = vec2<f32>(-3.0e38);
  for (var k = 0u; k < 5u; k++) {
    let v = t + 256u * k;
    if (v < u.V) { cv[k] = logits[cb + v]; uv[k] = logits[ub + v]; m = max(m, vec2<f32>(cv[k], uv[k])); }
  }
  r2[t] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { r2[t] = max(r2[t], r2[t + s]); } workgroupBarrier(); }
  let mx = r2[0];
  workgroupBarrier();
  var sum = vec2<f32>(0.0);
  for (var k = 0u; k < 5u; k++) { if (t + 256u * k < u.V) { sum += exp(vec2<f32>(cv[k], uv[k]) - mx); } }
  r2[t] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { r2[t] += r2[t + s]; } workgroupBarrier(); }
  let lse = mx + log(r2[0]);
  workgroupBarrier();
  var gv: array<f32, 5>;
  var gm = -3.0e38;
  for (var k = 0u; k < 5u; k++) {
    if (t + 256u * k < u.V) { gv[k] = (1.0 + u.g) * (cv[k] - lse.x) - u.g * (uv[k] - lse.y); gm = max(gm, gv[k]); }
  }
  r2[t] = vec2<f32>(gm, 0.0);
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { r2[t].x = max(r2[t].x, r2[t + s].x); } workgroupBarrier(); }
  let gmax = r2[0].x;
  workgroupBarrier();
  var gs = 0.0;
  for (var k = 0u; k < 5u; k++) { if (t + 256u * k < u.V) { gs += exp(gv[k] - gmax); } }
  r2[t] = vec2<f32>(gs, 0.0);
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { r2[t].x += r2[t + s].x; } workgroupBarrier(); }
  let glse = gmax + log(r2[0].x);
  workgroupBarrier();
  var bs = -3.0e38; var bv = 0xffffffffu;
  for (var k = 0u; k < 5u; k++) {
    let v = t + 256u * k;
    if (v < u.V && v != u.mask) { let lp = gv[k] - glse; if (lp > bs) { bs = lp; bv = v; } }
  }
  r2[t] = vec2<f32>(bs, 0.0); ri[t] = bv;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let o = r2[t + s].x; let oi = ri[t + s];
      if (o > r2[t].x || (o == r2[t].x && oi < ri[t])) { r2[t].x = o; ri[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) { pred[c * u.T + tt] = ri[0]; score[c * u.T + tt] = r2[0].x - u.penalty * f32(c); }
}`;

// ---------------------------------------------------------------- weights source

// Range reads from a URL (dev server); the app passes an OPFS-backed source with the same read().
export const httpSource = (url) => ({
  async read(offset, bytes) {
    const r = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + bytes - 1}` }, cache: 'no-store' });
    if (r.status !== 206) throw new Error(`engine: range request returned ${r.status}`);
    const data = await r.arrayBuffer();
    if (data.byteLength !== bytes) throw new Error(`engine: got ${data.byteLength} of ${bytes} bytes`);
    return data;
  },
});

const res0 = (buffer) => ({ buffer });

const f16ToF32 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

// gate_up columns regrouped per 64 as [gate 64b..64b+63][up 64b..64b+63], b = 0..FF/64, so a
// 128-column tile holds matching gate and up columns (gemmWGSL 'swiglu'). Any dtype: a row of
// the weights (or of the int8 scales) holds 2·FF columns of equal byte width.
function interleaveGateUp(data, rows, FF) {
  const src = new Uint8Array(data), dst = new Uint8Array(src.length);
  const rowBytes = src.length / rows, half = rowBytes / 2, cb = half / (FF / 64); // bytes per 64 columns
  for (let r = 0; r < rows; r++) {
    const o = r * rowBytes;
    for (let b = 0; b < FF / 64; b++) {
      dst.set(src.subarray(o + b * cb, o + (b + 1) * cb), o + 2 * b * cb);
      dst.set(src.subarray(o + half + b * cb, o + half + (b + 1) * cb), o + (2 * b + 1) * cb);
    }
  }
  return dst.buffer;
}

// ---------------------------------------------------------------- engine

export async function compile(device, code, label) {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label });
  const msgs = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
  if (msgs.length) throw new Error(`${label}: ${msgs.map((m) => `line ${m.lineNum}: ${m.message}`).join('; ')}`);
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
  const err = await device.popErrorScope();
  if (err) throw new Error(`${label}: ${err.message}`);
  return pipeline;
}

export class OmniVoiceEngine {
  // manifest: parsed <name>.json; source: { read(offset, bytes) -> ArrayBuffer } over <name>.bin
  // segments, segWait: command buffers per step and whether each waits for the last (step())
  static async load(device, manifest, source, { act = 'f32', onProgress = () => {}, chunk = 32 << 20, segments = 1, segWait = false } = {}) {
    if (manifest.version !== 1) throw new Error(`engine: weights in format ${manifest.version}, this code reads format 1`);
    const e = new OmniVoiceEngine(device, manifest, source, act);
    Object.assign(e, { segments, segWait, chunk });
    await e.loadWeights(onProgress);
    if (manifest.tensors['text_emb.s']) { // int8 rows: per-row scales, 4 bytes per row, kept on the CPU
      const s = manifest.tensors['text_emb.s'];
      e.textScales = new Float32Array(await source.read(s.offset, s.bytes));
    }
    await e.compile();
    return e;
  }

  // Uploads the weights; again after releaseWeights() (pipelines and tuned tiles are kept)
  async loadWeights(onProgress = () => {}) {
    const { device, man: manifest, source, chunk } = this;
    const gpuTensors = Object.entries(manifest.tensors).filter(([n]) => !n.startsWith('text_emb'));
    const total = gpuTensors.reduce((s, [, t]) => s + t.bytes, 0);
    let done = 0, pending = 0;
    for (const [name, t] of gpuTensors) {
      const buf = device.createBuffer({ size: ceil(t.bytes, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: name });
      const gateUp = /\.gate_up(\.s)?$/.test(name);
      if (gateUp && t.bytes > chunk) throw new Error(`engine: ${name} must load in one chunk`);
      for (let off = 0; off < t.bytes; off += chunk) {
        const n = Math.min(chunk, t.bytes - off);
        const data = await source.read(t.offset + off, n);
        device.queue.writeBuffer(buf, off, gateUp ? interleaveGateUp(data, t.shape[0], manifest.config.FF) : data);
        if (name === 'inv_freq') this.invFreq = new Float32Array(data.slice(0));
        done += n;
        onProgress(done / total, name);
        if ((pending += n) >= UPLOAD_BATCH) { await device.queue.onSubmittedWorkDone(); pending = 0; }
      }
      this.w[name] = buf;
    }
  }

  // Frees the weights' GPU memory until loadWeights(); destroy plans made before first
  releaseWeights() { for (const b of Object.values(this.w)) b.destroy(); this.w = {}; }

  constructor(device, manifest, source, act) {
    this.device = device;
    this.cfg = manifest.config;
    this.man = manifest;
    this.source = source;
    this.act = act;
    this.w = {};
    this.p = {};
  }

  async compile() {
    const { H, NH, NKV, HD, FF, C, V, EPS } = this.cfg;
    const d = this.device, t = this.man.tensors, a = this.act;
    const eps = EPS.toExponential();
    const jobs = [
      ['embed', EMBED(C, V, H, t.audio_emb.dtype)], ['norm', NORM(H, eps, false, a)], ['normAdd', NORM(H, eps, true, a)],
      ['final', FINAL(H, eps, a)], ['qkrope', QKROPE(NH, NKV, HD, eps)], ['post', POST],
    ];
    const pipes = await Promise.all(jobs.map(([k, code]) => compile(d, code, k)));
    jobs.forEach(([k], i) => { this.p[k] = pipes[i]; });
    // matmuls: the default tiles until tune() picks per kind
    this.lin = {}; this.att = {};
    await Promise.all([
      ...LIN_KINDS.map(async (k) => { this.lin[k] = await this.linPipe(k, LIN_CANDIDATES.find((c) => this.linOk(k, c))); }),
      ...ATT_KINDS.map(async (k) => { this.att[k] = await this.attPipe(k, ATT_CANDIDATES[k].find((c) => this.attOk(k, c))); }),
    ]);
  }

  // whether this device can use a tile for a linear layer kind / attention kind
  linOk(kind, name) { return linFits(kind, name) && gemmFits(this.device.limits, this.act, parseCfg(name)); }
  attOk(kind, name) { return gemmFits(this.device.limits, 'f32', parseCfg(name), kind); }

  // weight tensor of a linear layer kind (layer 0 stands for all layers: same shapes and dtype)
  linWeight(kind) { return kind === 'head' ? 'head' : `L0.${kind}`; }

  // { pipe, cfg } for a linear layer kind with a tile config, compiled once per (dtype, epilogue,
  // tile); gate_up writes SiLU(gate) · up (activation type), the others f32
  linPipe(kind, cfgName) {
    const wT = this.man.tensors[this.linWeight(kind)].dtype, mode = LIN_MODES[kind] || null;
    const key = `${wT}:${mode || 'plain'}:${cfgName}`;
    this.linCache ??= new Map();
    if (!this.linCache.has(key)) {
      const code = gemmWGSL(wT, this.act, this.act, mode ? this.act : 'f32', parseCfg(cfgName), mode);
      this.linCache.set(key, compile(this.device, code, `lin ${key}`).then((pipe) => ({ pipe, cfg: parseCfg(cfgName), name: cfgName })));
    }
    return this.linCache.get(key);
  }

  // { pipe, cfg } for attention scores ('score') or P·V ('pv') with a tile config: f32 operands
  // and staging, P·V writes the activation type
  attPipe(kind, cfgName) {
    const key = `${kind}:${cfgName}`;
    this.attCache ??= new Map();
    if (!this.attCache.has(key)) {
      const oT = kind === 'pv' ? this.act : 'f32';
      this.attCache.set(key, compile(this.device, gemmWGSL('f32', 'f32', 'f32', oT, parseCfg(cfgName), kind), `attn ${key}`)
        .then((pipe) => ({ pipe, cfg: parseCfg(cfgName), name: cfgName })));
    }
    return this.attCache.get(key);
  }

  // Fastest tiles on this device for a pass with L cond and T uncond rows (layout()), measured
  // with the real weights: linear layers per kind at their row counts, attention at the cond
  // segment. Every candidate runs `reps` times in its own timed pass, after a warm-up, timed on
  // the GPU ('timestamp-query') or else by wall clock. Same results for every tile.
  async tune(L, T, { candidates = LIN_CANDIDATES, attCandidates = ATT_CANDIDATES, reps = 3 } = {}) {
    const d = this.device;
    const t0 = performance.now();
    const { H, NH, NKV, HD, FF, HEAD_N } = this.cfg;
    const { Mr, Th, segs, M } = layout(L, T), n = segs[0].n;
    const shapes = { qkv: [Mr, (NH + 2 * NKV) * HD, H], o: [Mr, H, NH * HD], gate_up: [Mr, 2 * FF, H], down: [Mr, H, FF], head: [2 * Th, HEAD_N, H] };
    // compile every candidate first (in parallel; cached per tile)
    const lps = await Promise.all(LIN_KINDS.map((k) => Promise.all(candidates.filter((c) => this.linOk(k, c)).map((c) => this.linPipe(k, c)))));
    const aps = await Promise.all(ATT_KINDS.map((k) => Promise.all(attCandidates[k].filter((c) => this.attOk(k, c)).map((c) => this.attPipe(k, c)))));
    const ab = this.act === 'f16' ? 2 : 4;
    const mk = (bytes) => d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
    const A = mk(M * Math.max(FF, NH * HD, H) * ab), C = mk(Math.max(M, 2 * Th) * Math.max(2 * FF, HEAD_N) * 4);
    // attention: Q [NH][M][HD], Kt [NKV][HD][M] (also stands in for V), scores, stats (NT ≥ 32)
    const Q = mk(NH * M * HD * 4), Kt = mk(NKV * HD * M * 4), S = mk(NH * n * n * 4), ST = mk(NH * n * (n / 32) * 8);
    const runs = [];
    const add = (kind, lp, buffers, u, grid) => {
      const ub = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(ub, 0, new Uint32Array(u));
      const bg = d.createBindGroup({ layout: lp.pipe.getBindGroupLayout(0), entries: [ub, ...buffers].map((b, binding) => ({ binding, resource: res0(b) })) });
      runs.push({ kind, lp, bg, grid, ub });
    };
    LIN_KINDS.forEach((kind, ki) => {
      const [real, N, K] = shapes[kind];
      const w = this.linWeight(kind);
      for (const lp of lps[ki]) {
        const { MT, NT } = dims(lp.cfg);
        const rows = ceil(real, MT);
        if (N % NT || K % lp.cfg.KT) continue;
        add(kind, lp, [A, this.w[w], C, ...(this.w[`${w}.s`] ? [this.w[`${w}.s`]] : [])],
          [rows, N, K, K / 4, N / 4, kind === 'gate_up' ? FF / 4 : N / 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [N / NT, rows / MT, 1]);
      }
    });
    const scale = new Uint32Array(new Float32Array([1 / Math.sqrt(HD)]).buffer)[0];
    ATT_KINDS.forEach((kind, ki) => {
      for (const lp of aps[ki]) {
        const { MT, NT } = dims(lp.cfg);
        if (kind === 'score') {
          if (PAD % MT || PAD % NT || HD % lp.cfg.KT) continue;
          add(kind, lp, [Q, Kt, S, ST], [n, n, HD, HD / 4, M / 4, n / 4, 0, 0, 0, M * HD / 4, HD * M / 4, n * n / 4, NH / NKV, n, scale, n / NT],
            [n / NT, n / MT, NH]);
        } else {
          if (PAD % MT || HD % NT || PAD % lp.cfg.KT) continue;
          add(kind, lp, [S, Kt, A, ST], [n, HD, n, n / 4, HD / 4, NH * HD / 4, 0, 0, 0, n * n / 4, M * HD / 4, HD / 4, NH / NKV, n, 0, n / PAD],
            [HD / NT, n / MT, NH]);
        }
      }
    });
    const pass = (enc, r, n, timestampWrites) => {
      const p = enc.beginComputePass(timestampWrites ? { timestampWrites } : {});
      p.setPipeline(r.lp.pipe); p.setBindGroup(0, r.bg);
      for (let k = 0; k < n; k++) p.dispatchWorkgroups(...r.grid);
      p.end();
    };
    let times;
    if (d.features.has('timestamp-query')) {
      // warm-up and timed passes in one submission: an idle gap lets the GPU clock down, which
      // made the first candidates measure up to 2× slow on the iPhone
      const qs = d.createQuerySet({ type: 'timestamp', count: 2 * runs.length });
      const rbuf = d.createBuffer({ size: 16 * runs.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const rb = d.createBuffer({ size: 16 * runs.length, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = d.createCommandEncoder();
      runs.forEach((r) => pass(enc, r, 1));
      runs.forEach((r, i) => pass(enc, r, reps, { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 }));
      enc.resolveQuerySet(qs, 0, 2 * runs.length, rbuf, 0); enc.copyBufferToBuffer(rbuf, 0, rb, 0, 16 * runs.length);
      d.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rb.getMappedRange());
      times = runs.map((r, i) => Number(t[2 * i + 1] - t[2 * i]) / 1e6 / reps);
      rb.unmap(); rb.destroy(); rbuf.destroy(); qs.destroy();
    } else {
      // wall clock per candidate, right after its warm-up (includes one submit round trip, the
      // same for every candidate of a kind)
      times = [];
      for (const r of runs) {
        for (const n of [1, reps]) {
          const t = performance.now(), enc = d.createCommandEncoder();
          pass(enc, r, n); d.queue.submit([enc.finish()]);
          await d.queue.onSubmittedWorkDone();
          if (n === reps) times.push((performance.now() - t) / reps);
        }
      }
    }
    const best = {}, report = {};
    runs.forEach((r, i) => {
      const ms = times[i];
      (report[r.kind] ??= {})[r.lp.name] = +ms.toFixed(3);
      if (!best[r.kind] || ms < best[r.kind].ms) best[r.kind] = { ms, lp: r.lp };
    });
    for (const b of [A, C, Q, Kt, S, ST, ...runs.map((r) => r.ub)]) b.destroy();
    for (const k of LIN_KINDS) if (best[k]) this.lin[k] = best[k].lp;
    for (const k of ATT_KINDS) if (best[k]) this.att[k] = best[k].lp;
    this.tuned = Object.fromEntries([...LIN_KINDS.map((k) => [k, this.lin[k].name]), ...ATT_KINDS.map((k) => [k, this.att[k].name])]);
    return { chosen: this.tuned, ms: report, tuneMs: Math.round(performance.now() - t0) };
  }

  // text embedding rows (f32) for the given token ids, read from the weights file
  async textRows(ids) {
    const { H } = this.cfg;
    const t = this.man.tensors.text_emb;
    const bpe = { f32: 4, f16: 2, i8: 1 }[t.dtype];
    const rows = new Map();
    await Promise.all([...new Set(ids)].map(async (id) => {
      const raw = await this.source.read(t.offset + id * H * bpe, H * bpe);
      let row;
      if (t.dtype === 'f32') row = new Float32Array(raw);
      else if (t.dtype === 'f16') row = Float32Array.from(new Uint16Array(raw), (h) => f16ToF32[h]);
      else { const s = this.textScales[id]; row = Float32Array.from(new Int8Array(raw), (q) => Math.fround(q * s)); }
      rows.set(id, row);
    }));
    return rows;
  }

  // tune() once, at the row counts of the first real (L, T)
  async tuneFor(L, T) {
    if (!this.tuning) this.tuning = this.tune(L, T);
    return this.tuning;
  }

  plan(L, T) { return new Plan(this, L, T); }

  destroy() { this.releaseWeights(); }
}

class Plan {
  constructor(e, L, T) {
    const { H, NH, NKV, HD, FF, NL, C, HEAD_N } = e.cfg;
    const d = e.device;
    this.e = e; this.L = L; this.T = T;
    const { Ls, Mr, Th, segs, M } = layout(L, T);
    Object.assign(this, { Ls, Mr, Th, M });
    const QKV = (NH + 2 * NKV) * HD;
    const S = Math.max(...segs.map((s) => s.n));
    const att = dims(e.att.score.cfg), pv = dims(e.att.pv.cfg); // tiles as tuned when the plan is built
    const ab = e.act === 'f16' ? 2 : 4; // bytes per matmul-input element
    const mk = (bytes, label) => d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label });
    const b = this.b = {
      ids: mk(C * M * 4, 'ids'), rowinfo: mk(M * 4, 'rowinfo'), rowmap: mk(2 * Th * 4, 'rowmap'), trows: mk(M * H * 4, 'text rows'),
      cs: mk(M * (HD / 2) * 8, 'rope'),
      x: mk(M * H * 4, 'resid'), h: mk(M * H * ab, 'h'), qkv: mk(M * QKV * 4, 'qkv'),
      q: mk(NH * M * HD * 4, 'q'), kt: mk(NKV * HD * M * 4, 'kt'), v: mk(NKV * M * HD * 4, 'v'),
      s: mk(NH * S * S * 4, 'scores'), st: mk(NH * S * (S / att.NT) * 8, 'score stats'), attn: mk(M * NH * HD * ab, 'attn'), dlt: mk(M * H * 4, 'delta'),
      act: mk(M * FF * ab, 'act'),
      hT: mk(2 * Th * H * ab, 'head_in'), logits: mk(2 * Th * HEAD_N * 4, 'logits'),
    };
    // RoPE table for positions 0..M-1 (cos, sin), angles as torch computes them in fp32
    const cs = new Float32Array(M * HD);
    for (let p = 0; p < M; p++) for (let j = 0; j < HD / 2; j++) {
      const a = Math.fround(p * e.invFreq[j]);
      cs[(p * HD / 2 + j) * 2] = Math.cos(a); cs[(p * HD / 2 + j) * 2 + 1] = Math.sin(a);
    }
    d.queue.writeBuffer(b.cs, 0, cs);
    // head rows: cond targets are cond rows L-T..L-1, uncond targets are rows Ls..Ls+T-1
    const rowmap = new Uint32Array(2 * Th).fill(0xffffffff);
    for (let r = 0; r < T; r++) { rowmap[r] = L - T + r; rowmap[Th + r] = Ls + r; }
    d.queue.writeBuffer(b.rowmap, 0, rowmap);

    // uniforms: one 256-byte slot per dispatch that needs one
    const slots = [];
    const uni = (u32s) => { slots.push(u32s); return slots.length - 1; };
    const steps = [];
    const res = (buffer) => ({ buffer });
    const gemmU = (o) => uni(new Uint32Array([o.M, o.N, o.K, o.lda, o.ldw, o.ldc, o.aOff || 0, o.wOff || 0, o.cOff || 0, o.aZ || 0, o.wZ || 0, o.cZ || 0, o.wDiv || 1, o.p0 || 0, o.p1 || 0, o.p2 || 0]));
    // linear layer: out[rows][..] = in[rows][K] · W[K][N] (+ int8 scales), rows rounded up to the tile
    const linear = (kind, A, wname, Cb, real, N, K, ldc = N / 4) => {
      const { pipe, cfg } = e.lin[kind], { MT, NT } = dims(cfg), rows = ceil(real, MT);
      const ent = [res(A), res(e.w[wname]), res(Cb)];
      if (e.w[`${wname}.s`]) ent.push(res(e.w[`${wname}.s`]));
      steps.push({ pipe, label: wname.replace(/^L[0-9]+\./, 'gemm '), entries: ent, u: gemmU({ M: rows, N, K, lda: K / 4, ldw: N / 4, ldc }), grid: [N / NT, rows / MT, 1] });
    };

    steps.push({ pipe: e.p.embed, entries: [res(b.ids), res(b.rowinfo), res(b.trows), res(e.w.audio_emb), res(b.x)], u: uni(new Uint32Array([M, 0, 0, 0])), grid: [Mr, 1, 1] });
    const scale = new Uint32Array(new Float32Array([1 / Math.sqrt(HD)]).buffer)[0];
    for (let i = 0; i < NL; i++) {
      const W = (n) => e.w[`L${i}.${n}`];
      if (i === 0) steps.push({ pipe: e.p.norm, entries: [res(b.x), res(W('ln_in')), res(b.h)], grid: [Mr, 1, 1] });
      else steps.push({ pipe: e.p.normAdd, entries: [res(b.x), res(W('ln_in')), res(b.h), res(b.dlt)], grid: [Mr, 1, 1] });
      linear('qkv', b.h, `L${i}.qkv`, b.qkv, Mr, QKV, H);
      steps.push({ pipe: e.p.qkrope, entries: [res(b.qkv), res(b.rowinfo), res(W('q_norm')), res(W('k_norm')), res(b.cs), res(b.q), res(b.kt), res(b.v)],
        u: uni(new Uint32Array([M, 0, 0, 0])), grid: [Mr, NH + 2 * NKV, 1] });
      for (const { s0, n, valid } of segs) {
        // scores[h] = scale · Q[h][s0..s0+n) · Kt[h/2][:, s0..s0+n)    [n][n], + row stats per NT columns
        const tiles = n / att.NT;
        steps.push({ pipe: e.att.score.pipe, label: 'attn scores', entries: [res(b.q), res(b.kt), res(b.s), res(b.st)],
          u: gemmU({ M: n, N: n, K: HD, lda: HD / 4, ldw: M / 4, ldc: n / 4, aOff: s0 * HD / 4, wOff: s0 / 4, aZ: M * HD / 4, wZ: HD * M / 4, cZ: n * n / 4, wDiv: NH / NKV, p0: valid, p1: scale, p2: tiles }),
          grid: [tiles, n / att.MT, NH] });
        // attn[s0.., h·HD..] = softmax(scores[h]) · V[h/2][s0..s0+n)    [n][HD]
        steps.push({ pipe: e.att.pv.pipe, label: 'attn P·V', entries: [res(b.s), res(b.v), res(b.attn), res(b.st)],
          u: gemmU({ M: n, N: HD, K: n, lda: n / 4, ldw: HD / 4, ldc: NH * HD / 4, wOff: s0 * HD / 4, cOff: s0 * NH * HD / 4, aZ: n * n / 4, wZ: M * HD / 4, cZ: HD / 4, wDiv: NH / NKV, p0: valid, p2: tiles }),
          grid: [HD / pv.NT, n / pv.MT, NH] });
      }
      linear('o', b.attn, `L${i}.o`, b.dlt, Mr, H, NH * HD);
      steps.push({ pipe: e.p.normAdd, entries: [res(b.x), res(W('ln_post')), res(b.h), res(b.dlt)], grid: [Mr, 1, 1] });
      linear('gate_up', b.h, `L${i}.gate_up`, b.act, Mr, 2 * FF, H, FF / 4); // SiLU(gate) · up
      linear('down', b.act, `L${i}.down`, b.dlt, Mr, H, FF);
    }
    steps.push({ pipe: e.p.final, entries: [res(b.x), res(b.dlt), res(e.w.norm), res(b.rowmap), res(b.hT)], grid: [2 * Th, 1, 1] });
    linear('head', b.hT, 'head', b.logits, 2 * Th, HEAD_N, H);

    this.uniform = d.createBuffer({ size: Math.max(1, slots.length) * UNI, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'uniforms' });
    const all = new Uint8Array(slots.length * UNI);
    slots.forEach((s, k) => all.set(new Uint8Array(s.buffer), k * UNI));
    d.queue.writeBuffer(this.uniform, 0, all);
    this.steps = steps.map((s) => {
      const entries = [];
      let k = 0;
      if (s.u !== undefined) entries.push({ binding: k++, resource: { buffer: this.uniform, offset: s.u * UNI, size: 64 } });
      for (const r of s.entries) entries.push({ binding: k++, resource: r });
      return { pipe: s.pipe, label: s.label || s.pipe.label, grid: s.grid, bg: d.createBindGroup({ layout: s.pipe.getBindGroupLayout(0), entries }) };
    });
  }

  // condIds: Int32Array [C][L] (the full cond sequence), audioStart: first audio position.
  // Uploads ids, row info and the text embedding rows (read from the weights file).
  async setInputs(condIds, audioStart) {
    const { C, H } = this.e.cfg;
    const { L, T, Ls, M } = this;
    const ids = new Int32Array(C * M), info = new Uint32Array(M);
    for (let c = 0; c < C; c++) {
      ids.set(condIds.subarray(c * L, (c + 1) * L), c * M);
      ids.set(condIds.subarray(c * L + L - T, (c + 1) * L), c * M + Ls);
    }
    for (let p = 0; p < L; p++) info[p] = (p << 8) | 2 | (p >= audioStart ? 1 : 0);
    for (let r = 0; r < T; r++) info[Ls + r] = (r << 8) | 3;
    const textIds = Array.from(condIds.subarray(0, audioStart));
    const rows = await this.e.textRows(textIds);
    const trows = new Float32Array(M * H);
    textIds.forEach((id, p) => trows.set(rows.get(id), p * H));
    const q = this.e.device.queue;
    q.writeBuffer(this.b.ids, 0, ids); q.writeBuffer(this.b.rowinfo, 0, info); q.writeBuffer(this.b.trows, 0, trows);
    this.ids = ids;
  }

  // target tokens [C][T] (any integer array; mask id where not committed yet) -> cond and uncond rows
  setTargets(tokens) {
    const { C } = this.e.cfg;
    const { L, T, Ls, M, ids } = this;
    for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
      const v = Number(tokens[c * T + t]);
      ids[c * M + L - T + t] = v; ids[c * M + Ls + t] = v;
    }
    this.e.device.queue.writeBuffer(this.b.ids, 0, ids);
  }

  // One diffusion step: forward pass + CFG post-processing on the GPU; fills pred (Int32Array
  // [C·T]) and scores (Float32Array [C·T]) like cpuPostProcess.
  async step(pred, scores, { guidanceScale, layerPenalty }) {
    const { C, V, HEAD_N, MASK_ID } = this.e.cfg;
    const d = this.e.device, n = C * this.T;
    if (!this.post) {
      const mk = (usage, bytes) => d.createBuffer({ size: bytes, usage });
      this.post = {
        u: mk(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 32),
        pred: mk(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, n * 4),
        score: mk(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, n * 4),
        rb: mk(GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, n * 8),
      };
      this.post.bg = d.createBindGroup({ layout: this.e.p.post.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.post.u } }, { binding: 1, resource: { buffer: this.b.logits } },
        { binding: 2, resource: { buffer: this.post.pred } }, { binding: 3, resource: { buffer: this.post.score } }] });
    }
    const p = this.post;
    const uv = new ArrayBuffer(32), u32 = new Uint32Array(uv), f32 = new Float32Array(uv);
    u32.set([this.T, this.Th, HEAD_N, V, MASK_ID]); f32[5] = guidanceScale; f32[6] = layerPenalty;
    d.queue.writeBuffer(p.u, 0, uv);
    // The pass goes to the GPU in `segments` command buffers: one ~100 ms buffer (~700 ms on an
    // iPhone) left the browser no GPU time to draw, the page stalled until the step was done.
    // Same dispatches in the same order, so the same results.
    const st = this.steps, per = Math.ceil(st.length / Math.max(1, Math.min(this.e.segments, st.length)));
    for (let a = 0; a < st.length; a += per) {
      const enc = d.createCommandEncoder(), last = a + per >= st.length;
      this.encode(enc, a, Math.min(st.length, a + per));
      if (last) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.e.p.post); pass.setBindGroup(0, p.bg); pass.dispatchWorkgroups(this.T, C);
        pass.end();
        enc.copyBufferToBuffer(p.pred, 0, p.rb, 0, n * 4); enc.copyBufferToBuffer(p.score, 0, p.rb, n * 4, n * 4);
      }
      d.queue.submit([enc.finish()]);
      if (!last && this.e.segWait) await d.queue.onSubmittedWorkDone();
    }
    await p.rb.mapAsync(GPUMapMode.READ);
    const m = p.rb.getMappedRange();
    pred.set(new Int32Array(m, 0, n)); scores.set(new Float32Array(m, n * 4, n));
    p.rb.unmap();
  }

  encode(enc, a = 0, b = this.steps.length) {
    const pass = enc.beginComputePass();
    for (let i = a; i < b; i++) { const s = this.steps[i]; pass.setPipeline(s.pipe); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.grid); }
    pass.end();
  }

  async run() {
    const d = this.e.device;
    const enc = d.createCommandEncoder();
    this.encode(enc);
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
  }

  // GPU time per kernel class (ms, summed over the pass), every dispatch in its own timed
  // pass; needs the device's 'timestamp-query' feature. Separate passes add some overhead,
  // so the total is a bit above run()'s.
  async profile() {
    const d = this.e.device;
    if (!d.features.has('timestamp-query')) return null;
    const n = this.steps.length;
    const qs = d.createQuerySet({ type: 'timestamp', count: 2 * n });
    const res = d.createBuffer({ size: 16 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rb = d.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    this.steps.forEach((s, i) => {
      const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      p.setPipeline(s.pipe); p.setBindGroup(0, s.bg); p.dispatchWorkgroups(...s.grid); p.end();
    });
    enc.resolveQuerySet(qs, 0, 2 * n, res, 0);
    enc.copyBufferToBuffer(res, 0, rb, 0, 16 * n);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rb.getMappedRange());
    const by = {};
    this.steps.forEach((s, i) => { by[s.label] = (by[s.label] || 0) + Number(t[2 * i + 1] - t[2 * i]) / 1e6; });
    rb.unmap(); rb.destroy(); res.destroy(); qs.destroy();
    for (const k of Object.keys(by)) by[k] = +by[k].toFixed(2);
    return by;
  }

  async read(buf, bytes = buf.size) {
    const d = this.e.device;
    const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, bytes); d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    return out;
  }

  destroy() {
    for (const b of Object.values(this.b)) b.destroy();
    this.uniform.destroy();
    if (this.post) for (const k of ['u', 'pred', 'score', 'rb']) this.post[k].destroy();
  }
}
export { gemmWGSL, parseCfg, dims, LIN_CANDIDATES, ATT_CANDIDATES }; // for kernel tests
