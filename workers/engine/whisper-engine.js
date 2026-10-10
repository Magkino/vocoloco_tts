// VocoLoco engine: Whisper large-v3-turbo speech recognition on WebGPU (the voice wizard's
// automatic transcription). Weights: a JSON manifest (tensors, config, vocab) and one binary file.
// The checkpoint is stored in fp16, so f16 weights are its exact values; they are read as f32 and
// all math is float32, like the PyTorch reference (eager attention, exact GELU).
//
//   pcm 16 kHz ─ log-mel [3000][128] (CPU, float64, rounded to f32 where the reference rounds)
//   ─ conv1 + GELU ─ conv2 (stride 2) + GELU + positions ─ 32 pre-LN layers ─ LayerNorm
//   ─ cross-attention K | V per decoder layer ─ greedy decoding: 4 pre-LN layers, KV cache
//
// Activations are time-major [rows][channels]; GEMMs and convolutions are the voice encoder's
// convWGSL, attention over the 1500 encoder frames its score / pv kernels. Encoder weights go to
// the GPU layer by layer and are dropped after it (phones: ~40 MB at a time); the decoder's stay
// while a transcription runs (every token uses all of them). `resident` keeps all of them (desktop).

import { compile, gemmWGSL, parseCfg, dims } from './omnivoice-engine.js';
import { convWGSL, QKVSPLIT, ATT_SCORE, ATT_PV } from './audio-encoder.js';
import { GELU } from './gelu.js';

const UNI = 256;
const HEAD_GROUP = 4;          // encoder attention heads per pass (bounds the score buffer)
const UPLOAD_BATCH = 64 << 20; // weight bytes written before waiting (see AudioEncoder.upload)
const TILE = '16x16r4v1k16';   // GEMM tile: 64 × 64 outputs per workgroup
const GPU_SKIP = /^d\.(tok|pos)/; // read on the CPU (embedding rows, scales, positions)
const EPI = {
  raw: { add: false, out: 'y' }, gelu: { add: false, out: 'gelu(y)' }, rmw: { add: true, out: 'y' },
  geluAdd: { add: false, out: 'gelu(y) + OUT[o]' }, // the stem: OUT holds the positions
};

// Rows of W channels: Y = LayerNorm(X) · g + b, in PyTorch's order ((x·r + (−r·mean)) · g + b,
// r = 1 / sqrt(var + eps), biased variance). One workgroup of 64 threads per row.
const LN = (W, eps) => `struct U { rows: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> G: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Bb: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<vec4<f32>>;
var<workgroup> red: array<f32, 64>;
fn total(t: u32) -> f32 {
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  if (wg.x >= u.rows) { return; }
  let base = wg.x * ${W / 4}u;
  var s = 0.0;
  for (var i = t; i < ${W / 4}u; i += 64u) { let x = X[base + i]; s += (x.x + x.y) + (x.z + x.w); }
  red[t] = s;
  workgroupBarrier();
  let mean = total(t) / ${W}.0;
  var q = 0.0;
  for (var i = t; i < ${W / 4}u; i += 64u) { let d = X[base + i] - vec4<f32>(mean); q += dot(d, d); }
  red[t] = q;
  workgroupBarrier();
  let r = 1.0 / sqrt(total(t) / ${W}.0 + ${eps});
  let sh = -r * mean;
  for (var i = t; i < ${W / 4}u; i += 64u) { Y[base + i] = (X[base + i] * r + vec4<f32>(sh)) * G[i] + Bb[i]; }
}`;

// One input row (decoding): OUT[n] = epi(B[n] + Σ_k X[k] · W[k][n]). A workgroup computes 64
// columns (16 vec4) in 16 slices of K, each summed in order, then added in a fixed tree order
// (deterministic). K a multiple of 16, N of 64. i8 weights as convWGSL's (scales at binding 5).
const GEMV = (wT, epi) => `${wT === 'f32' ? '' : 'enable f16;\n'}struct U { N: u32, K: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: ${wT === 'i8' ? 'array<u32>' : `array<vec4<${wT}>>`};
@group(0) @binding(3) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> OUT: array<vec4<f32>>;${wT === 'i8' ? `
@group(0) @binding(5) var<storage, read> S: array<vec4<f16>>;
fn wq(i: u32, s: u32) -> vec4<f32> {
  let p = W[i];
  let q = vec4<i32>(bitcast<i32>(p << 24u) >> 24u, bitcast<i32>(p << 16u) >> 24u, bitcast<i32>(p << 8u) >> 24u, bitcast<i32>(p) >> 24u);
  return vec4<f32>(q) * vec4<f32>(S[s]);
}` : ''}
${GELU}
var<workgroup> part: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let N4 = u.N / 4u; let c = wg.x * 16u + lid.x; let ks = u.K / 16u; let i = lid.y * 16u + lid.x;
  var acc = vec4<f32>(0.0);
  for (var k = lid.y * ks; k < (lid.y + 1u) * ks; k++) { acc = fma(vec4<f32>(X[k]), ${wT === 'i8' ? 'wq(k * N4 + c, (k / 32u) * N4 + c)' : 'vec4<f32>(W[k * N4 + c])'}, acc); }
  part[i] = acc;
  workgroupBarrier();
  for (var s = 8u; s > 0u; s >>= 1u) {
    if (lid.y < s) { part[i] += part[i + s * 16u]; }
    workgroupBarrier();
  }
  if (lid.y == 0u) {
    var y = part[lid.x] + B[c];${epi.add ? '\n    y += OUT[c];' : ''}
    OUT[c] = ${epi.out};
  }
}`;

// The new token's k and v (columns H..3H of its q | k | v row) into the self-attention caches at row pos
const KVPUT = (H) => `struct U { pos: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> QKV: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Kc: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Vc: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${H / 4}u) { return; }
  Kc[u.pos * ${H / 4}u + i] = QKV[${H / 4}u + i];
  Vc[u.pos * ${H / 4}u + i] = QKV[${H / 2}u + i];
}`;

// Attention of one query row over L keys, per head (workgroup): s = (q · scale) · k, p = softmax(s)
// (exp(s − max) / sum, like torch), out = p · v. K and V rows have stride ks4 vec4s, from koff4 /
// voff4 (the decoder's caches, or the cross-attention K | V rows).
const ATT1 = (HD, maxL) => `struct U { L: u32, ks4: u32, koff4: u32, voff4: u32, qoff4: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> s: array<f32, ${maxL}>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h = wg.x;
  var q: array<vec4<f32>, ${HD / 4}>;
  for (var i = 0u; i < ${HD / 4}u; i++) { q[i] = Q[u.qoff4 + h * ${HD / 4}u + i] * u.scale; }
  var mx = -3.402823e38;
  for (var j = t; j < u.L; j += 256u) {
    var acc = 0.0;
    for (var i = 0u; i < ${HD / 4}u; i++) { acc += dot(q[i], K[u.koff4 + j * u.ks4 + h * ${HD / 4}u + i]); }
    s[j] = acc;
    mx = max(mx, acc);
  }
  red[t] = mx;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (t < k) { red[t] = max(red[t], red[t + k]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = t; j < u.L; j += 256u) { let e = exp(s[j] - m); s[j] = e; sum += e; }
  red[t] = sum;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (t < k) { red[t] += red[t + k]; } workgroupBarrier(); }
  let tot = red[0];
  workgroupBarrier();
  let d = t % ${HD}u; let part = t / ${HD}u;
  var o = 0.0;
  for (var j = part; j < u.L; j += ${256 / HD}u) { o += (s[j] / tot) * V[(u.voff4 + j * u.ks4 + h * ${HD / 4}u) * 4u + d]; }
  red[t] = o;
  workgroupBarrier();
  if (t < ${HD}u) { O[h * ${HD}u + t] = (red[t] + red[t + ${HD}u]) + (red[t + ${2 * HD}u] + red[t + ${3 * HD}u]); }
}`;

// ─── Log-mel front end (CPU) ────────────────────────────────────────────────
// Like the reference's float64 recipe: f32 samples, zero-padded to 30 s; reflect padding of n_fft/2;
// periodic Hann window (the manifest's f32 values); |rfft|² in float64; mel = power · filters,
// log10(max(mel, f32(1e-10))) in float64, rounded to f32; then in f32 as torch does: floor at
// max − 8, (x + 4) / 4. The last frame is dropped. Frames of zeros are not transformed.
function logMel(pcm, window, filters, { n_samples: NS, n_fft: NF, hop, n_frames: T, log_floor: floor, dynamic_range: range }, nMels) {
  const x = new Float64Array(NS + NF); // reflect-padded
  const P = NF / 2, n = Math.min(pcm.length, NS);
  for (let i = 0; i < n; i++) x[P + i] = pcm[i];
  for (let i = 1; i <= P; i++) { x[P - i] = x[P + i]; x[P + NS - 1 + i] = x[P + NS - 1 - i]; }
  const nb = NF / 2 + 1, cos = new Float64Array(NF), sin = new Float64Array(NF);
  for (let i = 0; i < NF; i++) { cos[i] = Math.cos(2 * Math.PI * i / NF); sin[i] = Math.sin(2 * Math.PI * i / NF); }
  // per mel bin, its non-zero filter rows (adding the zero terms would not change the sums)
  const spans = [];
  for (let m = 0; m < nMels; m++) {
    let a = 0, b = nb;
    while (a < nb && filters[a * nMels + m] === 0) a++;
    while (b > a && filters[(b - 1) * nMels + m] === 0) b--;
    spans.push([a, b]);
  }
  const lg = new Float32Array(T * nMels), fr = new Float64Array(NF), pw = new Float64Array(nb);
  const zeroLog = Math.fround(Math.log10(Math.fround(floor)));
  for (let t = 0; t < T; t++) {
    let any = false;
    for (let i = 0; i < NF; i++) { const v = x[t * hop + i]; fr[i] = v * window[i]; if (v !== 0) any = true; }
    if (!any) { lg.fill(zeroLog, t * nMels, (t + 1) * nMels); continue; }
    for (let k = 0; k < nb; k++) {
      let re = 0, im = 0;
      for (let i = 0, idx = 0; i < NF; i++) {
        re += fr[i] * cos[idx]; im -= fr[i] * sin[idx];
        idx += k; if (idx >= NF) idx -= NF; // k · i mod NF
      }
      pw[k] = re * re + im * im;
    }
    for (let m = 0; m < nMels; m++) {
      let s = 0;
      for (let k = spans[m][0]; k < spans[m][1]; k++) s += pw[k] * filters[k * nMels + m];
      lg[t * nMels + m] = Math.fround(Math.log10(Math.max(s, Math.fround(floor))));
    }
  }
  let mx = -Infinity;
  for (const v of lg) if (v > mx) mx = v;
  const fl = Math.fround(mx - range);
  for (let i = 0; i < lg.length; i++) lg[i] = Math.fround(Math.fround(Math.max(lg[i], fl) + 4) / 4);
  return lg;
}

// ─── Tokens to text ─────────────────────────────────────────────────────────
// GPT-2's byte-level alphabet: the unicode character standing for each byte
function byteDecoder() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let k = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + k++); }
  return new Map(cs.map((c, i) => [String.fromCharCode(c), bs[i]]));
}

// f16 bits -> number (exact)
function half(h) {
  const e = (h >> 10) & 31, f = h & 1023, s = h & 32768 ? -1 : 1;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1024 + f) * 2 ** (e - 25);
}

export class WhisperEngine {
  // manifest: parsed whisper-<dtype>.json; source: { read(offset, bytes) } over its .bin. resident:
  // keep all weights on the GPU between transcriptions (desktop).
  static async load(device, manifest, source, { resident = false } = {}) {
    if (manifest.version !== 1) throw new Error(`whisper: weights in format ${manifest.version}, this code reads format 1`);
    const e = new WhisperEngine(device, manifest, source, resident);
    await e.compile();
    const rd = async (name) => {
      const t = manifest.tensors[name];
      if (t.dtype !== 'f32') throw new Error(`whisper: ${name} must be f32`);
      return new Float32Array(await source.read(t.offset, t.bytes));
    };
    [e.melWindow, e.melFilters, e.dpos] = await Promise.all(['mel.window', 'mel.filters', 'd.pos'].map(rd));
    e.window64 = Float64Array.from(e.melWindow);
    e.filters64 = Float64Array.from(e.melFilters);
    if (manifest.tensors['d.tok.s']) e.tokScale = await rd('d.tok.s');
    if (resident) await e.upload(Object.keys(manifest.tensors).filter((n) => /^[ed]\./.test(n) && !GPU_SKIP.test(n)));
    return e;
  }

  constructor(device, manifest, source, resident) {
    Object.assign(this, { device, man: manifest, cfg: manifest.config, source, resident, w: {}, p: {} });
    this.gen = this.cfg.generation;
    this.zeros = device.createBuffer({ size: this.cfg.vocab_padded * 4, usage: GPUBufferUsage.STORAGE });
    this.bytes = byteDecoder();
  }

  async compile() {
    const c = this.cfg, d = this.device, H = c.d_model;
    const wTypes = new Set(Object.entries(this.man.tensors).filter(([n]) => /\.w$/.test(n)).map(([, t]) => t.dtype));
    for (const t of wTypes) if (!['f32', 'f16', 'i8'].includes(t)) throw new Error(`whisper: ${t} weights are not supported`);
    const jobs = [['ln', LN(H, c.layer_norm_eps.toExponential())], ['kvput', KVPUT(H)],
      ['att1', ATT1(c.head_dim, c.max_source_positions)], ['qkv', QKVSPLIT(c.heads, c.head_dim)],
      ['score', gemmWGSL('f32', 'f32', 'f32', 'f32', parseCfg(ATT_SCORE), 'score')], ['pv', gemmWGSL('f32', 'f32', 'f32', 'f32', parseCfg(ATT_PV), 'pv')]];
    for (const wT of wTypes) {
      for (const [k, epi] of Object.entries(EPI)) jobs.push([`conv.${k}.${wT}`, convWGSL(parseCfg(TILE), k, { wT, epi })]);
      for (const k of ['raw', 'gelu', 'rmw']) jobs.push([`gemv.${k}.${wT}`, GEMV(wT, EPI[k])]);
    }
    const pipes = await Promise.all(jobs.map(([k, code]) => compile(d, code, `whisper ${k}`)));
    jobs.forEach(([k], i) => { this.p[k] = pipes[i]; });
    this.tile = dims(parseCfg(TILE));
  }

  // the bytes of the tensors not on the GPU yet, read while the GPU works (for upload)
  prefetch(names) {
    return Promise.all(names.map((n) => (this.w[n] ? null : this.source.read(this.man.tensors[n].offset, this.man.tensors[n].bytes))));
  }

  // names' tensors to the GPU (data: their bytes from prefetch, else read here)
  async upload(names, data = null) {
    let pending = 0;
    for (const [k, name] of names.entries()) {
      if (this.w[name]) continue;
      const t = this.man.tensors[name];
      // COPY_SRC: e.pos is copied into the stem's output, which then adds the convolution to it
      const buf = this.device.createBuffer({ size: Math.ceil(t.bytes / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label: `whisper ${name}` });
      this.device.queue.writeBuffer(buf, 0, data?.[k] ?? await this.source.read(t.offset, t.bytes));
      this.w[name] = buf;
      if ((pending += t.bytes) >= UPLOAD_BATCH) { await this.device.queue.onSubmittedWorkDone(); pending = 0; }
    }
  }

  drop(names) {
    for (const n of names) { this.w[n]?.destroy(); delete this.w[n]; }
  }

  // 16 kHz mono pcm -> { text, lang, ids, ms }. debug: names of intermediates to return (mel,
  // e_stem, e_L0, e_L15, e_out, logits_first: the logits of the first 4 positions). force: token
  // ids to feed instead of the greedy choices (teacher forcing, for parity tests).
  async transcribe(pcm, { debug = [], force = null } = {}) {
    const c = this.cfg, d = this.device, H = c.d_model, F = c.max_source_positions;
    const keep = [];
    const mk = (bytes, label) => { const b = d.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label }); keep.push(b); return b; };
    const out = { ms: {} };
    const tap = async (name, buf, floats) => { if (debug.includes(name)) out[name] = await this.read(buf, floats); };
    let t0 = performance.now();
    try {
      const mel = logMel(pcm, this.window64, this.filters64, c.mel, c.n_mels);
      if (debug.includes('mel')) out.mel = mel;
      out.ms.mel = Math.round(performance.now() - t0);
      t0 = performance.now();
      const enc = await this.encode(mel, mk, tap);
      out.ms.encoder = Math.round(performance.now() - t0);
      t0 = performance.now();
      Object.assign(out, await this.decode(enc, mk, debug, force));
      out.ms.decoder = Math.round(performance.now() - t0);
      return out;
    } finally {
      await d.queue.onSubmittedWorkDone();
      for (const b of keep) b.destroy();
    }
  }

  // mel [3000][128] -> the encoder output [1500][H] (a buffer)
  async encode(mel, mk, tap) {
    const c = this.cfg, d = this.device, H = c.d_model, F = c.max_source_positions, T = c.mel.n_frames;
    const NH = c.heads, HD = c.head_dim, Mp = Math.ceil(F / 64) * 64;
    const X = mk(mel.byteLength, 'whisper mel');
    d.queue.writeBuffer(X, 0, mel);
    const h = mk(F * H * 4, 'whisper h'), hn = mk(F * H * 4, 'whisper hn');
    // stem
    const all = Object.keys(this.man.tensors);
    const stem = all.filter((n) => /^e\.(c1\.|c2\.|pos$)/.test(n));
    const layers = Array.from({ length: c.encoder_layers }, (_, i) => all.filter((n) => n.startsWith(`e.L${i}.`)));
    const last = ['e.ln.g', 'e.ln.b', ...this.decoderNames()]; // read during the last layer
    await this.upload(stem);
    let next = this.prefetch(layers[0]);
    {
      const c1 = mk(T * H * 4, 'whisper c1');
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(this.w['e.pos'], 0, h, 0, F * H * 4);
      d.queue.submit([enc.finish()]);
      const P = this.prog();
      this.conv(P, 'gelu', X, 'e.c1.w', 'e.c1.b', c1, { M: T, N: H, cin: c.conv1.cin, taps: c.conv1.kernel, pad: c.conv1.pad });
      this.conv(P, 'geluAdd', c1, 'e.c2.w', 'e.c2.b', h, { M: F, N: H, cin: c.conv2.cin, taps: c.conv2.kernel, stride: c.conv2.stride, pad: c.conv2.pad, xlen: T });
      await this.submit(P);
      c1.destroy();
    }
    if (!this.resident) this.drop(stem);
    await tap('e_stem', h, F * H);
    // layers
    const qkv = mk(F * 3 * H * 4, 'whisper qkv'), ff = mk(F * c.ffn * 4, 'whisper ff'), att = mk(Mp * H * 4, 'whisper att');
    const q = mk(NH * Mp * HD * 4, 'whisper q'), kt = mk(NH * HD * Mp * 4, 'whisper kt'), v = mk(NH * Mp * HD * 4, 'whisper v');
    const G = Math.min(HEAD_GROUP, NH), sc = mk(G * Mp * Mp * 4, 'whisper scores'), stt = mk(G * Mp * (Mp / 64) * 8, 'whisper score stats');
    const scale = new Uint32Array(new Float32Array([c.q_scale]).buffer)[0];
    const att0 = dims(parseCfg(ATT_SCORE)), pv0 = dims(parseCfg(ATT_PV)), tiles = Mp / att0.NT;
    for (let i = 0; i < c.encoder_layers; i++) {
      const Lw = (n) => `e.L${i}.${n}`;
      await this.upload(layers[i], await next);
      next = this.prefetch(layers[i + 1] ?? last); // read while this layer runs
      const P = this.prog();
      this.ln(P, h, Lw('ln1'), hn, F);
      this.conv(P, 'raw', hn, Lw('qkv.w'), Lw('qkv.b'), qkv, { M: F, N: 3 * H, cin: H });
      P.add(this.p.qkv, [qkv, q, kt, v], [Mp], [F, 3 * NH, 1]);
      for (let g0 = 0; g0 < NH; g0 += G) {
        P.add(this.p.score, [q, kt, sc, stt], [Mp, Mp, HD, HD / 4, Mp / 4, Mp / 4, g0 * Mp * HD / 4, g0 * HD * Mp / 4, 0, Mp * HD / 4, HD * Mp / 4, Mp * Mp / 4, 1, F, scale, tiles],
          [tiles, Mp / att0.MT, G]);
        P.add(this.p.pv, [sc, v, att, stt], [Mp, HD, Mp, Mp / 4, HD / 4, H / 4, 0, g0 * Mp * HD / 4, g0 * HD / 4, Mp * Mp / 4, Mp * HD / 4, HD / 4, 1, F, 0, tiles],
          [HD / pv0.NT, Mp / pv0.MT, G]);
      }
      this.conv(P, 'rmw', att, Lw('o.w'), Lw('o.b'), h, { M: F, N: H, cin: H });
      this.ln(P, h, Lw('ln2'), hn, F);
      this.conv(P, 'gelu', hn, Lw('ff1.w'), Lw('ff1.b'), ff, { M: F, N: c.ffn, cin: H });
      this.conv(P, 'rmw', ff, Lw('ff2.w'), Lw('ff2.b'), h, { M: F, N: H, cin: c.ffn });
      await this.submit(P);
      if (!this.resident) this.drop(layers[i]);
      if (i === 0) await tap('e_L0', h, F * H);
      if (i === 15) await tap('e_L15', h, F * H);
    }
    for (const b of [qkv, ff, att, q, kt, v, sc, stt]) b.destroy();
    const encOut = mk(F * H * 4, 'whisper enc');
    await this.upload(last, await next); // the decoder's weights too (decode's upload skips them)
    const P = this.prog();
    this.ln(P, h, 'e.ln', encOut, F);
    await this.submit(P);
    if (!this.resident) this.drop(['e.ln.g', 'e.ln.b']);
    await tap('e_out', encOut, F * H);
    return encOut;
  }

  // Greedy decoding as the app always did (asr-worker.js): <|startoftranscript|>; the language
  // token by argmax over the language ids; <|transcribe|> <|notimestamps|> forced; then text with
  // suppress_tokens (minus <|transcribe|>) on every step and begin_suppress_tokens on the first.
  async decode(enc, mk, debug, force) {
    const c = this.cfg, g = this.gen, d = this.device, H = c.d_model, F = c.max_source_positions, NL = c.decoder_layers;
    const VP = c.vocab_padded, V = c.vocab_size, maxIds = 1 + g.max_new_tokens;
    const names = this.decoderNames();
    await this.upload(names);
    try {
      // cross-attention K | V of every layer, once
      const ckv = [];
      {
        const P = this.prog();
        for (let l = 0; l < NL; l++) {
          ckv.push(mk(F * 2 * H * 4, `whisper ckv${l}`));
          this.conv(P, 'raw', enc, `d.L${l}.ca.kv.w`, `d.L${l}.ca.kv.b`, ckv[l], { M: F, N: 2 * H, cin: H });
        }
        await this.submit(P);
      }
      const kc = [], vc = [];
      for (let l = 0; l < NL; l++) { kc.push(mk(c.max_target_positions * H * 4, `whisper kc${l}`)); vc.push(mk(c.max_target_positions * H * 4, `whisper vc${l}`)); }
      const x = mk(H * 4, 'whisper x'), xn = mk(H * 4, 'whisper xn'), qkv = mk(3 * H * 4, 'whisper dqkv'), att = mk(H * 4, 'whisper datt');
      const cq = mk(H * 4, 'whisper cq'), ff = mk(c.ffn * 4, 'whisper dff'), logits = mk(VP * 4, 'whisper logits');
      const scale = new Uint32Array(new Float32Array([c.q_scale]).buffer)[0];
      const ids = [g.decoder_start_token_id];
      const langIds = Object.values(g.lang_to_id).sort((a, b) => a - b), suppress = g.app_suppress_tokens, begin = g.begin_suppress_tokens;
      const forced = [g.task_to_id.transcribe, g.no_timestamps_token_id];
      const first = [];
      for (let pos = 0; pos < c.max_target_positions && ids.length < maxIds; pos++) {
        const n = ids.length;
        const needLogits = n === 1 || n >= 4 || (debug.includes('logits_first') && pos < 4);
        await this.step(ids[n - 1], pos, { x, xn, qkv, att, cq, ff, logits, ckv, kc, vc, scale }, needLogits);
        if (!needLogits) { ids.push(force ? force[n] : forced[n - 2]); continue; }
        const lg = await this.read(logits, V);
        if (debug.includes('logits_first') && pos < 4) first.push(lg.slice());
        if (n === 2 || n === 3) { ids.push(force ? force[n] : forced[n - 2]); continue; }
        let next;
        if (n === 1) next = argmax(lg, langIds);
        else {
          for (const id of suppress) lg[id] = -Infinity;
          if (n === 4) for (const id of begin) lg[id] = -Infinity;
          next = argmax(lg, null);
        }
        if (force) { if (n >= force.length) break; next = force[n]; }
        ids.push(next);
        if (next === g.eos_token_id) break;
      }
      const res = { ids, lang: Object.keys(g.lang_to_id).find((k) => g.lang_to_id[k] === ids[1])?.slice(2, -2) ?? null, text: this.text(ids) };
      if (debug.includes('logits_first')) res.logits_first = first;
      return res;
    } finally {
      await d.queue.onSubmittedWorkDone();
      if (!this.resident) this.drop(names);
    }
  }

  // one decoder position: token `id` at `pos` through the layers; logits (if wanted) into B.logits
  async step(id, pos, B, wantLogits) {
    const c = this.cfg, H = c.d_model, F = c.max_source_positions, NH = c.heads, HD = c.head_dim, NL = c.decoder_layers;
    const x0 = await this.embed(id, pos);
    this.device.queue.writeBuffer(B.x, 0, x0);
    const P = this.prog();
    for (let l = 0; l < NL; l++) {
      const Lw = (n) => `d.L${l}.${n}`;
      this.ln(P, B.x, Lw('ln1'), B.xn, 1);
      this.gemv(P, 'raw', B.xn, Lw('sa.qkv.w'), Lw('sa.qkv.b'), B.qkv, 3 * H, H);
      P.add(this.p.kvput, [B.qkv, B.kc[l], B.vc[l]], [pos], [Math.ceil(H / 4 / 64), 1, 1]);
      P.add(this.p.att1, [B.qkv, B.kc[l], B.vc[l], B.att], [pos + 1, H / 4, 0, 0, 0, B.scale], [NH, 1, 1]);
      this.gemv(P, 'rmw', B.att, Lw('sa.o.w'), Lw('sa.o.b'), B.x, H, H);
      this.ln(P, B.x, Lw('ln2'), B.xn, 1);
      this.gemv(P, 'raw', B.xn, Lw('ca.q.w'), Lw('ca.q.b'), B.cq, H, H);
      P.add(this.p.att1, [B.cq, B.ckv[l], B.ckv[l], B.att], [F, H / 2, 0, H / 4, 0, B.scale], [NH, 1, 1]);
      this.gemv(P, 'rmw', B.att, Lw('ca.o.w'), Lw('ca.o.b'), B.x, H, H);
      this.ln(P, B.x, Lw('ln3'), B.xn, 1);
      this.gemv(P, 'gelu', B.xn, Lw('ff1.w'), Lw('ff1.b'), B.ff, c.ffn, H);
      this.gemv(P, 'rmw', B.ff, Lw('ff2.w'), Lw('ff2.b'), B.x, H, c.ffn);
    }
    if (wantLogits) {
      this.ln(P, B.x, 'd.ln', B.xn, 1);
      this.gemv(P, 'raw', B.xn, 'd.out.w', null, B.logits, c.vocab_padded, H);
    }
    await this.submit(P);
  }

  // token embedding + position, in f32 like torch (inputs_embeds + positions)
  async embed(id, pos) {
    const H = this.cfg.d_model, t = this.man.tensors['d.tok'];
    const out = new Float32Array(H), pr = pos * H;
    if (t.dtype === 'f32') {
      const row = new Float32Array(await this.source.read(t.offset + id * H * 4, H * 4));
      for (let i = 0; i < H; i++) out[i] = Math.fround(row[i] + this.dpos[pr + i]);
    } else if (t.dtype === 'f16') {
      const row = new Uint16Array(await this.source.read(t.offset + id * H * 2, H * 2));
      for (let i = 0; i < H; i++) out[i] = Math.fround(half(row[i]) + this.dpos[pr + i]);
    } else if (t.dtype === 'i8') {
      const row = new Int8Array(await this.source.read(t.offset + id * H, H)), s = this.tokScale[id];
      for (let i = 0; i < H; i++) out[i] = Math.fround(Math.fround(row[i] * s) + this.dpos[pr + i]);
    } else throw new Error(`whisper: d.tok as ${t.dtype} is not supported`);
    return out;
  }

  // the manifest's decode recipe: special and timestamp ids dropped, bytes to UTF-8, clean-up, trim
  text(ids) {
    const dc = this.cfg.decode, voc = this.man.vocab, bytes = [];
    for (const id of ids) {
      if (id >= dc.skip_ids_from) continue;
      for (const ch of voc[id]) bytes.push(this.bytes.get(ch));
    }
    let s = new TextDecoder('utf-8').decode(new Uint8Array(bytes));
    if (dc.clean_up_tokenization_spaces) for (const [a, b] of dc.clean_up) s = s.split(a).join(b);
    return dc.strip ? s.trim() : s;
  }

  // the decoder's tensors on the GPU (all of them while it decodes)
  decoderNames() {
    return Object.keys(this.man.tensors).filter((n) => n.startsWith('d.') && !GPU_SKIP.test(n));
  }

  // bindings of a GEMM / GEMV: X, W, B (zeros if none), OUT, and the scales of i8 weights
  bind(X, wname, bname, OUT) {
    const b = [X, this.w[wname], bname ? this.w[bname] : this.zeros, OUT];
    if (this.man.tensors[wname].dtype === 'i8') b.push(this.w[`${wname}.s`]);
    return b;
  }

  // one row: OUT = epi(X · W + B) (GEMV)
  gemv(P, epi, X, wname, bname, OUT, N, K) {
    P.add(this.p[`gemv.${epi}.${this.man.tensors[wname].dtype}`], this.bind(X, wname, bname, OUT), [N, K], [N / 64, 1, 1]);
  }

  ln(P, X, prefix, Y, rows) {
    P.add(this.p.ln, [X, this.w[`${prefix}.g`], this.w[`${prefix}.b`], Y], [rows], [rows, 1, 1]);
  }

  // GEMM / convolution (convWGSL): defaults one tap, stride 1, rows from 0, row strides = channels
  conv(P, epi, X, wname, bname, OUT, o) {
    const { M, N, cin, taps = 1, stride = 1, dil = 1, pad = 0, xlen = o.M * stride } = o;
    const wT = this.man.tensors[wname].dtype, { MT, NT } = this.tile;
    P.add(this.p[`conv.${epi}.${wT}`], this.bind(X, wname, bname, OUT),
      [M, N, taps * cin, cin, cin / 4, stride, dil, pad, 0, 0, xlen, 0, N / 4, 0, 0, 0], [N / NT, Math.ceil(M / MT), 1]);
  }

  prog() {
    const steps = [];
    return { steps, add: (pipe, buffers, u, grid) => steps.push({ pipe, buffers, u, grid }) };
  }

  async submit(P) {
    const d = this.device;
    const ub = d.createBuffer({ size: Math.max(1, P.steps.length) * UNI, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const all = new Int32Array(P.steps.length * UNI / 4);
    P.steps.forEach((s, k) => all.set(s.u.map((v) => v | 0), k * UNI / 4));
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

  destroy() { this.drop(Object.keys(this.w)); this.zeros.destroy(); }
}

// first index of the maximum (over `ids` only, if given)
function argmax(a, ids) {
  let best = -Infinity, bi = -1;
  if (ids) { for (const i of ids) if (a[i] > best) { best = a[i]; bi = i; } return bi; }
  for (let i = 0; i < a.length; i++) if (a[i] > best) { best = a[i]; bi = i; }
  return bi;
}
