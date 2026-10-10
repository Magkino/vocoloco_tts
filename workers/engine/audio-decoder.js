// VocoLoco engine: the audio decoder (HiggsAudioV2 tokenizer: RVQ codebooks + DAC decoder) on
// WebGPU. Weights: a JSON manifest and one binary file.
//
// f32 throughout. Activations are time-major [T][C], so every convolution is a GEMM over
// (tap, input channel) with vec4 loads along channels and zero padding at the edges. Transposed
// convolutions run per output phase as 2-tap convolutions on the input grid (no zero insertion).
// Each Snake activation runs once, in the epilogue of the convolution that produces its input;
// sin() is range-reduced and evaluated with Cephes polynomials, because GPU sin() is only
// accurate on [-π, π]. Long outputs are decoded in windows with CTX frames of context on either
// side, which covers the decoder's receptive field (the same scheme as the worker's ORT decoder).

import { compile } from './omnivoice-engine.js';

const UNI = 256;
const CTX = 12;
const WINDOW = 128; // frames decoded per window (plus context); bounds memory on phones
const UPLOAD_BATCH = 64 << 20; // weight bytes written before waiting (as OmniVoiceEngine.loadWeights)

export const SNAKE = `fn psin(x: f32) -> f32 {
  let k = round(x * 0.636619772367581343);
  var r = x - k * 1.5703125;
  r = r - k * 4.837512969970703125e-4;
  r = r - k * 7.54978995489188216e-8;
  let r2 = r * r;
  let s = r + r * r2 * (-1.6666654611e-1 + r2 * (8.3321608736e-3 + r2 * -1.9515295891e-4));
  let c = 1.0 - 0.5 * r2 + r2 * r2 * (4.166664568298827e-2 + r2 * (-1.388731625493765e-3 + r2 * 2.443315711809948e-5));
  let q = i32(k) & 3;
  return select(select(-c, -s, q == 2), select(c, s, q == 0), q < 2);
}
fn snake(y: vec4<f32>, a: vec4<f32>) -> vec4<f32> {
  let s = vec4<f32>(psin(a.x * y.x), psin(a.y * y.y), psin(a.z * y.z), psin(a.w * y.w));
  return y + (vec4<f32>(1.0) / (a + vec4<f32>(1e-9))) * (s * s);
}`;

// out[orow][n] = B[n] + Σ_(j, ci) W[z][j·cin + ci][n] · X[m + j·dil − pad][ci],  orow = m·omul + oadd + z
// Epilogue: rmw (y += OUT, residual stream updated in place), raw (OUT = y), snake (SN = snake(y, AL)).
function convWGSL({ TM, TN, RM, VN, KT }, { rmw = false, raw = false, snake = false }) {
  const MT = TM * RM, NT = TN * VN * 4, THREADS = TM * TN, KT4 = KT / 4, NT4 = NT / 4;
  let b = 4;
  const decls = [];
  if (raw || rmw) decls.push(`@group(0) @binding(${b++}) var<storage, read_write> OUT: array<vec4<f32>>;`);
  if (snake) decls.push(`@group(0) @binding(${b++}) var<storage, read_write> SN: array<vec4<f32>>;`, `@group(0) @binding(${b++}) var<storage, read> AL: array<vec4<f32>>;`);
  let inner = '';
  for (let i = 0; i < RM; i++) inner += `\n      let a${i} = As[(ly + ${i * TM}u) * ${KT4}u + k4];`;
  for (let kk = 0; kk < 4; kk++) {
    const ch = 'xyzw'[kk];
    for (let v = 0; v < VN; v++) inner += `\n      let b${kk}_${v} = Ws[(k4 * 4u + ${kk}u) * ${NT4}u + ${v * TN}u + lx];`;
    for (let i = 0; i < RM; i++) for (let v = 0; v < VN; v++) inner += `\n      c${i}_${v} = fma(vec4<f32>(a${i}.${ch}), b${kk}_${v}, c${i}_${v});`;
  }
  let decl = '', store = '';
  for (let i = 0; i < RM; i++) for (let v = 0; v < VN; v++) {
    decl += `\n  var c${i}_${v} = vec4<f32>(0.0);`;
    store += `
  {
    let m = m0 + ly + ${i * TM}u; let orow = i32(m * u.omul) + u.oadd + i32(wg.z);
    if (m < u.M && orow >= 0 && orow < i32(u.tout)) {
      let col = n0 / 4u + ${v * TN}u + lx; let o = u32(orow) * N4 + col;
      var y = c${i}_${v} + B[col];${rmw ? '\n      y += OUT[o];' : ''}${raw || rmw ? '\n      OUT[o] = y;' : ''}${snake ? '\n      SN[o] = snake(y, AL[col]);' : ''}
    }
  }`;
  }
  return `struct U { M: u32, N: u32, K: u32, cin: u32, tin: u32, dil: u32, pad: u32, omul: u32, oadd: i32, tout: u32, wz: u32, p0: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> B: array<vec4<f32>>;
${decls.join('\n')}
${SNAKE}
var<workgroup> As: array<vec4<f32>, ${MT * KT4}>;
var<workgroup> Ws: array<vec4<f32>, ${KT * NT4}>;
@compute @workgroup_size(${TN}, ${TM})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * ${MT}u; let n0 = wg.x * ${NT}u; let lx = lid.x; let ly = lid.y;
  let wB = wg.z * u.wz; let cin4 = u.cin / 4u; let N4 = u.N / 4u;${decl}
  for (var k0 = 0u; k0 < u.K; k0 += ${KT}u) {
    let shift = i32((k0 / u.cin) * u.dil) - i32(u.pad); let ci4 = (k0 % u.cin) / 4u;
    for (var e = li; e < ${MT * KT4}u; e += ${THREADS}u) {
      let r = e / ${KT4}u; let c = e % ${KT4}u;
      let src = i32(m0 + r) + shift;
      var v = vec4<f32>(0.0);
      if (src >= 0 && src < i32(u.tin)) { v = X[u32(src) * cin4 + ci4 + c]; }
      As[e] = v;
    }
    for (var e = li; e < ${KT * NT4}u; e += ${THREADS}u) {
      let r = e / ${NT4}u; let c = e % ${NT4}u;
      Ws[e] = W[wB + (k0 + r) * N4 + n0 / 4u + c];
    }
    workgroupBarrier();
    for (var k4 = 0u; k4 < ${KT4}u; k4++) {${inner}
    }
    workgroupBarrier();
  }${store}
}`;
}

// x0[t] = bias + Σ_c tables[c][codes[c][t]]   (H = 256: 64 threads × vec4)
const EMBED = (C, V, H) => `struct U { T: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> codes: array<i32>;
@group(0) @binding(2) var<storage, read> tables: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> x: array<vec4<f32>>;
@compute @workgroup_size(${H / 4})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  var v = vec4<f32>(0.0);
  for (var c = 0u; c < ${C}u; c++) { v += tables[(c * ${V}u + u32(codes[c * u.T + wg.x])) * ${H / 4}u + t]; }
  x[wg.x * ${H / 4}u + t] = v + bias[t];
}`;

// pcm[t] = b + Σ_(j, ci) w[j·C + ci] · S[t + j − 3][ci]   (kernel 7, C channels -> 1)
const OUTCONV = (C) => `struct U { T: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> pcm: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.y * 65535u * 256u + gid.x;
  if (t >= u.T) { return; }
  var acc = vec4<f32>(0.0);
  for (var j = 0u; j < 7u; j++) {
    let src = i32(t) + i32(j) - 3;
    if (src >= 0 && src < i32(u.T)) {
      for (var c = 0u; c < ${C / 4}u; c++) { acc = fma(S[u32(src) * ${C / 4}u + c], w[j * ${C / 4}u + c], acc); }
    }
  }
  pcm[t] = acc.x + acc.y + acc.z + acc.w + b[0];
}`;

const CFG_WIDE = { TM: 16, TN: 16, RM: 4, VN: 1, KT: 16 };  // N ≥ 64
const CFG_NARROW = { TM: 16, TN: 8, RM: 4, VN: 1, KT: 16 }; // N = 32
const tile = (N) => (N % 64 === 0 ? CFG_WIDE : CFG_NARROW);

export class AudioDecoder {
  // manifest: parsed decoder-<dtype>.json; source: { read(offset, bytes) -> ArrayBuffer } over its .bin
  static async load(device, manifest, source) {
    if (manifest.version !== 1) throw new Error(`decoder: weights in format ${manifest.version}, this code reads format 1`);
    const d = new AudioDecoder(device, manifest.config);
    let pending = 0;
    for (const [name, t] of Object.entries(manifest.tensors)) {
      const buf = device.createBuffer({ size: Math.ceil(t.bytes / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `dec ${name}` });
      device.queue.writeBuffer(buf, 0, await source.read(t.offset, t.bytes));
      d.w[name] = buf;
      if ((pending += t.bytes) >= UPLOAD_BATCH) { await device.queue.onSubmittedWorkDone(); pending = 0; }
    }
    const { codebooks, codebook_size: V, hidden } = d.cfg;
    const C = d.cfg.blocks.at(-1).cout;
    const variants = { snake: { snake: true }, rawSnake: { raw: true, snake: true }, rmwSnake: { rmw: true, snake: true } };
    const jobs = [['embed', EMBED(codebooks, V, hidden)], ['out', OUTCONV(C)]];
    for (const [k, epi] of Object.entries(variants)) for (const [tk, cfg] of [['wide', CFG_WIDE], ['narrow', CFG_NARROW]]) jobs.push([`${k}.${tk}`, convWGSL(cfg, epi)]);
    const pipes = await Promise.all(jobs.map(([k, code]) => compile(device, code, `dec ${k}`)));
    jobs.forEach(([k], i) => { d.p[k] = pipes[i]; });
    return d;
  }

  constructor(device, cfg) {
    this.device = device;
    this.cfg = cfg;
    this.w = {};
    this.p = {};
    this.plans = new Map();
  }

  // codes: Int32Array [8][T] -> Float32Array of T · hop samples
  async decode(codes, T) {
    const { hop } = this.cfg;
    if (T <= WINDOW + 2 * CTX) return this.run(codes, T);
    const out = new Float32Array(T * hop);
    for (let a = 0; a < T; a += WINDOW) {
      const b = Math.min(T, a + WINDOW), lo = Math.max(0, a - CTX), hi = Math.min(T, b + CTX), n = hi - lo;
      const win = new Int32Array(8 * n);
      for (let c = 0; c < 8; c++) win.set(codes.subarray(c * T + lo, c * T + hi), c * n);
      const pcm = await this.run(win, n);
      out.set(pcm.subarray((a - lo) * hop, (b - lo) * hop), a * hop);
    }
    return out;
  }

  async run(codes, n) {
    let plan = this.plans.get(n);
    if (!plan) {
      if (this.plans.size >= 3) { for (const p of this.plans.values()) p.destroy(); this.plans.clear(); }
      plan = this.plan(n);
      this.plans.set(n, plan);
    }
    const d = this.device;
    d.queue.writeBuffer(plan.codes, 0, codes);
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const s of plan.steps) { pass.setPipeline(s.pipe); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.grid); }
    pass.end();
    enc.copyBufferToBuffer(plan.pcm, 0, plan.rb, 0, plan.pcm.size);
    d.queue.submit([enc.finish()]);
    await plan.rb.mapAsync(GPUMapMode.READ);
    const pcm = new Float32Array(plan.rb.getMappedRange().slice(0));
    plan.rb.unmap();
    return pcm;
  }

  plan(n) {
    const d = this.device, w = this.w, { hidden, channels, blocks, hop } = this.cfg;
    let per = 0, t = n; // largest activation (floats) of any stage
    per = Math.max(per, n * channels);
    for (const b of blocks) { t *= b.stride; per = Math.max(per, t * b.cout); }
    const mk = (bytes, label, usage = GPUBufferUsage.STORAGE) => d.createBuffer({ size: Math.max(16, bytes), usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label });
    const buf = {
      codes: mk(8 * n * 4, 'dec codes'), x0: mk(n * hidden * 4, 'dec x0'),
      X: mk(per * 4, 'dec X'), H: mk(per * 4, 'dec H'), Sa: mk(per * 4, 'dec Sa'), Sb: mk(per * 4, 'dec Sb'),
      pcm: mk(n * hop * 4, 'dec pcm'),
    };
    buf.rb = d.createBuffer({ size: n * hop * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const slots = [], steps = [];
    const res = (buffer) => ({ buffer });
    const uni = (vals) => { slots.push(new Uint32Array(new Int32Array(vals).buffer)); return slots.length - 1; };
    // conv: x [tin][cin] -> [tout][N]; phases > 1: transposed conv (taps 2, rows q = 0..tin)
    const conv = (kind, A, wname, bname, outs, { tin, cin, N, taps, dil = 1, pad, phases = 1, omul = 1, oadd = 0, tout }) => {
      const cfg = tile(N), MT = cfg.TM * cfg.RM, NT = cfg.TN * cfg.VN * 4;
      const M = phases > 1 ? tin + 1 : tin;
      steps.push({ pipe: this.p[`${kind}.${N % 64 === 0 ? 'wide' : 'narrow'}`], entries: [res(A), res(w[wname]), res(w[bname]), ...outs.map(res)],
        u: uni([M, N, taps * cin, cin, tin, dil, pad, omul, oadd, tout, phases > 1 ? taps * cin * N / 4 : 0, 0]), grid: [N / NT, Math.ceil(M / MT), phases] });
    };
    steps.push({ pipe: this.p.embed, entries: [res(buf.codes), res(w.tables), res(w.in_bias), res(buf.x0)], u: uni([n, 0, 0, 0]), grid: [n, 1, 1] });
    conv('snake', buf.x0, 'conv1.w', 'conv1.b', [buf.Sa, w['b0.a']], { tin: n, cin: hidden, N: channels, taps: 7, pad: 3, tout: n });
    let S = buf.Sa, S2 = buf.Sb;
    t = n;
    blocks.forEach((b, i) => {
      const tin = t; t *= b.stride;
      conv('rawSnake', S, `b${i}.t.w`, `b${i}.t.b`, [buf.X, S2, w[`b${i}.u0.a1`]],
        { tin, cin: b.cin, N: b.cout, taps: 2, pad: 1, phases: b.stride, omul: b.stride, oadd: -b.pad, tout: t });
      [S, S2] = [S2, S];
      b.dilations.forEach((dil, u) => {
        const p = `b${i}.u${u}`;
        const nextAlpha = u < 2 ? `b${i}.u${u + 1}.a1` : i < blocks.length - 1 ? `b${i + 1}.a` : 'out.a';
        conv('snake', S, `${p}.w1`, `${p}.b1`, [buf.H, w[`${p}.a2`]], { tin: t, cin: b.cout, N: b.cout, taps: 7, dil, pad: 3 * dil, tout: t });
        conv('rmwSnake', buf.H, `${p}.w2`, `${p}.b2`, [buf.X, S, w[nextAlpha]], { tin: t, cin: b.cout, N: b.cout, taps: 1, pad: 0, tout: t });
      });
    });
    const groups = Math.ceil(t / 256);
    steps.push({ pipe: this.p.out, entries: [res(S), res(w['out.w']), res(w['out.b']), res(buf.pcm)], u: uni([t, 0, 0, 0]), grid: [Math.min(groups, 65535), Math.ceil(groups / 65535), 1] });

    buf.uniform = d.createBuffer({ size: slots.length * UNI, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const all = new Uint8Array(slots.length * UNI);
    slots.forEach((s, k) => all.set(new Uint8Array(s.buffer), k * UNI));
    d.queue.writeBuffer(buf.uniform, 0, all);
    buf.steps = steps.map((s) => ({
      pipe: s.pipe, grid: s.grid,
      bg: d.createBindGroup({ layout: s.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: buf.uniform, offset: s.u * UNI, size: 64 } },
        ...s.entries.map((r, k) => ({ binding: k + 1, resource: r })),
      ] }),
    }));
    buf.destroy = () => { for (const k of ['codes', 'x0', 'X', 'H', 'Sa', 'Sb', 'pcm', 'rb', 'uniform']) buf[k].destroy(); };
    return buf;
  }

  // Frees the working buffers kept for recent clip lengths (decode() makes them again)
  releasePlans() {
    for (const p of this.plans.values()) p.destroy();
    this.plans.clear();
  }

  destroy() {
    this.releasePlans();
    for (const b of Object.values(this.w)) b.destroy();
  }
}
