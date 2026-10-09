/**
 * GPU post-processing for the TTS diffusion loop: log-softmax + CFG fusion +
 * argmax as a WebGPU compute shader. Runs on ONNX Runtime's own GPUDevice and
 * reads the logits straight from ORT's output buffers ('gpu-buffer' output
 * location), so per step only the predictions and scores (a few KB) come back
 * to the CPU instead of the full logits tensor (tens of MB).
 *
 * One workgroup per (codebook, target position); its threads split the
 * vocabulary and combine maxima / sums / argmax through shared memory.
 *
 * Logits layout: cond at (c * condStride + targetOff + t) * V, uncond at
 * uncondOff + (c * uncondStride + t) * V. That covers both the batched pass
 * (one (2, C, L, V) buffer, uncond = batch 1) and separate cond / uncond runs.
 */

const WG = 256;

const WGSL = /* wgsl */ `
struct Params {
  C:             u32,  // num codebooks (8)
  T:             u32,  // target tokens
  V:             u32,  // vocab size (1025)
  maskId:        u32,
  condStride:    u32,  // positions per codebook in the cond logits
  targetOff:     u32,  // offset of the target region in the cond sequence
  uncondStride:  u32,  // positions per codebook in the uncond logits
  uncondOff:     u32,  // element offset of the uncond logits in their buffer
  guidanceScale: f32,
  layerPenalty:  f32,
};

const WG: u32 = ${WG}u;
const PER: u32 = 5u;  // ceil(V / WG) for V = 1025
const NEG: f32 = -3.402823e+38;

@group(0) @binding(0) var<uniform> p : Params;
@group(0) @binding(1) var<storage, read>       condLogits   : array<f32>;
@group(0) @binding(2) var<storage, read>       uncondLogits : array<f32>;
@group(0) @binding(3) var<storage, read_write> pred         : array<i32>;
@group(0) @binding(4) var<storage, read_write> scores       : array<f32>;

var<workgroup> redA : array<f32, ${WG}>;
var<workgroup> redB : array<f32, ${WG}>;
var<workgroup> redI : array<u32, ${WG}>;

fn reduceMax2(lid: u32) {
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      redA[lid] = max(redA[lid], redA[lid + s]);
      redB[lid] = max(redB[lid], redB[lid + s]);
    }
    workgroupBarrier();
  }
}

fn reduceSum2(lid: u32) {
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      redA[lid] = redA[lid] + redA[lid + s];
      redB[lid] = redB[lid] + redB[lid + s];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid : vec3u, @builtin(local_invocation_id) lidv : vec3u) {
  let idx = wid.x;               // = c * T + t
  let lid = lidv.x;
  let c = idx / p.T;
  let t = idx % p.T;
  let V = p.V;
  let cBase = (c * p.condStride + p.targetOff + t) * V;
  let uBase = p.uncondOff + (c * p.uncondStride + t) * V;

  // this thread's share of the vocabulary: v = lid + k * WG
  var cv : array<f32, PER>;
  var uv : array<f32, PER>;
  var cMax = NEG;
  var uMax = NEG;
  for (var k = 0u; k < PER; k++) {
    let v = lid + k * WG;
    if (v < V) {
      cv[k] = condLogits[cBase + v];
      uv[k] = uncondLogits[uBase + v];
      cMax = max(cMax, cv[k]);
      uMax = max(uMax, uv[k]);
    }
  }
  redA[lid] = cMax; redB[lid] = uMax;
  workgroupBarrier();
  reduceMax2(lid);
  cMax = redA[0]; uMax = redB[0];
  workgroupBarrier();

  var cSum = 0.0;
  var uSum = 0.0;
  for (var k = 0u; k < PER; k++) {
    let v = lid + k * WG;
    if (v < V) {
      cSum += exp(cv[k] - cMax);
      uSum += exp(uv[k] - uMax);
    }
  }
  redA[lid] = cSum; redB[lid] = uSum;
  workgroupBarrier();
  reduceSum2(lid);
  let cLse = cMax + log(redA[0]);
  let uLse = uMax + log(redB[0]);
  workgroupBarrier();

  // CFG fusion; the guided distribution's max (all tokens) and argmax
  // (mask token excluded, lowest index wins ties)
  let g1 = 1.0 + p.guidanceScale;
  var gMax = NEG;
  var bestS = NEG;
  var bestV = 0xffffffffu;
  for (var k = 0u; k < PER; k++) {
    let v = lid + k * WG;
    if (v < V) {
      let g = g1 * (cv[k] - cLse) - p.guidanceScale * (uv[k] - uLse);
      cv[k] = g;
      gMax = max(gMax, g);
      if (v != p.maskId && g > bestS) { bestS = g; bestV = v; }
    }
  }
  redA[lid] = gMax; redB[lid] = bestS; redI[lid] = bestV;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      redA[lid] = max(redA[lid], redA[lid + s]);
      let os = redB[lid + s];
      let oi = redI[lid + s];
      if (os > redB[lid] || (os == redB[lid] && oi < redI[lid])) { redB[lid] = os; redI[lid] = oi; }
    }
    workgroupBarrier();
  }
  gMax = redA[0];
  bestS = redB[0];
  bestV = redI[0];
  workgroupBarrier();

  var gSum = 0.0;
  for (var k = 0u; k < PER; k++) {
    let v = lid + k * WG;
    if (v < V) { gSum += exp(cv[k] - gMax); }
  }
  redA[lid] = gSum;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) { redA[lid] = redA[lid] + redA[lid + s]; }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let gLse = gMax + log(redA[0]);
    pred[idx] = i32(bestV);
    scores[idx] = (bestS - gLse) - p.layerPenalty * f32(c);
  }
}
`;

export class GpuPostProcessor {
  /** @param {GPUDevice} device — ONNX Runtime's device (not owned: never destroyed here) */
  constructor(device) {
    this.device = device;
    const module = device.createShaderModule({ code: WGSL });
    const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    this.layout = device.createBindGroupLayout({
      entries: [entry(0, 'uniform'), entry(1, 'read-only-storage'), entry(2, 'read-only-storage'), entry(3, 'storage'), entry(4, 'storage')],
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module, entryPoint: 'main' },
    });
    this.paramsBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._nPos = 0;
  }

  _ensure(nPos) {
    if (nPos <= this._nPos) return;
    for (const k of ['predBuf', 'scoresBuf', 'readBuf']) if (this[k]) this[k].destroy();
    const dev = this.device, bytes = nPos * 4;
    this.predBuf = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.scoresBuf = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.readBuf = dev.createBuffer({ size: 2 * bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this._nPos = nPos;
  }

  /**
   * @param {GPUBuffer} condBuf   cond logits, f32
   * @param {GPUBuffer} uncondBuf uncond logits, f32 (may be the same buffer as condBuf)
   * Sizes: condElems / uncondElems = number of f32 values bound from each buffer.
   */
  async run(condBuf, uncondBuf, {
    C, T, V, maskId, condStride, targetOff, uncondStride, uncondOff = 0, guidanceScale, layerPenalty,
    condElems, uncondElems,
  }, predOut, scoresOut) {
    if (V > 5 * WG) throw new Error(`vocab ${V} too large for the post-processing shader`);
    const dev = this.device;
    const nPos = C * T;
    this._ensure(nPos);

    const params = new ArrayBuffer(48);
    const u32 = new Uint32Array(params), f32 = new Float32Array(params);
    u32[0] = C; u32[1] = T; u32[2] = V; u32[3] = maskId;
    u32[4] = condStride; u32[5] = targetOff; u32[6] = uncondStride; u32[7] = uncondOff;
    f32[8] = guidanceScale; f32[9] = layerPenalty;
    dev.queue.writeBuffer(this.paramsBuf, 0, params);

    dev.pushErrorScope('validation');
    const bindGroup = dev.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: condBuf, size: condElems * 4 } },
        { binding: 2, resource: { buffer: uncondBuf, size: uncondElems * 4 } },
        { binding: 3, resource: { buffer: this.predBuf, size: nPos * 4 } },
        { binding: 4, resource: { buffer: this.scoresBuf, size: nPos * 4 } },
      ],
    });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(nPos);
    pass.end();
    enc.copyBufferToBuffer(this.predBuf, 0, this.readBuf, 0, nPos * 4);
    enc.copyBufferToBuffer(this.scoresBuf, 0, this.readBuf, nPos * 4, nPos * 4);
    dev.queue.submit([enc.finish()]);
    const errScope = dev.popErrorScope();

    await this.readBuf.mapAsync(GPUMapMode.READ, 0, 2 * nPos * 4);
    // An invalid dispatch would otherwise read back zeros without any error
    const err = await errScope;
    if (err) { this.readBuf.unmap(); throw new Error(err.message); }
    const mapped = this.readBuf.getMappedRange(0, 2 * nPos * 4);
    predOut.set(new Int32Array(mapped, 0, nPos));
    scoresOut.set(new Float32Array(mapped, nPos * 4, nPos));
    this.readBuf.unmap();
  }

  destroy() {
    for (const k of ['paramsBuf', 'predBuf', 'scoresBuf', 'readBuf']) {
      if (this[k]) { this[k].destroy(); this[k] = null; }
    }
  }
}
