/**
 * GPU post-processing for TTS diffusion loop.
 * Moves log-softmax + CFG fusion + argmax from JS CPU to WebGPU compute shaders.
 */

const WGSL = /* wgsl */ `

struct Params {
  C:              u32,  // num codebooks (8)
  maxLen:         u32,  // padded sequence length
  V:              u32,  // vocab size (1025)
  numTargetTokens:u32,
  targetOff:      u32,  // offset of target region in conditional sequence
  maskId:         u32,
  guidanceScale:  f32,
  layerPenalty:   f32,
};

@group(0) @binding(0) var<uniform> p : Params;
@group(0) @binding(1) var<storage, read>       logits : array<f32>;
@group(0) @binding(2) var<storage, read_write> pred   : array<i32>;
@group(0) @binding(3) var<storage, read_write> scores : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let idx = gid.x;
  let nPos = p.C * p.numTargetTokens;
  if (idx >= nPos) { return; }

  let c = idx / p.numTargetTokens;
  let t = idx % p.numTargetTokens;
  let V = p.V;

  let cBase = (c * p.maxLen + p.targetOff + t) * V;
  let uBase = ((p.C + c) * p.maxLen + t) * V;

  // ── Pass 1: log-softmax for conditional logits ──
  var cMax: f32 = -3.402823e+38;
  for (var v: u32 = 0u; v < V; v++) {
    let val = logits[cBase + v];
    cMax = max(cMax, val);
  }
  var cSum: f32 = 0.0;
  for (var v: u32 = 0u; v < V; v++) {
    cSum += exp(logits[cBase + v] - cMax);
  }
  let cLse = cMax + log(cSum);

  // ── Pass 2: log-softmax for unconditional logits ──
  var uMax: f32 = -3.402823e+38;
  for (var v: u32 = 0u; v < V; v++) {
    let val = logits[uBase + v];
    uMax = max(uMax, val);
  }
  var uSum: f32 = 0.0;
  for (var v: u32 = 0u; v < V; v++) {
    uSum += exp(logits[uBase + v] - uMax);
  }
  let uLse = uMax + log(uSum);

  // ── Pass 3: CFG fusion + find max for final log-softmax ──
  let gScale1 = 1.0 + p.guidanceScale;
  var gMax: f32 = -3.402823e+38;
  for (var v: u32 = 0u; v < V; v++) {
    let cLP = logits[cBase + v] - cLse;
    let uLP = logits[uBase + v] - uLse;
    let gv = gScale1 * cLP - p.guidanceScale * uLP;
    gMax = max(gMax, gv);
  }

  // ── Pass 4: exp-sum for final log-softmax denominator ──
  var gSum: f32 = 0.0;
  for (var v: u32 = 0u; v < V; v++) {
    let cLP = logits[cBase + v] - cLse;
    let uLP = logits[uBase + v] - uLse;
    let gv = gScale1 * cLP - p.guidanceScale * uLP;
    gSum += exp(gv - gMax);
  }
  let gLse = gMax + log(gSum);

  // ── Pass 5: argmax (skip maskId) ──
  var bestV: i32 = 0;
  var bestS: f32 = -3.402823e+38;
  for (var v: u32 = 0u; v < V; v++) {
    if (v == p.maskId) { continue; }
    let cLP = logits[cBase + v] - cLse;
    let uLP = logits[uBase + v] - uLse;
    let gv = gScale1 * cLP - p.guidanceScale * uLP;
    let lp = gv - gLse;
    if (lp > bestS) {
      bestS = lp;
      bestV = i32(v);
    }
  }

  pred[idx] = bestV;
  scores[idx] = bestS - p.layerPenalty * f32(c);
}
`;

export class GpuPostProcessor {
  constructor() {
    this.device = null;
    this.pipeline = null;
    this.bindGroupLayout = null;
    // Buffers
    this.logitsBuf = null;
    this.paramsBuf = null;
    this.predBuf = null;
    this.scoresBuf = null;
    this.predReadBuf = null;
    this.scoresReadBuf = null;
    // Current capacity
    this._logitsSize = 0;
    this._nPos = 0;
  }

  async init() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });

    const module = this.device.createShaderModule({ code: WGSL });
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });

    // Param buffer is always 32 bytes (8 x u32/f32)
    this.paramsBuf = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Allocate / grow buffers for a new synthesis run. */
  prepare(C, maxLen, V, numTargetTokens) {
    const dev = this.device;
    const logitsBytes = 2 * C * maxLen * V * 4;  // float32
    const nPos = C * numTargetTokens;
    const posBytes = nPos * 4;

    // Grow logits buffer if needed
    if (!this.logitsBuf || logitsBytes > this._logitsSize) {
      if (this.logitsBuf) this.logitsBuf.destroy();
      this.logitsBuf = dev.createBuffer({
        size: logitsBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._logitsSize = logitsBytes;
    }

    // Grow output buffers if needed
    if (!this.predBuf || nPos > this._nPos) {
      if (this.predBuf) this.predBuf.destroy();
      if (this.scoresBuf) this.scoresBuf.destroy();
      if (this.predReadBuf) this.predReadBuf.destroy();
      if (this.scoresReadBuf) this.scoresReadBuf.destroy();

      this.predBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.scoresBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.predReadBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.scoresReadBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this._nPos = nPos;
    }
  }

  /**
   * Run the post-processing shader for one diffusion step.
   * @param {Float32Array} logits - raw logits from ONNX (2, C, maxLen, V)
   * @param {object} params - { C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty }
   * @param {Int32Array} predOut - output array for predicted tokens
   * @param {Float32Array} scoresOut - output array for scores
   */
  async run(logits, params, predOut, scoresOut) {
    const dev = this.device;
    const { C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty } = params;
    const nPos = C * numTargetTokens;

    // Upload params (8 × 4 bytes = 32)
    const paramData = new ArrayBuffer(32);
    const u32 = new Uint32Array(paramData);
    const f32 = new Float32Array(paramData);
    u32[0] = C;
    u32[1] = maxLen;
    u32[2] = V;
    u32[3] = numTargetTokens;
    u32[4] = targetOff;
    u32[5] = maskId;
    f32[6] = guidanceScale;
    f32[7] = layerPenalty;
    dev.queue.writeBuffer(this.paramsBuf, 0, paramData);

    // Upload logits
    const logitsBytes = 2 * C * maxLen * V * 4;
    dev.queue.writeBuffer(this.logitsBuf, 0, logits.buffer, logits.byteOffset, logitsBytes);

    // Create bind group
    const bindGroup = dev.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.logitsBuf, size: logitsBytes } },
        { binding: 2, resource: { buffer: this.predBuf, size: nPos * 4 } },
        { binding: 3, resource: { buffer: this.scoresBuf, size: nPos * 4 } },
      ],
    });

    // Dispatch
    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(nPos / 64));
    pass.end();

    // Copy results to read buffers
    encoder.copyBufferToBuffer(this.predBuf, 0, this.predReadBuf, 0, nPos * 4);
    encoder.copyBufferToBuffer(this.scoresBuf, 0, this.scoresReadBuf, 0, nPos * 4);
    dev.queue.submit([encoder.finish()]);

    // Map both read buffers in parallel
    await Promise.all([
      this.predReadBuf.mapAsync(GPUMapMode.READ),
      this.scoresReadBuf.mapAsync(GPUMapMode.READ),
    ]);

    predOut.set(new Int32Array(this.predReadBuf.getMappedRange()));
    scoresOut.set(new Float32Array(this.scoresReadBuf.getMappedRange()));

    this.predReadBuf.unmap();
    this.scoresReadBuf.unmap();
  }

  destroy() {
    for (const k of ['logitsBuf', 'paramsBuf', 'predBuf', 'scoresBuf', 'predReadBuf', 'scoresReadBuf']) {
      if (this[k]) { this[k].destroy(); this[k] = null; }
    }
    if (this.device) { this.device.destroy(); this.device = null; }
  }
}
