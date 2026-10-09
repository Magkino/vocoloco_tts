/**
 * Forward-pass parity harness for ONNX Runtime Web upgrades / execution
 * changes. Runs the main model once on a fixed, app-shaped input and stores a
 * summary (per-position argmax + a strided logits sample) in IndexedDB, so two
 * runtimes can be compared without holding both in memory.
 *
 * Use from the browser console on the dev server (same origin as /models):
 *   const p = await import('/tests/browser/parity-forward.js');
 *   await p.run('1.20.1-jsep', 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/', 'ort.all.mjs');
 *   // reload the page, then
 *   await p.run('1.30.0-webgpu', 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/', 'ort.webgpu.min.mjs');
 *   await p.compare('1.20.1-jsep', '1.30.0-webgpu');
 */

const C = 8, V = 1025, MASK = 1024;

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('vocoloco-parity', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('runs');
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function put(key, value) {
  const db = await idb();
  await new Promise((res, rej) => { const tx = db.transaction('runs', 'readwrite'); tx.objectStore('runs').put(value, key); tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
  db.close();
}

async function get(key) {
  const db = await idb();
  const v = await new Promise((res) => { const q = db.transaction('runs').objectStore('runs').get(key); q.onsuccess = () => res(q.result); });
  db.close();
  return v;
}

// App-shaped batch: cond = [text 40 | ref audio 120 | target 140], uncond = target padded to L
export function makeInputs({ textLen = 40, refLen = 120, T = 140 } = {}) {
  const L = textLen + refLen + T;
  const ids = new BigInt64Array(2 * C * L).fill(BigInt(MASK));
  const amask = new Uint8Array(2 * L);
  const attn = new Uint8Array(2 * L * L);
  for (let c = 0; c < C; c++) {
    for (let i = 0; i < textLen; i++) ids[c * L + i] = BigInt(1000 + (i * 37) % 50000);
    for (let i = textLen; i < L; i++) {
      const isTarget = i >= textLen + refLen;
      const code = (i * 7 + c * 13) % 1024;
      ids[c * L + i] = BigInt(isTarget && i % 3 ? MASK : code);
    }
    for (let t = 0; t < T; t++) ids[(C + c) * L + t] = ids[c * L + textLen + refLen + t];
  }
  for (let i = textLen; i < L; i++) amask[i] = 1;
  for (let t = 0; t < T; t++) amask[L + t] = 1;
  for (let q = 0; q < L; q++) for (let k = 0; k < L; k++) attn[q * L + k] = 1;
  for (let q = 0; q < T; q++) for (let k = 0; k < T; k++) attn[L * L + q * L + k] = 1;
  for (let p = T; p < L; p++) attn[L * L + p * L + p] = 1;
  return { ids, amask, attn, L, T, textLen, refLen };
}

export async function loadSession(ortBase, ortFile, ep = 'webgpu') {
  const ort = await import(ortBase + ortFile);
  ort.env.wasm.wasmPaths = ortBase;
  const files = await (await fetch('/models/omnivoice-main-manifest.json')).json();
  const externalData = await Promise.all(files.map(async (f) => ({ path: f, data: await (await fetch('/models/' + f)).arrayBuffer() })));
  const t0 = performance.now();
  const session = await ort.InferenceSession.create('/models/omnivoice-main-split.onnx', { executionProviders: [ep], externalData, graphOptimizationLevel: 'all' });
  return { ort, session, createMs: Math.round(performance.now() - t0) };
}

export async function run(label, ortBase, ortFile, { ep = 'webgpu', reps = 3 } = {}) {
  const { ort, session, createMs } = await loadSession(ortBase, ortFile, ep);
  const inp = makeInputs();
  const feeds = () => ({
    input_ids: new ort.Tensor('int64', inp.ids, [2, C, inp.L]),
    audio_mask: new ort.Tensor('bool', inp.amask, [2, inp.L]),
    attention_mask: new ort.Tensor('bool', inp.attn, [2, 1, inp.L, inp.L]),
  });
  let out = await session.run(feeds()); // warm-up (shader compile)
  const times = [];
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    out = await session.run(feeds());
    times.push(Math.round(performance.now() - t0));
  }
  const logits = out.audio_logits.data;
  const nPos = 2 * C * inp.L;
  const argmax = new Int32Array(nPos);
  for (let p = 0; p < nPos; p++) {
    let best = 0, bv = -Infinity;
    for (let v = 0; v < V; v++) { const x = logits[p * V + v]; if (x > bv) { bv = x; best = v; } }
    argmax[p] = best;
  }
  const sample = new Float32Array(Math.floor(logits.length / 97));
  for (let i = 0; i < sample.length; i++) sample[i] = logits[i * 97];
  const summary = { label, ortFile, ep, createMs, times, L: inp.L, T: inp.T, textLen: inp.textLen, refLen: inp.refLen, argmax, sample };
  await put(label, summary);
  await session.release();
  return { label, createMs, times };
}

export async function compare(a, b) {
  const A = await get(a), B = await get(b);
  const L = A.L, T = A.T, tgt0 = A.textLen + A.refLen;
  let agreeAll = 0, agreeTarget = 0, nTarget = 0;
  for (let p = 0; p < A.argmax.length; p++) {
    const same = A.argmax[p] === B.argmax[p];
    if (same) agreeAll++;
    const b2 = Math.floor(p / (C * L)), pos = p % L;
    const isTarget = b2 === 0 ? pos >= tgt0 : pos < T;
    if (isTarget) { nTarget++; if (same) agreeTarget++; }
  }
  let maxAbs = 0, sumAbs = 0, sumMag = 0;
  for (let i = 0; i < A.sample.length; i++) {
    const d = Math.abs(A.sample[i] - B.sample[i]);
    maxAbs = Math.max(maxAbs, d); sumAbs += d; sumMag += Math.abs(A.sample[i]);
  }
  return {
    [a]: { times: A.times }, [b]: { times: B.times },
    argmaxAgreeTargetPct: +(100 * agreeTarget / nTarget).toFixed(3),
    argmaxAgreeAllPct: +(100 * agreeAll / A.argmax.length).toFixed(3),
    maxAbsDiff: +maxAbs.toExponential(2),
    meanAbsDiff: +(sumAbs / A.sample.length).toExponential(2),
    meanAbsLogit: +(sumMag / A.sample.length).toFixed(3),
  };
}
