/**
 * Seeded A/B harness for TTS worker changes: runs the same texts through a
 * worker build and stores the generated tokens + timings in IndexedDB, so two
 * builds can be compared one after the other (only one model in memory).
 *
 * From the browser console on the dev server:
 *   const ab = await import('/tests/browser/ab-worker.js');
 *   await ab.run('old', '/ab-v130/workers/tts-worker.js');   // a checkout of the previous version
 *   await ab.run('new', '/workers/tts-worker.js');   // ?passes=batched|split forces a layout
 *   await ab.compare('old', 'new');
 */

const CASES = [
  { text: 'The quick brown fox jumps over the lazy dog near the riverbank.' },
  { text: 'Guten Morgen! Heute scheint die Sonne, und wir gehen spazieren.', lang: 'German' },
  { text: 'A designed voice speaks this sentence slowly and clearly.', instruct: 'female, low pitch' },
  // reference-conditioned (cloned voice): uses the tokens of an earlier case as the reference
  { text: 'This sentence continues in the voice of the first one.', refFrom: 0 },
  { text: 'Long passages are split into chunks of a few sentences each, so every chunk is a longer sequence than a single short line. This one checks how the runtime handles that length, and whether the result stays the same.' },
  { text: 'A short line spoken with a long reference voice.', refFrom: 4 },
];

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('vocoloco-ab', 1);
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

function call(w, msg, done) {
  return new Promise((resolve, reject) => {
    const steps = [];
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress' && m.stage === 'generating') steps.push(m.stepMs);
      else if (m.type === 'error') reject(new Error(m.message));
      else if (done(m)) resolve({ m, steps });
    };
    w.postMessage(msg);
  });
}

export async function run(label, workerUrl, { seed = 1234, numStep = 32, forceCPU = false, reps = 2 } = {}) {
  const w = new Worker(`${workerUrl}${workerUrl.includes('?') ? '&' : '?'}ab=${Date.now()}`, { type: 'module' });
  const t0 = performance.now();
  await call(w, { type: 'init', modelBaseUrl: new URL('/models', location.href).href, forceCPU }, (m) => m.type === 'ready');
  const initMs = Math.round(performance.now() - t0);
  const out = [];
  for (const [i, c] of CASES.entries()) {
    const ref = c.refFrom != null ? out[c.refFrom] : null;
    const times = [];
    let res;
    for (let r = 0; r < reps; r++) { // first rep compiles shaders for new shapes
      const s0 = performance.now();
      res = await call(w, {
        type: 'synthesize', jobId: i, text: c.text, lang: c.lang || null, instruct: c.instruct || null,
        refTokens: ref ? new Int32Array(ref.tokens) : null, refText: ref ? CASES[c.refFrom].text : null,
        seed, numStep, guidanceScale: 2.0, tShift: 0.1, returnTokens: true,
      }, (m) => m.type === 'audio');
      times.push(Math.round(performance.now() - s0));
    }
    out.push({ text: c.text, tokens: res.m.tokens, tokenCount: res.m.tokenCount, times, stepMs: res.steps, samples: res.m.pcm.length });
  }
  w.terminate();
  await put(label, { label, workerUrl, initMs, out });
  return { label, initMs, times: out.map((o) => o.times) };
}

export async function compare(a, b) {
  const A = await get(a), B = await get(b);
  const median = (xs) => { const s = [...xs].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  return A.out.map((x, i) => {
    const y = B.out[i];
    let same = 0;
    for (let k = 0; k < x.tokens.length; k++) if (x.tokens[k] === y.tokens[k]) same++;
    return {
      text: x.text.slice(0, 32),
      tokenAgreePct: +(100 * same / x.tokens.length).toFixed(2),
      [`${a}Ms`]: x.times.at(-1), [`${b}Ms`]: y.times.at(-1),
      [`${a}StepMs`]: median(x.stepMs), [`${b}StepMs`]: median(y.stepMs),
      speedup: +(x.times.at(-1) / y.times.at(-1)).toFixed(2),
    };
  });
}
