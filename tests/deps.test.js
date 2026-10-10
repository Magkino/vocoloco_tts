import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// The app loads ONNX Runtime from jsDelivr; package.json lists the same version so updates are
// proposed (Dependabot). An update there must reach the URL too
test('the ONNX Runtime URL matches package.json', () => {
  const read = (f) => readFileSync(new URL(`../${f}`, import.meta.url), 'utf8');
  const want = JSON.parse(read('package.json')).optionalDependencies['onnxruntime-web'];
  const url = read('workers/tts-worker.js').match(/onnxruntime-web@([\w.-]+)\//)?.[1];
  assert.equal(url, want, `workers/tts-worker.js loads onnxruntime-web@${url}, package.json says ${want}`);
});
