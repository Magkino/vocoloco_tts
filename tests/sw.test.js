import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';

// sw.js keeps the app's files for offline use: every shipped file must be on its list, or an
// offline start misses it
const root = new URL('../', import.meta.url);
const listed = JSON.parse(`[${readFileSync(new URL('sw.js', root), 'utf8').match(/const APP_FILES = \[([^\]]*)\]/)[1].replace(/'/g, '"').replace(/,\s*$/, '')}]`);

test('sw.js lists every file the app ships', () => {
  const dir = (d, re) => readdirSync(new URL(d, root)).filter((f) => re.test(f)).map((f) => d + f);
  const shipped = [
    ...dir('', /\.(js|css|html|json)$/).filter((f) => !['tailwind-input.css', 'sw.js', 'package.json', 'package-lock.json'].includes(f)),
    ...dir('workers/', /\.js$/), ...dir('workers/engine/', /\.js$/), ...dir('icons/', /\.png$/), 'lib/lamejs.min.js',
  ];
  for (const f of shipped) assert.ok(listed.includes(f), `${f} is missing from APP_FILES in sw.js`);
});

test('every file sw.js lists exists', () => {
  for (const f of listed.filter((f) => f !== './')) assert.ok(existsSync(new URL(f, root)), `${f} is listed in sw.js but doesn't exist`);
});
