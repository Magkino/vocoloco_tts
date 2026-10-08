import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getTimeSteps, unmaskSchedule } from '../workers/unmask-schedule.js';

test('time steps span [0, 1] with numStep intervals', () => {
  const ts = getTimeSteps(0, 1, 20, 0.1);
  assert.equal(ts.length, 21);
  assert.equal(ts[0], 0);
  assert.ok(Math.abs(ts[20] - 1) < 1e-12);
  for (let i = 1; i < ts.length; i++) assert.ok(ts[i] > ts[i - 1]);
});

test('schedule unmasks every token exactly once', () => {
  for (const numStep of [8, 16, 20, 32]) {
    for (const total of [8, 1600, 3600]) {
      const sched = unmaskSchedule(total, numStep, 0.1);
      assert.equal(sched.length, numStep);
      assert.equal(sched.reduce((a, b) => a + b, 0), total);
      assert.ok(sched.every((n) => n >= 0));
    }
  }
});

test('matches upstream OmniVoice: last of 20 steps commits ~34.5%, not 68%', () => {
  const total = 3600;
  const sched = unmaskSchedule(total, 20, 0.1);
  const lastShare = sched[19] / total;
  assert.ok(lastShare > 0.33 && lastShare < 0.36, `last share ${lastShare}`);
});
