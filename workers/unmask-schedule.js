/**
 * unmask-schedule.js — how many masked tokens each diffusion step commits.
 * Port of OmniVoice's _get_time_steps + schedule (upstream commit d8ab65f,
 * "fix unmask schedule"): `numStep` intervals, the last step takes the
 * remainder. The pre-fix version used numStep + 1 intervals, which made the
 * final step commit most of the audio at once (68% at 20 steps).
 */

export function getTimeSteps(tStart, tEnd, numStep, tShift) {
  const steps = [];
  for (let i = 0; i <= numStep; i++) {
    let t = tStart + (tEnd - tStart) * (i / numStep);
    t = tShift * t / (1 + (tShift - 1) * t);
    steps.push(t);
  }
  return steps;
}

export function unmaskSchedule(totalMask, numStep, tShift) {
  const timesteps = getTimeSteps(0, 1, numStep, tShift);
  let rem = totalMask;
  const sched = [];
  for (let s = 0; s < numStep; s++) {
    const n = s === numStep - 1 ? rem : Math.min(Math.ceil(totalMask * (timesteps[s + 1] - timesteps[s])), rem);
    sched.push(n);
    rem -= n;
  }
  return sched;
}
