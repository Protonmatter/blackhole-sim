import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MAX_SIMULATION_TIME_SECONDS,
  nextRefinementState,
  parseFilmConfig,
  resolveInitialRunning,
  resolveUiMode,
  sanitizeSimulationTime,
} from '../src/ui_state.mjs';

test('Explore is the default and film overrides Lab', () => {
  assert.equal(resolveUiMode(new URLSearchParams()), 'explore');
  assert.equal(resolveUiMode(new URLSearchParams('ui=lab')), 'lab');
  assert.equal(resolveUiMode(new URLSearchParams('ui=unknown')), 'explore');
  assert.equal(resolveUiMode(new URLSearchParams('ui=lab&film=1')), 'film');
});

test('film values use exact defaults and clamps', () => {
  assert.deepEqual(parseFilmConfig(new URLSearchParams('film=1')), {
    enabled: true,
    timeSeconds: 0,
    width: 1280,
    height: 720,
  });
  assert.deepEqual(
    parseFilmConfig(new URLSearchParams('film=1&t=-2&w=99999&h=bad')),
    {
      enabled: true,
      timeSeconds: 0,
      width: 1920,
      height: 720,
    },
  );
});

test('invalid and fractional film values produce finite integer output', () => {
  const invalid = parseFilmConfig(
    new URLSearchParams('film=1&t=Infinity&w=0&h=-1'),
  );
  assert.deepEqual(invalid, {
    enabled: true,
    timeSeconds: 0,
    width: 64,
    height: 64,
  });

  const fractional = parseFilmConfig(
    new URLSearchParams('film=1&t=NaN&w=640.9&h=360.7'),
  );
  assert.deepEqual(fractional, {
    enabled: true,
    timeSeconds: 0,
    width: 640,
    height: 360,
  });
  assert.ok(Number.isFinite(fractional.width));
  assert.ok(Number.isInteger(fractional.width));
  assert.ok(Number.isFinite(fractional.height));
  assert.ok(Number.isInteger(fractional.height));
});

test('simulation time stays inside the supported browser GPU range', () => {
  assert.equal(
    parseFilmConfig(new URLSearchParams('film=1&t=1e308')).timeSeconds,
    MAX_SIMULATION_TIME_SECONDS,
  );
  assert.equal(sanitizeSimulationTime('1e308'), MAX_SIMULATION_TIME_SECONDS);
  assert.equal(sanitizeSimulationTime('-12'), 0);
  assert.equal(sanitizeSimulationTime('-12', true), -12);
  assert.equal(
    sanitizeSimulationTime('-1e308', true),
    -MAX_SIMULATION_TIME_SECONDS,
  );
  assert.equal(sanitizeSimulationTime('not-a-number'), 0);
});

test('reduced motion pauses only when running was not explicit', () => {
  assert.equal(resolveInitialRunning(new URLSearchParams(), true), false);
  assert.equal(resolveInitialRunning(new URLSearchParams(), false), true);
  assert.equal(
    resolveInitialRunning(new URLSearchParams('running=1'), true),
    true,
  );
  assert.equal(
    resolveInitialRunning(new URLSearchParams('running=0'), false),
    false,
  );
});

function sample(overrides = {}) {
  return {
    nowMs: 0,
    interactionVersion: 1,
    width: 1920,
    height: 1080,
    levelCount: 4,
    forceFull: false,
    ...overrides,
  };
}

test('refinement advances every 850ms and holds the final level', () => {
  const levelCount = 4;
  const stateAt0 = nextRefinementState(null, sample({ levelCount }));
  const stateAt849 = nextRefinementState(
    stateAt0,
    sample({ nowMs: 849, levelCount }),
  );
  const stateAt850 = nextRefinementState(
    stateAt849,
    sample({ nowMs: 850, levelCount }),
  );
  const stateLongAfter = nextRefinementState(
    stateAt850,
    sample({ nowMs: 5000, levelCount }),
  );
  const stateEvenLater = nextRefinementState(
    stateLongAfter,
    sample({ nowMs: 50000, levelCount }),
  );

  assert.equal(stateAt0.levelIndex, 0);
  assert.equal(stateAt849.levelIndex, 0);
  assert.equal(stateAt850.levelIndex, 1);
  assert.equal(stateLongAfter.levelIndex, levelCount - 1);
  assert.equal(stateEvenLater.levelIndex, levelCount - 1);
});

test('interaction and dimension changes reset refinement', () => {
  const refined = nextRefinementState(
    null,
    sample({ nowMs: 5000, forceFull: true }),
  );
  assert.equal(refined.levelIndex, 3);

  const interactionReset = nextRefinementState(
    refined,
    sample({ nowMs: 5100, interactionVersion: 2 }),
  );
  assert.equal(interactionReset.levelIndex, 0);

  const widthReset = nextRefinementState(
    refined,
    sample({ nowMs: 5100, width: 1280 }),
  );
  assert.equal(widthReset.levelIndex, 0);

  const heightReset = nextRefinementState(
    refined,
    sample({ nowMs: 5100, height: 720 }),
  );
  assert.equal(heightReset.levelIndex, 0);
});

test('unchanged inputs never decrease and force-full selects the last level', () => {
  const state = nextRefinementState(
    null,
    sample({ nowMs: 3400, forceFull: true }),
  );
  const clockMovedBack = nextRefinementState(
    state,
    sample({ nowMs: 100, forceFull: false }),
  );

  assert.equal(state.levelIndex, 3);
  assert.equal(clockMovedBack.levelIndex, 3);
  assert.equal(
    nextRefinementState(null, sample({ levelCount: 0 })).levelIndex,
    0,
  );
  assert.equal(
    nextRefinementState(null, sample({ levelCount: -2 })).levelIndex,
    0,
  );
});
