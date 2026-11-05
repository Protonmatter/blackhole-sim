import test from 'node:test';
import assert from 'node:assert/strict';

import {
  captureSubmittedFrame,
  createExclusiveOperationRunner,
  watchGpuDevice,
} from '../src/runtime_control.mjs';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function fakeDevice(lostPromise = new Promise(() => {})) {
  const listeners = new Map();
  return {
    lost: lostPromise,
    addEventListener(type, listener) {
      listeners.set(type, listener);
    },
    removeEventListener(type, listener) {
      if (listeners.get(type) === listener) listeners.delete(type);
    },
    emit(type, event) {
      listeners.get(type)?.(event);
    },
  };
}

test('capture snapshots the submitted frame before waiting for GPU completion', async () => {
  const encoding = deferred();
  const submitted = deferred();
  const calls = [];

  const result = captureSubmittedFrame({
    render() {
      calls.push('render');
    },
    encode() {
      calls.push('encode');
      return encoding.promise;
    },
    submitted() {
      calls.push('submitted');
      return submitted.promise;
    },
  });

  assert.deepEqual(calls, ['render', 'encode', 'submitted']);

  let settled = false;
  result.finally(() => {
    settled = true;
  });
  encoding.resolve('encoded-png');
  await Promise.resolve();
  assert.equal(settled, false);

  submitted.resolve();
  assert.equal(await result, 'encoded-png');
});

test('public render and capture operations reject overlap and recover afterward', async () => {
  const release = deferred();
  const runExclusive = createExclusiveOperationRunner();
  let overlappingTaskStarted = false;

  const first = runExclusive(async () => {
    await release.promise;
    return 'first-result';
  });

  await assert.rejects(
    runExclusive(async () => {
      overlappingTaskStarted = true;
      return 'overlap-result';
    }),
    /already in progress/i,
  );
  assert.equal(overlappingTaskStarted, false);

  release.resolve();
  assert.equal(await first, 'first-result');
  assert.equal(await runExclusive(async () => 'next-result'), 'next-result');
});

test('uncaptured GPU errors invalidate the runtime exactly once', () => {
  const device = fakeDevice();
  const failures = [];
  const runtime = watchGpuDevice(device, (error) => failures.push(error));
  let prevented = 0;

  device.emit('uncapturederror', {
    error: new Error('validation exploded'),
    preventDefault() {
      prevented += 1;
    },
  });
  device.emit('uncapturederror', {
    error: new Error('second error'),
    preventDefault() {
      prevented += 1;
    },
  });

  assert.equal(prevented, 2);
  assert.equal(runtime.failed, true);
  assert.equal(failures.length, 1);
  assert.match(failures[0].message, /validation exploded/);
  assert.throws(() => runtime.assertHealthy(), /validation exploded/);

  let lateStatusUpdates = 0;
  assert.equal(runtime.runIfHealthy(() => {
    lateStatusUpdates += 1;
  }), false);
  assert.equal(lateStatusUpdates, 0);
});

test('device loss invalidates readiness through the same runtime guard', async () => {
  const lost = deferred();
  const device = fakeDevice(lost.promise);
  const failures = [];
  const runtime = watchGpuDevice(device, (error) => failures.push(error));

  lost.resolve({ reason: 'destroyed', message: 'adapter reset' });
  await lost.promise;
  await Promise.resolve();

  assert.equal(runtime.failed, true);
  assert.equal(failures.length, 1);
  assert.match(failures[0].message, /device lost.*adapter reset/i);
});
