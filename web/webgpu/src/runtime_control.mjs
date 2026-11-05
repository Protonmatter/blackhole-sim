function asError(value, prefix) {
  const detail = value instanceof Error
    ? value.message
    : (value && typeof value.message === 'string' ? value.message : String(value));
  return new Error(`${prefix}: ${detail || 'unknown error'}`);
}

/**
 * Submit a fresh frame and start the canvas snapshot synchronously, before the
 * browser is allowed to recycle the WebGPU presentation buffer. Resolution
 * waits for both the GPU submission and the encoder.
 *
 * @template T
 * @param {{render: () => void, encode: () => Promise<T> | T, submitted: () => Promise<unknown>}} operations
 * @returns {Promise<T>}
 */
export async function captureSubmittedFrame({ render, encode, submitted }) {
  render();

  let encoded;
  try {
    encoded = Promise.resolve(encode());
  } catch (error) {
    encoded = Promise.reject(error);
  }

  let completion;
  try {
    completion = Promise.resolve(submitted());
  } catch (error) {
    completion = Promise.reject(error);
  }

  const [result] = await Promise.all([encoded, completion]);
  return result;
}

/**
 * Reject overlapping operations that would otherwise mutate one shared canvas
 * and simulation state concurrently.
 *
 * @returns {<T>(operation: () => Promise<T> | T) => Promise<T>}
 */
export function createExclusiveOperationRunner() {
  let active = false;

  return async function runExclusive(operation) {
    if (active) {
      throw new Error('A render or capture operation is already in progress');
    }
    active = true;
    try {
      return await operation();
    } finally {
      active = false;
    }
  };
}

/**
 * Convert asynchronous WebGPU device failures into one terminal runtime state.
 *
 * @param {{lost: Promise<unknown>, addEventListener: Function, removeEventListener?: Function}} device
 * @param {(error: Error) => void} onFailure
 */
export function watchGpuDevice(device, onFailure) {
  let failure = null;
  let active = true;

  const fail = (reason) => {
    if (!active || failure) return failure;
    failure = reason instanceof Error ? reason : asError(reason, 'WebGPU runtime failed');
    onFailure(failure);
    return failure;
  };

  const onUncapturedError = (event) => {
    if (typeof event.preventDefault === 'function') event.preventDefault();
    fail(asError(event.error, 'Uncaptured WebGPU error'));
  };

  device.addEventListener('uncapturederror', onUncapturedError);
  Promise.resolve(device.lost).then(
    (info) => {
      if (!active) return;
      const detail = info && (info.message || info.reason)
        ? (info.message || info.reason)
        : 'unknown reason';
      fail(new Error(`WebGPU device lost: ${detail}`));
    },
    (error) => fail(asError(error, 'WebGPU device loss notification failed')),
  );

  return {
    get failed() {
      return failure !== null;
    },
    get error() {
      return failure;
    },
    fail,
    assertHealthy() {
      if (failure) throw failure;
    },
    runIfHealthy(callback) {
      if (failure) return false;
      callback();
      return true;
    },
    dispose() {
      active = false;
      if (typeof device.removeEventListener === 'function') {
        device.removeEventListener('uncapturederror', onUncapturedError);
      }
    },
  };
}
