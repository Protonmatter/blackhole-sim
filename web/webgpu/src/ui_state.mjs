const FILM_DEFAULT_WIDTH = 1280;
const FILM_DEFAULT_HEIGHT = 720;
const FILM_MIN_WIDTH = 64;
const FILM_MAX_WIDTH = 1920;
const FILM_MIN_HEIGHT = 64;
const FILM_MAX_HEIGHT = 1080;
const REFINEMENT_INTERVAL_MS = 850;

/**
 * @param {URLSearchParams} params
 * @returns {'explore' | 'lab' | 'film'}
 */
export function resolveUiMode(params) {
  if (params.get('film') === '1') return 'film';
  return params.get('ui') === 'lab' ? 'lab' : 'explore';
}

function finiteNonNegative(value, fallback) {
  const parsed = Number.parseFloat(value ?? '');
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function boundedInteger(value, fallback, minimum, maximum) {
  const parsed = Number.parseInt(value ?? '', 10);
  const finite = Number.isFinite(parsed) ? parsed : fallback;
  return Math.min(maximum, Math.max(minimum, finite));
}

/**
 * @param {URLSearchParams} params
 * @returns {{enabled: boolean, timeSeconds: number, width: number, height: number}}
 */
export function parseFilmConfig(params) {
  return {
    enabled: params.get('film') === '1',
    timeSeconds: finiteNonNegative(params.get('t'), 0),
    width: boundedInteger(
      params.get('w'),
      FILM_DEFAULT_WIDTH,
      FILM_MIN_WIDTH,
      FILM_MAX_WIDTH,
    ),
    height: boundedInteger(
      params.get('h'),
      FILM_DEFAULT_HEIGHT,
      FILM_MIN_HEIGHT,
      FILM_MAX_HEIGHT,
    ),
  };
}

/**
 * @param {URLSearchParams} params
 * @param {boolean} prefersReducedMotion
 * @returns {boolean}
 */
export function resolveInitialRunning(params, prefersReducedMotion) {
  const explicit = params.get('running');
  if (explicit === '0') return false;
  if (explicit === '1') return true;
  return !prefersReducedMotion;
}

/**
 * @typedef {object} RefinementState
 * @property {number} interactionVersion
 * @property {number} width
 * @property {number} height
 * @property {number} stableSinceMs
 * @property {number} levelIndex
 */

/**
 * @typedef {object} RefinementSample
 * @property {number} nowMs
 * @property {number} interactionVersion
 * @property {number} width
 * @property {number} height
 * @property {number} levelCount
 * @property {boolean} forceFull
 */

/**
 * @param {RefinementState | null} previous
 * @param {RefinementSample} sample
 * @returns {RefinementState}
 */
export function nextRefinementState(previous, sample) {
  const levelCount = Number.isFinite(sample.levelCount)
    ? Math.max(0, Math.trunc(sample.levelCount))
    : 0;
  const lastIndex = Math.max(0, levelCount - 1);
  const nowMs = Number.isFinite(sample.nowMs) ? sample.nowMs : 0;
  const invalidated =
    previous === null ||
    previous.interactionVersion !== sample.interactionVersion ||
    previous.width !== sample.width ||
    previous.height !== sample.height;
  const stableSinceMs = invalidated ? nowMs : previous.stableSinceMs;
  const elapsedMs = Math.max(0, nowMs - stableSinceMs);
  const computedIndex = Math.min(
    lastIndex,
    Math.floor(elapsedMs / REFINEMENT_INTERVAL_MS),
  );
  const previousIndex = invalidated ? 0 : previous.levelIndex;
  const levelIndex = sample.forceFull
    ? lastIndex
    : Math.min(lastIndex, Math.max(previousIndex, computedIndex));

  return {
    interactionVersion: sample.interactionVersion,
    width: sample.width,
    height: sample.height,
    stableSinceMs,
    levelIndex,
  };
}
