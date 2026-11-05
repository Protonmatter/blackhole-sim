# Cinematic-First WebGPU Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing real WebGPU renderer open as a cinematic Explore experience, preserve the current controls as an in-page Lab mode, make Stokes refinement interaction-driven and monotonic, and expose deterministic frame capture.

**Architecture:** Keep `web/webgpu/index.html` as the only renderer entry point and keep one canvas, one WebGPU device, and one simulation state. Extract URL/refinement decisions into a dependency-free pure ES module, leave WebGPU and DOM ownership in `main.js`, and centralize frame scheduling so interactive animation and deterministic film rendering share the same renderer factories.

**Tech Stack:** Static HTML/CSS, browser ES modules, WebGPU/WGSL, Node 22 built-in test runner, Python/pytest asset-contract tests, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-04-cinematic-webgpu-experience-design.md`

**Baseline:** Plan authored against commit `a1df45c`. If the branch has moved, re-read `AGENTS.md` and re-anchor line references before editing; preserve intervening user changes.

## Global Constraints

- Read `AGENTS.md` before acting and create an isolated worktree with `superpowers:using-git-worktrees` at execution time.
- Use test-driven development: add each failing test, observe the expected failure, implement only the owning behavior, and rerun the focused test.
- Do not modify WGSL shaders, Python/native physics, coefficient generation, geodesic equations, or parity tolerances.
- Do not add a framework, bundler, npm dependency, generated capture, binary artifact, or checked-in render output.
- Preserve all existing URL parameters and Lab control semantics.
- Default UI is Explore; `ui=lab` opens Lab; `film=1` overrides both and hides all UI.
- Film defaults are `t=0`, `w=1280`, and `h=720`; clamp width to `64..1920`, height to `64..1080`, and time to `0..100000` seconds.
- Explore must say `Educational browser renderer` and must not claim `validated`, `accurate`, or `physics-equivalent` rendering.
- Never claim GPU/native physics parity from these presentation changes.
- Per repository policy, do not commit, push, tag, or publish unless the human explicitly authorizes it. Each task includes a suggested commit message only as an optional checkpoint.

## File Map

- Create `web/webgpu/src/ui_state.mjs`: pure URL, reduced-motion, film-config, and progressive-refinement state logic.
- Create `web/webgpu/tests/ui_state.test.mjs`: dependency-free Node unit tests for that pure module.
- Create `web/webgpu/src/runtime_control.mjs`: dependency-free capture ordering and GPU-device lifecycle control.
- Create `web/webgpu/tests/runtime_control.test.mjs`: orchestration tests for delayed capture, uncaptured errors, and device loss.
- Create `python/tests/test_webgpu_ui_contract.py`: semantic HTML and browser-entrypoint source contracts that do not require a GPU.
- Modify `web/webgpu/index.html`: cinematic HUD, accessible canvas, hidden/inert Lab panel, film-mode styles, and responsive/reduced-motion CSS.
- Modify `web/webgpu/src/main.js`: mode wiring, preserved URL state, monotonic refinement integration, renderer factories, central scheduler, deterministic API, readiness/error markers, and capture reuse.
- Modify `python/tests/test_webgpu_progressive_assets.py`: integration assertions for the new refinement and scheduler seams while preserving existing bounded-work checks.
- Modify `.github/workflows/ci.yml`: required Node 22 web test job.
- Modify `web/webgpu/README.md`: UI modes, deterministic URL/API contract, and local smoke steps.
- Modify `web/README_SHOWCASE.md`: cinematic renderer first, Lab URL, and unchanged scientific boundary.

## Review Focus

- Existing stateful URLs: Task 2 tests that `shader`, quality, running/time, slider, and diagnostics parameters remain owned by the existing restoration path while `ui` only changes presentation.
- Hidden Lab accessibility: Task 2 tests `aria-hidden` plus runtime `inert`, focusable canvas instructions, and Escape-to-close behavior.
- Rapid interaction and resize: Tasks 1 and 3 test that either invalidates refinement, which then advances monotonically and never wraps.
- Invalid film input: Task 1 tests absent, non-numeric, infinite, negative, and oversized `t/w/h` values against exact defaults and clamps.
- WebGPU or shader startup failure: Task 4 tests that the source installs an error marker and never marks readiness on failure; the browser smoke verifies the visible fallback remains usable.

## Production-Readiness Review Addendum (2026-11-05)

- Values may be finite in JavaScript yet overflow browser `f32` uniforms or
  intermediate phase arithmetic. Clamp URL time, film time, animation advance,
  and `renderAt` to the documented browser presentation range.
- A delayed canvas export must submit a fresh frame and begin its snapshot before
  the WebGPU presentation buffer can be recycled. Runtime acceptance must decode
  the PNG, confirm its exact dimensions, and reject blank or uniform output;
  positive base64 length alone is insufficient.
- Treat `device.lost` and uncaptured GPU errors as terminal for the page:
  readiness clears, the RAF loop stops, an error marker becomes visible, and
  later public API calls reject.
- Reject overlapping `renderAt`/`frame` calls so shared simulation state and the
  presentation texture have exactly one owner per deterministic operation.
- A fallback-only smoke is useful failure-path evidence but is not sufficient for
  production acceptance. Record at least one successful disk, volume, and Stokes
  run on a supported WebGPU adapter, including delayed nonblank capture.
- Keep the hardening layer outside WGSL, coefficient formulas, geodesics,
  radiative transfer, Python/native physics, and parity tolerances.

---

### Task 1: Pure UI and refinement state module

**Files:**
- Create: `web/webgpu/src/ui_state.mjs`
- Create: `web/webgpu/tests/ui_state.test.mjs`

**Interfaces:**
- Consumes: Standard `URLSearchParams`; plain JavaScript values only.
- Produces:
  - `resolveUiMode(params: URLSearchParams): "explore" | "lab" | "film"`
  - `parseFilmConfig(params: URLSearchParams): { enabled: boolean, timeSeconds: number, width: number, height: number }`
  - `resolveInitialRunning(params: URLSearchParams, prefersReducedMotion: boolean): boolean`
  - `nextRefinementState(previous, sample): { interactionVersion: number, width: number, height: number, stableSinceMs: number, levelIndex: number }`

- [ ] **Step 1: Write failing Node tests for UI and film parsing**

Create `web/webgpu/tests/ui_state.test.mjs` with `node:test` and `node:assert/strict`. Import the four interfaces above and add these cases:

```js
test('Explore is the default and film overrides Lab', () => {
  assert.equal(resolveUiMode(new URLSearchParams()), 'explore');
  assert.equal(resolveUiMode(new URLSearchParams('ui=lab')), 'lab');
  assert.equal(resolveUiMode(new URLSearchParams('ui=unknown')), 'explore');
  assert.equal(resolveUiMode(new URLSearchParams('ui=lab&film=1')), 'film');
});

test('film values use exact defaults and clamps', () => {
  assert.deepEqual(parseFilmConfig(new URLSearchParams('film=1')), {
    enabled: true, timeSeconds: 0, width: 1280, height: 720,
  });
  assert.deepEqual(parseFilmConfig(new URLSearchParams('film=1&t=-2&w=99999&h=bad')), {
    enabled: true, timeSeconds: 0, width: 1920, height: 720,
  });
});

test('reduced motion pauses only when running was not explicit', () => {
  assert.equal(resolveInitialRunning(new URLSearchParams(), true), false);
  assert.equal(resolveInitialRunning(new URLSearchParams('running=1'), true), true);
  assert.equal(resolveInitialRunning(new URLSearchParams('running=0'), false), false);
});
```

Also assert that `t=Infinity`, `t=NaN`, `w=0`, `h=-1`, and decimal dimensions produce finite, integer, clamped results.

- [ ] **Step 2: Write failing Node tests for refinement progression**

Add tests that construct a first state, sample at `0`, `849`, `850`, and several later intervals, and assert:

```js
assert.equal(stateAt0.levelIndex, 0);
assert.equal(stateAt849.levelIndex, 0);
assert.equal(stateAt850.levelIndex, 1);
assert.equal(stateLongAfter.levelIndex, levelCount - 1);
assert.equal(stateEvenLater.levelIndex, levelCount - 1);
```

Add separate cases proving a changed `interactionVersion`, width, or height resets to `0`, unchanged inputs never decrease the selected index, `forceFull: true` selects the last index immediately, and `levelCount <= 0` safely returns index `0`.

- [ ] **Step 3: Run the Node tests and observe the import failure**

Run: `node --test web/webgpu/tests/ui_state.test.mjs`

Expected: FAIL because `web/webgpu/src/ui_state.mjs` does not exist or does not export the required functions.

- [ ] **Step 4: Implement `ui_state.mjs`**

Use named exports and JSDoc types. Exact rules:

- `film=1` is the only value that enables film.
- `ui=lab` is the only value that selects Lab; everything else selects Explore.
- An explicit `running=0` is false and explicit `running=1` is true. With no explicit value, reduced motion returns false and all other users return true.
- Parse time with `Number.parseFloat`; non-finite or negative becomes `0`, and
  values above `100000` seconds clamp to `100000`.
- Parse dimensions with `Number.parseInt(..., 10)`; non-finite input uses the default, then clamp width to `64..1920` and height to `64..1080`.
- Refinement invalidates on first call, interaction-version change, or dimension change. It advances one index per complete `850 ms` since invalidation, uses `Math.max(previous.levelIndex, computedIndex)` when inputs are unchanged, clamps to the last index, never uses modulo, and honors `forceFull`.

- [ ] **Step 5: Run the Node tests**

Run: `node --test web/webgpu/tests/ui_state.test.mjs`

Expected: all tests PASS with no third-party packages installed.

- [ ] **Step 6: Record the optional checkpoint**

Do not commit without explicit authorization. If authorization exists:

```bash
git add web/webgpu/src/ui_state.mjs web/webgpu/tests/ui_state.test.mjs
git commit -m "test: define cinematic UI state contracts"
```

---

### Task 2: Cinematic Explore shell and accessible Lab mode

**Files:**
- Create: `python/tests/test_webgpu_ui_contract.py`
- Modify: `web/webgpu/index.html:7-83`
- Modify: `web/webgpu/src/main.js:1-134, 488-560, 588-596`

**Interfaces:**
- Consumes: `resolveUiMode`, `parseFilmConfig`, and `resolveInitialRunning` from Task 1; existing DOM IDs for shader, sliders, telemetry, and action buttons.
- Produces:
  - DOM IDs `hud`, `labToggle`, and `labPanel`.
  - `setUiMode(mode: "explore" | "lab" | "film", updateUrl?: boolean): void` in `main.js`.
  - One unchanged WebGPU canvas and unchanged control values across UI-mode switches.

- [ ] **Step 1: Write failing semantic-markup tests**

Create `python/tests/test_webgpu_ui_contract.py`. Use `html.parser.HTMLParser` from the standard library to collect tags and attributes. Add tests named:

- `test_webgpu_entrypoint_defaults_to_cinematic_explore`
- `test_webgpu_canvas_is_keyboard_focusable_and_described`
- `test_lab_panel_has_accessible_hidden_contract`
- `test_explore_copy_preserves_scientific_boundary`

Assert the exact contracts:

```python
assert body_attrs["data-ui"] == "explore"
assert canvas_attrs["tabindex"] == "0"
assert "Drag" in canvas_attrs["aria-label"]
assert "Space" in canvas_attrs["aria-label"]
assert ids["labToggle"]["type"] == "button"
assert ids["labPanel"]["aria-hidden"] == "true"
assert "inert" in ids["labPanel"]
assert "Educational browser renderer" in html
```

For boundary copy, inspect only visible Explore markup and assert that its lowercased text does not contain `physics-equivalent` or a standalone claim containing `validated` or `accurate`.

- [ ] **Step 2: Write failing source-integration tests**

In the same file add:

- `test_main_wires_ui_mode_without_renderer_reload`
- `test_hidden_lab_is_inert_and_escape_closes_it`
- `test_reduced_motion_default_is_wired`

Assert `main.js` imports Task 1 functions, defines `setUiMode`, sets both `labPanel.inert` and `aria-hidden`, uses `history.replaceState`, handles `Escape`, and uses `matchMedia('(prefers-reduced-motion: reduce)')`. Assert the Lab click handler does not assign `location.href` and does not call `applyControlState`.

- [ ] **Step 3: Run focused Python tests and observe failures**

Run from `python/`:

`python -m pytest -q tests/test_webgpu_ui_contract.py`

Expected: FAIL because the cinematic markup and mode wiring do not exist.

- [ ] **Step 4: Restructure `index.html` without duplicating controls**

Make these exact structural changes:

- Set `<body data-ui="explore">`.
- Change the canvas to `tabindex="0"` and give it an `aria-label` covering drag, wheel, Space, and R.
- Create `<header id="hud">` containing the title `BlackHole Sim`, subtitle `Interactive Kerr spacetime`, visible text `Educational browser renderer`, the existing `#status`, and the existing `#playToggle`, `#resetView`, and `#captureFrame` buttons.
- Add one `#labToggle` button to that same toolbar; initial text is `Open Lab` and `aria-pressed="false"`.
- Convert the existing advanced panel into `<aside id="labPanel" aria-hidden="true" inert>`, retaining every shader, quality, slider, preset, telemetry, and units element exactly once.
- Do not create a second canvas, toolbar, or copy of any input.

Update CSS so Explore shows the compact HUD and hides the Lab panel, Lab reveals the panel without covering the HUD controls, film hides both, mobile layouts remain usable, canvas focus is visible, and `prefers-reduced-motion: reduce` disables nonessential transitions.

- [ ] **Step 5: Wire UI mode and reduced-motion startup in `main.js`**

Import Task 1 at the top of `main.js`. Cache `labToggle` and `labPanel`. Parse `filmConfig` once. Initialize `simulationState.running` with `resolveInitialRunning(params, matchMedia(...).matches)`.

Implement `setUiMode` with these behaviors:

- Set `document.body.dataset.ui`.
- Set `labPanel.inert = mode !== 'lab'`.
- Set `labPanel` `aria-hidden` to the string form of `mode !== 'lab'`.
- Set `labToggle` text to `Close Lab` in Lab and `Open Lab` otherwise; keep `aria-pressed` synchronized.
- When `updateUrl` is true and mode is not film, update only the `ui` parameter using `history.replaceState`; do not navigate and do not touch renderer controls.

Call `setUiMode(resolveUiMode(params), false)` during restoration. Toggle Explore/Lab on `#labToggle`; Escape closes Lab. Do not install pointer or keyboard simulation handlers in film mode. Keep existing shader changes and preset changes using their current URL navigation behavior.

- [ ] **Step 6: Run focused UI and existing asset tests**

Run from `python/`:

```bash
python -m pytest -q tests/test_webgpu_ui_contract.py tests/test_webgpu_progressive_assets.py tests/test_web_showcase_workflow.py
```

Expected: PASS. Then run `node --test web/webgpu/tests/ui_state.test.mjs` from the repository root; expected PASS.

- [ ] **Step 7: Record the optional checkpoint**

Do not commit without explicit authorization. If authorization exists:

```bash
git add web/webgpu/index.html web/webgpu/src/main.js python/tests/test_webgpu_ui_contract.py
git commit -m "feat: make WebGPU experience cinematic first"
```

---

### Task 3: Interaction-driven monotonic Stokes refinement

**Files:**
- Modify: `web/webgpu/src/main.js:408-460`
- Modify: `python/tests/test_webgpu_progressive_assets.py:4-47`

**Interfaces:**
- Consumes: `nextRefinementState` from Task 1, existing `progressiveLevels(...)`, `simulationState.interactionVersion`, and canvas dimensions.
- Produces: Stokes level selection that resets only on invalidation, refines every stable `850 ms`, reaches full resolution, and remains there.

- [ ] **Step 1: Strengthen the failing integration test**

In `test_webgpu_main_has_progressive_compute_renderer_hooks`, require these tokens:

```python
for token in (
    "nextRefinementState",
    "interactionVersion: simulationState.interactionVersion",
    "levelCount: levels.length",
    "forceFull",
):
    assert token in text
assert "% levels.length" not in text
```

Also assert `interactionVersion` appears in a call argument, not only at its declaration/increment sites.

- [ ] **Step 2: Run the focused test and observe the old timer-cycle failure**

Run from `python/`:

`python -m pytest -q tests/test_webgpu_progressive_assets.py::test_webgpu_main_has_progressive_compute_renderer_hooks`

Expected: FAIL because the Stokes frame still selects a level with a repeating modulo clock.

- [ ] **Step 3: Integrate the refinement reducer in the Stokes renderer**

Add renderer-local `refinementState = null`. On every Stokes render:

1. Compute `levels = progressiveLevels(canvas.width, canvas.height, 480)`.
2. Call `nextRefinementState(refinementState, { nowMs: timeMs, interactionVersion: simulationState.interactionVersion, width: canvas.width, height: canvas.height, levelCount: levels.length, forceFull: filmConfig.enabled })`.
3. Store the returned state.
4. Select `levels[refinementState.levelIndex]`, falling back to the last concrete level only if needed defensively.

Remove the modulo selector. Do not change workgroup size, maximum steps, coefficient bricks, readback behavior, or shader code.

- [ ] **Step 4: Run refinement tests**

Run from the repository root:

```bash
node --test web/webgpu/tests/ui_state.test.mjs
cd python
python -m pytest -q tests/test_webgpu_progressive_assets.py
```

Expected: all PASS.

- [ ] **Step 5: Record the optional checkpoint**

Do not commit without explicit authorization. If authorization exists:

```bash
git add web/webgpu/src/main.js python/tests/test_webgpu_progressive_assets.py
git commit -m "fix: hold final WebGPU refinement level"
```

---

### Task 4: Central scheduler and deterministic frame API

**Files:**
- Modify: `web/webgpu/src/main.js:160-170, 323-483, 521-596`
- Create: `web/webgpu/src/runtime_control.mjs`
- Create: `web/webgpu/tests/runtime_control.test.mjs`
- Modify: `python/tests/test_webgpu_ui_contract.py`
- Modify: `python/tests/test_webgpu_progressive_assets.py`

**Interfaces:**
- Consumes: `filmConfig` from Task 1, Task 3 refinement integration, existing WebGPU initialization and draw code.
- Produces:
  - Renderer factories returning `{ renderFrame(timeMs: number, options?: { forceFull?: boolean }): void }`.
  - One central interactive animation loop.
  - `window.__blackhole.state(): object`.
  - `window.__blackhole.renderAt(seconds: number): Promise<object>`.
  - `window.__blackhole.frame(type?: string, quality?: number): Promise<string>`.
  - `data-blackhole-ready` and `data-blackhole-error` document markers.

- [ ] **Step 1: Write failing scheduler and public-API source tests**

Add tests named:

- `test_renderer_factories_return_frames_to_one_scheduler`
- `test_main_exposes_deterministic_browser_capture_contract`
- `test_film_mode_uses_exact_dimensions_and_readiness_markers`
- `test_startup_failure_does_not_claim_readiness`

Assert `main.js` contains `createStokesRenderer`, `createFragmentRenderer`, `renderFrame`, `startInteractiveLoop`, `window.__blackhole`, `renderAt`, `device.queue.onSubmittedWorkDone`, `blackholeReady`, and `blackholeError`. Assert only the central scheduler contains `requestAnimationFrame(frame)` or its chosen single-loop equivalent; renderer factories must not schedule themselves. Assert film resize uses `filmConfig.width` and `filmConfig.height` directly.

- [ ] **Step 2: Run focused tests and observe failures**

Run from `python/`:

```bash
python -m pytest -q tests/test_webgpu_ui_contract.py tests/test_webgpu_progressive_assets.py
```

Expected: FAIL because both renderer functions currently own their own RAF loops and no deterministic API exists.

- [ ] **Step 3: Refactor both renderer paths into factories**

Rename/refactor the current async entry functions to:

- `async function createStokesRenderer()`
- `async function createFragmentRenderer()`

Each factory may allocate its existing resources and closures, but must return one object containing `renderFrame(timeMs, { forceFull = false } = {})`. A render-frame call performs exactly one submission and never advances simulation time, updates global telemetry, or schedules another frame. Stokes passes `forceFull` into Task 3's reducer; fragment rendering accepts and ignores it.

Keep device, buffer, pipeline, shader, readback, and draw commands otherwise unchanged.

- [ ] **Step 4: Centralize sizing, simulation clock, telemetry, and RAF**

Change `resize()` so film mode writes the exact parsed film dimensions and interactive mode retains its existing client-size/quality-scale behavior.

Implement one `startInteractiveLoop(activeRenderer)` whose frame callback:

1. calls `updateSimulationClock(timeMs)`;
2. calls `resize()`;
3. calls `activeRenderer.renderFrame(timeMs, { forceFull: simulationState.forceFull })`;
4. calls `updateTelemetry(timeMs)`;
5. schedules the next frame.

After the first submitted frame completes, set
`document.documentElement.dataset.blackholeReady = 'true'` once. Do not await the GPU queue on every interactive frame.

- [ ] **Step 5: Implement the deterministic public API**

Add one serializable state snapshot with exact keys:

```js
{
  ui, film, ready, shader, quality, running, timeSeconds,
  resolution: [canvas.width, canvas.height], interactionVersion,
}
```

Install `window.__blackhole` after the active renderer exists:

- `state()` returns a fresh snapshot.
- `renderAt(seconds)` sanitizes to `0..100000` seconds, pauses via `setPlayState(false)`, assigns `simulationState.timeSeconds`, sets `simulationState.forceFull = true`, increments `interactionVersion`, resizes, renders once with `{ forceFull: true }`, awaits `device.queue.onSubmittedWorkDone()`, sets readiness, and returns `state()`.
- `frame(type = 'image/png', quality = 0.95)` submits a fresh frame, begins `canvas.toBlob` synchronously before the presentation buffer can be recycled, awaits both encoding and submitted GPU work, rejects if encoding returns null, converts the blob bytes to a base64 string in bounded chunks, and returns it without a data-URL prefix.
- `renderAt` and `frame` reject overlapping calls with a clear busy error; they never mutate shared simulation state or use the presentation texture concurrently.

Introduce `markInteraction()` as the only user-driven increment path for
`interactionVersion`; it also clears `simulationState.forceFull`. Route slider,
preset, pointer, wheel, reset, and other user-driven simulation changes through
it. `renderAt` intentionally performs its own automation-driven increment after
setting `forceFull`. This keeps `renderAt` at full refinement while background
RAF redraws the same paused state, and returns to normal progressive behavior
only after real user input.

Route the Capture button through the same encoding helper, prepending the correct data-URL header only for the temporary download link. Revoke any object URL that remains in use. Do not create or commit a capture file during tests.

- [ ] **Step 6: Implement film initialization and failure markers**

When `filmConfig.enabled`:

- skip pointer, wheel, and global simulation-key handlers;
- set simulation time from `filmConfig.timeSeconds` and running false;
- render exactly once with force-full refinement;
- await GPU completion;
- mark readiness;
- do not start RAF.

Wrap WebGPU initialization, shader fetch/compile setup, renderer creation, and first render in one `try/catch`. On failure, retain the visible existing fallback/error message, set `data-blackhole-error` to a concise message, delete any stale `data-blackhole-ready`, and do not claim readiness.

- [ ] **Step 7: Run automated tests**

Run:

```bash
node --test web/webgpu/tests/ui_state.test.mjs web/webgpu/tests/runtime_control.test.mjs
cd python
python -m pytest -q tests/test_webgpu_ui_contract.py tests/test_webgpu_progressive_assets.py tests/test_web_showcase_workflow.py
```

Expected: all PASS.

- [ ] **Step 8: Run local browser smoke without saving artifacts**

From `web/`, start `python -m http.server 8080`, then inspect:

- `http://127.0.0.1:8080/webgpu/` — Explore HUD is default and the real renderer remains visible.
- `http://127.0.0.1:8080/webgpu/?ui=lab` — Lab opens with restored controls.
- `http://127.0.0.1:8080/webgpu/?shader=stokes&quality=fast&steps=80&film=1&t=4&w=640&h=360` — no UI, exact `640x360`, no animation, and ready marker after GPU completion.

In browser developer tools, run:

```js
const a = await window.__blackhole.renderAt(4);
const first = await window.__blackhole.frame();
await window.__blackhole.renderAt(4);
const second = await window.__blackhole.frame();
({ a, same: first === second, bytes: first.length, ready: document.documentElement.dataset.blackholeReady })
```

Expected: `same === true`, `bytes > 0`, `ready === 'true'`, `a.timeSeconds === 4`, and `a.resolution` matches the requested film size. If the local adapter is unavailable, record the exact unavailable state and still verify that `blackholeError` is set while `blackholeReady` is absent.

- [ ] **Step 9: Record the optional checkpoint**

Do not commit without explicit authorization. If authorization exists:

```bash
git add web/webgpu/src/main.js web/webgpu/src/runtime_control.mjs web/webgpu/tests/runtime_control.test.mjs python/tests/test_webgpu_ui_contract.py python/tests/test_webgpu_progressive_assets.py
git commit -m "feat: add deterministic WebGPU frame capture"
```

---

### Task 5: Required web CI and operator documentation

**Files:**
- Modify: `.github/workflows/ci.yml:16-51, 191-209`
- Modify: `python/tests/test_webgpu_ui_contract.py`
- Modify: `web/webgpu/README.md`
- Modify: `web/README_SHOWCASE.md`

**Interfaces:**
- Consumes: Node test command and browser contracts from Tasks 1-4.
- Produces: Required CI coverage and user-facing run/capture instructions with no new runtime dependency.

- [ ] **Step 1: Write the failing CI contract test**

Add `test_ci_requires_dependency_free_web_tests` to `python/tests/test_webgpu_ui_contract.py`. Read `.github/workflows/ci.yml` as text and assert it contains:

```python
assert "web:" in workflow
assert "actions/setup-node@v4" in workflow
assert 'node-version: "22"' in workflow
assert "node --test web/webgpu/tests/ui_state.test.mjs web/webgpu/tests/runtime_control.test.mjs" in workflow
assert "needs: [hygiene, web, python, rust, science-contracts, native-wheel]" in workflow
```

- [ ] **Step 2: Run the focused test and observe failure**

Run from `python/`:

`python -m pytest -q tests/test_webgpu_ui_contract.py::test_ci_requires_dependency_free_web_tests`

Expected: FAIL because no required web job exists.

- [ ] **Step 3: Add the required `web` CI job**

In `.github/workflows/ci.yml`, add a job named `web` that:

- needs `hygiene`;
- runs on `ubuntu-24.04` with a 10-minute timeout;
- checks out the repository;
- uses `actions/setup-node@v4` with Node `22`;
- runs `node --test web/webgpu/tests/ui_state.test.mjs web/webgpu/tests/runtime_control.test.mjs` from the repository root, so omitting either runtime file fails CI.

Add `web` to the exact `required.needs` list asserted above. Do not add npm install or a lockfile.

- [ ] **Step 4: Update browser documentation**

In `web/webgpu/README.md`, document:

- Explore as the default real-renderer experience;
- `?ui=lab` for advanced controls;
- `?film=1&t=4&w=640&h=360` and dimension limits;
- the exact `window.__blackhole` `state`, `renderAt`, and `frame` methods;
- reduced-motion startup behavior;
- the three local smoke URLs from Task 4;
- the unchanged statement that Python is authoritative and browser paths remain parity-gated.

In `web/README_SHOWCASE.md`, make the real cinematic WebGPU page the first recommended browser experience, retain the guided canvas as explicitly presentational, and add the Lab URL. Do not describe the presentation proxy as physics-equivalent.

- [ ] **Step 5: Run focused CI/documentation contracts**

Run:

```bash
node --test web/webgpu/tests/*.test.mjs
cd python
python -m pytest -q tests/test_webgpu_ui_contract.py tests/test_webgpu_progressive_assets.py tests/test_web_showcase_workflow.py
```

Expected: all PASS.

- [ ] **Step 6: Record the optional checkpoint**

Do not commit without explicit authorization. If authorization exists:

```bash
git add .github/workflows/ci.yml web/webgpu/README.md web/README_SHOWCASE.md python/tests/test_webgpu_ui_contract.py
git commit -m "ci: require cinematic WebGPU contracts"
```

---

### Task 6: Full verification and handoff evidence

**Files:**
- Verify only; do not create generated artifacts.

**Interfaces:**
- Consumes: Completed Tasks 1-5.
- Produces: Reproducible validation evidence and a clean, reviewable diff.

- [ ] **Step 1: Run JavaScript and focused browser-asset tests**

```bash
node --test web/webgpu/tests/*.test.mjs
cd python
python -m pytest -q tests/test_webgpu_ui_contract.py tests/test_webgpu_progressive_assets.py tests/test_webgpu_regression.py tests/test_web_showcase_workflow.py
```

Expected: all PASS.

- [ ] **Step 2: Run the repository Python gate**

From `python/`:

```bash
python -m compileall blackhole_sim
python -m pytest -q
```

Expected: compile succeeds and the full suite passes, allowing only already-documented optional skips.

- [ ] **Step 3: Run native formatting and unit gates**

From the repository root:

```bash
cargo fmt --check --manifest-path native/core/Cargo.toml
cargo test --manifest-path native/core/Cargo.toml
```

Expected: both PASS despite no native changes.

- [ ] **Step 4: Run the accelerator architecture gate**

From the repository root with the Python package available:

`python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`

Expected: PASS. If the package or optional native module is not installed, report the exact command and error rather than weakening the gate.

- [ ] **Step 5: Repeat the Task 4 browser smoke**

Verify Explore, Lab, film, deterministic repeated frame encoding, reduced-motion startup, state preservation while toggling Lab, and WebGPU-unavailable behavior. Do not save or track screenshots.

- [ ] **Step 6: Inspect the final diff and repository hygiene**

Run:

```bash
git diff --check
git status --short
git diff -- web/webgpu/index.html web/webgpu/src/main.js web/webgpu/src/ui_state.mjs web/webgpu/tests/ui_state.test.mjs python/tests/test_webgpu_ui_contract.py python/tests/test_webgpu_progressive_assets.py .github/workflows/ci.yml web/webgpu/README.md web/README_SHOWCASE.md
```

Expected: no whitespace errors, no generated images/binaries, no shader or physics diffs, and only planned files modified plus the design/plan documents if they are intentionally retained.

- [ ] **Step 7: Request review before any commit or publication**

Use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`. Report:

- exact tests run and outcomes;
- any unavailable command and why;
- browser/GPU environment used;
- confirmation that shaders and physics were unchanged;
- confirmation that no generated artifact is tracked;
- remaining parity limitations.

Do not commit, push, tag, or publish unless the human explicitly authorizes it.

## Copy-Paste Execution Prompt

```text
Implement the cinematic-first WebGPU experience in /workspace/blackhole-sim.

Before editing:
1. Read /workspace/blackhole-sim/AGENTS.md.
2. Read docs/superpowers/specs/2026-11-04-cinematic-webgpu-experience-design.md completely.
3. Read docs/superpowers/plans/2026-11-04-cinematic-webgpu-experience.md completely.
4. Use superpowers:using-git-worktrees to create an isolated worktree.
5. Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans, and use superpowers:test-driven-development for every behavior change.

Execute the plan task-by-task in order. Do not modify any WGSL shader or Python/native physics, do not add frontend dependencies or generated captures, and do not make new scientific-parity claims. Preserve all existing URL parameters and Lab controls. Run every focused and full verification command in the plan.

Repository policy forbids commits, pushes, tags, and publication unless I explicitly authorize them. Treat each plan checkpoint as uncommitted unless that authorization is present.

When implementation is complete, use superpowers:verification-before-completion and superpowers:requesting-code-review, then return the exact validation evidence, environment limitations, changed files, and remaining parity boundaries.
```
