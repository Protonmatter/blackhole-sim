# Cinematic-First WebGPU Experience Design

## Intent

Make the existing `web/webgpu/` renderer immediately compelling to a general
visitor without weakening its scientific boundaries. The existing WebGPU page
becomes cinematic-first: it opens in a minimal Explore experience, while the
current controls, presets, backend status, and telemetry remain one action away
in an in-page Lab mode.

The change also fixes the Stokes renderer's repeating coarse-to-fine resolution
cycle and adds a deterministic browser-frame interface for documentation,
visual review, and future regression automation.

## Goals

- Open `web/webgpu/index.html` in Explore mode unless `ui=lab` is present.
- Keep one renderer, one canvas, and one source of simulation state; switching
  UI modes must not reload or reset the simulation.
- Preserve every existing query parameter and advanced control.
- Give Explore mode a minimal HUD with title, renderer status, pause/run, reset,
  capture, and Lab access.
- Keep the advanced panel out of focus order while it is hidden.
- Respect reduced-motion preferences when the URL does not explicitly choose a
  running state.
- Make Stokes refinement restart after interaction or a size change, progress
  monotonically from coarse to full resolution, and remain at full resolution
  until the next invalidating change.
- Add deterministic film mode and a small browser automation API that can render
  an explicit simulation time at an explicit canvas size and export a frame.
- Add executable, dependency-free JavaScript tests for URL parsing and
  refinement state, while retaining the repository's existing Python asset
  checks.

## Non-Goals

- No shader, geodesic, radiative-transfer, coefficient, native-kernel, or Python
  physics changes.
- No new claim of WebGPU/native physics parity or scientific validation.
- No HDR/bloom post-processing, comparison mode, celestial texture work,
  tidal-disruption event, or WebGL2 physics fallback in this increment.
- No new framework, bundler, package manager dependency, or generated image
  committed to the repository.
- No split Explore and Lab applications and no iframe between them.

## Anti-Goals

- Do not replace the real WebGPU renderer with the existing 2D showcase proxy.
- Do not hide whether a view is an educational fixture, analytic volume, or
  Stokes coefficient-brick path.
- Do not let deterministic capture depend on wall-clock animation history.
- Do not make the Lab controls inaccessible by deleting, duplicating, or
  silently changing their semantics.

## Constraints

- `python/` remains the authoritative physics reference.
- WebGPU remains an acceleration and browser-presentation target subject to the
  repository's existing parity gates.
- Existing URLs using `shader`, `quality`, `running`, `time`, `spin`, `inc`,
  `camr`, `steps`, `density`, `temp`, `absorb`, `timeScale`, and `diagnostics`
  must continue to work.
- New URL parameters are `ui`, `film`, `t`, `w`, and `h`.
- `ui=lab` opens Lab; absent or invalid `ui` opens Explore. `film=1` overrides
  both and hides all UI.
- Film dimensions default to `1280x720` and clamp independently to width
  `64..1920` and height `64..1080`. Film time defaults to `0` and clamps to
  `0..100000` seconds so JavaScript intermediates and browser `f32` uniforms
  remain finite.
- Film mode renders at exact requested canvas dimensions, forces the final
  progressive level, does not advance simulation time from `requestAnimationFrame`,
  and sets `document.documentElement.dataset.blackholeReady = "true"` only
  after the submitted GPU work completes.
- Explore copy must use the phrase `educational browser renderer`; it must not
  use `validated`, `accurate`, or `physics-equivalent` as a claim.
- Follow `AGENTS.md`: deterministic and testable changes, no generated render
  output, and no commit/push/tag unless the human explicitly requests it.

## Selected Approach

### Single-page progressive enhancement

Keep `web/webgpu/index.html` and its canvas as the sole browser entry point.
Move the existing global toolbar into a compact HUD that is visible in Explore
and Lab. Convert the existing advanced control surface into an `aside` panel
whose visibility is controlled by `body[data-ui]`, `aria-hidden`, and `inert`.
The mode toggle updates the URL with `history.replaceState` and never recreates
the renderer.

Extract deterministic, DOM-independent state logic into
`web/webgpu/src/ui_state.mjs` and capture/device-lifecycle orchestration into
`web/webgpu/src/runtime_control.mjs`. The existing `main.js` continues to own
DOM, WebGPU resources, and the render loop. This gives Node's built-in test
runner small pure modules to exercise without introducing a frontend toolchain.

Centralize frame scheduling in `main.js`: each renderer factory returns one
`renderFrame(timeMs)` function instead of starting its own animation loop. The
central scheduler handles interactive animation, one-shot film rendering,
readiness, and the public capture API.

## Alternatives Considered

### Separate Explore page wrapping the current renderer

This would preserve the current page byte-for-byte, but it would duplicate URL
state, accessibility behavior, and capture logic or require an iframe message
protocol. It also perpetuates the current separation between presentation and
the real renderer. Rejected.

### Shared renderer library with two independent entry points

This is the cleanest long-term separation, but it would require a broad WebGPU
resource-lifecycle refactor before the current parity milestone is complete.
The selected approach creates the necessary central render seam without paying
the cost of two applications now. Deferred.

## User Experience

### Explore mode

- Full-screen real WebGPU canvas.
- Compact title and status: `BlackHole Sim`, `Interactive Kerr spacetime`, and
  the existing backend/mode status.
- Compact actions: Pause/Run, Reset, Capture, and Open Lab.
- A short visible boundary label: `Educational browser renderer`.
- Canvas is keyboard focusable and has an accessible instruction describing
  pointer drag, wheel distance, Space pause, and R reset.

### Lab mode

- The same canvas, simulation state, toolbar, and running renderer.
- Existing shader, quality, spin, inclination, camera, step, density,
  temperature, absorption, time, preset, and telemetry controls.
- Close Lab returns to Explore without resetting parameters or time.
- Hidden Lab content is both visually hidden and excluded from sequential focus;
  the initial HTML carries both `aria-hidden="true"` and `inert` so this is true
  before JavaScript initializes.

### Film mode

- `?film=1&t=4&w=640&h=360` renders one deterministic frame with no HUD.
- Existing renderer and simulation parameters may be supplied in the same URL.
- `window.__blackhole` exposes:
  - `state(): object` returning serializable UI, simulation, resolution, and
    readiness state;
  - `renderAt(seconds: number): Promise<object>` pausing wall-clock animation,
    rendering full refinement at the requested time clamped to `0..100000`
    seconds, waiting for submitted GPU work, keeping full refinement selected
    until the next user interaction, marking readiness, and returning `state()`;
  - `frame(type = "image/png", quality = 0.95): Promise<string>` returning the
    encoded frame bytes as a base64 string without a data-URL prefix.
- `renderAt` and `frame` are exclusive deterministic operations; a caller that
  overlaps either operation receives a clear rejection rather than racing the
  shared simulation state or presentation texture.
- Initialization failures set
  `document.documentElement.dataset.blackholeError` to a concise message and do
  not mark the page ready.
- Device loss and uncaptured GPU errors after initialization clear readiness,
  stop interactive scheduling, expose `data-blackhole-error`, and make public
  rendering/capture calls reject until the page is reloaded.

## Progressive Refinement Contract

`ui_state.mjs` owns a pure refinement reducer. It receives the previous state
and a sample containing `nowMs`, `interactionVersion`, `width`, `height`,
`levelCount`, and `forceFull`.

- A new interaction version or size resets to level index `0` and records the
  current time as the stable start.
- With no invalidation, the selected index is
  `max(previous.levelIndex, min(floor((nowMs - stableSinceMs) / 850),
  levelCount - 1))`.
- The index never decreases without invalidation and never wraps back to zero.
- `forceFull` selects `levelCount - 1` immediately.
- Empty/invalid level counts safely select index `0`; the caller must still
  provide at least one concrete render level.

The Stokes render path uses this reducer. Fragment disk and volume paths retain
their existing quality scaling and step bounds.

## Test Strategy

- Node built-in tests cover default Explore parsing, explicit Lab parsing, film
  precedence, invalid/clamped film parameters, refinement invalidation,
  monotonic advancement, final-level hold, and force-full behavior.
- Existing Python asset tests are updated to assert the cinematic-first markup,
  accessible hidden panel contract, deterministic API hooks, central scheduler,
  and absence of the old modulo refinement cycle.
- CI adds a dependency-free Node 22 web job and includes it in `ci-required`.
- Local browser smoke covers Explore, Lab state preservation, WebGPU unavailable
  messaging, reduced-motion startup, film readiness, deterministic repeated
  `renderAt`, and non-empty frame export. Generated captures remain untracked.

## Documentation

Update `web/webgpu/README.md` with UI modes, URL parameters, the public browser
API, exact local smoke URLs, and the explicit educational/validation boundary.
Update `web/README_SHOWCASE.md` so it points users to the cinematic real renderer
first and documents `?ui=lab` for the advanced surface.

## Decisions Left to the Implementer

- Exact colors, spacing, and responsive breakpoints, provided the existing dark
  visual language and focus visibility are preserved.
- Whether the Lab panel enters from the left or appears without motion; reduced
  motion must always disable nonessential transitions.
- Internal names for renderer-factory closures that are not part of the public
  `window.__blackhole` API.
