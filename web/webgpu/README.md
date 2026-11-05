# Cinematic WebGPU Kerr renderer

The WebGPU entry point opens in **Explore** mode: a compact cinematic interface
over the real browser renderer. It keeps the same Kerr thin-disk, GRRT-volume,
and Stokes-brick render paths while moving advanced controls into an in-place
Lab panel.

The Python implementation remains authoritative. Browser render paths are
educational ports and stay subject to the repository's parity and regression
gates; they do not replace Python validation or native-kernel verification.

## Run locally

From `web/`:

```bash
python -m http.server 8080
```

Use these smoke-test URLs:

```text
http://localhost:8080/webgpu/
http://localhost:8080/webgpu/?ui=lab
http://localhost:8080/webgpu/?shader=stokes&quality=fast&steps=80&film=1&t=4&w=640&h=360
```

- The first URL opens Explore with the live renderer and compact controls.
- `?ui=lab` opens the same canvas and state with advanced controls visible.
- `film=1` hides the interface, renders one deterministic frame, and takes
  precedence over `ui=lab`.

Film mode accepts `t` from 0–100000 seconds, `w` from 64–1920 pixels, and `h`
from 64–1080 pixels. Absent or nonnumeric values use `t=0`, `w=1280`, and
`h=720`; numeric values are clamped to their limits and dimensions are
truncated to integers. Existing `shader`, `quality`, `running`, camera,
transfer, step, and diagnostics query parameters continue to work.

## Interaction and motion

Drag horizontally to adjust spin and vertically to adjust inclination, use the
wheel to change camera radius, press Space to pause or run, and press R to
reset. The Lab panel exposes shader, quality, physical controls, presets, and
telemetry without loading a second renderer.

When the operating system requests reduced motion and the URL does not include
an explicit `running=0` or `running=1`, simulation time starts paused. Rendering
and progressive refinement still occur so a crisp static view can settle.

## Deterministic browser API

After renderer creation, `window.__blackhole` exposes:

- `state()` — returns a fresh serializable snapshot with UI mode, film and
  readiness flags, shader, quality, run state, simulation time, resolution, and
  interaction version.
- `await renderAt(seconds)` — pauses time, renders the requested time clamped to
  0–100000 seconds at full refinement, waits for GPU completion, and returns
  the state snapshot.
- `await frame(type = 'image/png', quality = 0.95)` — submits a fresh frame,
  begins the canvas snapshot before the presentation buffer can be recycled,
  waits for GPU completion, and returns base64-encoded image bytes without a
  data-URL prefix.

`renderAt` and `frame` are exclusive operations. If one is already running, an
overlapping call rejects with a busy error instead of racing the shared canvas
or simulation state; retry after the first promise settles.

Automation can wait for
`document.documentElement.dataset.blackholeReady === 'true'`. Startup failures,
device loss, and uncaptured GPU errors set `data-blackhole-error`, leave the
ready marker absent, stop interactive rendering, and make later API work reject
until the page is reloaded.

Example:

```js
const state = await window.__blackhole.renderAt(4);
const pngBase64 = await window.__blackhole.frame();
console.log(state.resolution, pngBase64.length);
```

## Scientific boundary

The volume shader currently uses an analytic fixture and has no external
dataset dependency. Python is authoritative for HDF5/NPZ GRMHD snapshot
ingestion. A production browser path should upload validated snapshot bricks
into WebGPU storage buffers or 3D textures and pass the same parity gates before
being used for scientific conclusions.
