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

Film mode accepts `t` in non-negative seconds, `w` from 64–1920 pixels, and
`h` from 64–1080 pixels. Absent or nonnumeric values use `t=0`, `w=1280`,
and `h=720`; numeric dimensions are truncated to integers and clamped to their
limits. Existing `shader`, `quality`, `running`, camera, transfer, step, and
diagnostics query parameters continue to work.

## Interaction and motion

Drag to orbit, use the wheel to change camera radius, press Space to pause or
run, and press R to reset. The Lab panel exposes shader, quality, physical
controls, presets, and telemetry without loading a second renderer.

When the operating system requests reduced motion and the URL does not include
an explicit `running=0` or `running=1`, simulation time starts paused. Rendering
and progressive refinement still occur so a crisp static view can settle.

## Deterministic browser API

After renderer creation, `window.__blackhole` exposes:

- `state()` — returns a fresh serializable snapshot with UI mode, film and
  readiness flags, shader, quality, run state, simulation time, resolution, and
  interaction version.
- `await renderAt(seconds)` — pauses time, renders the requested non-negative
  simulation time at full refinement, waits for GPU completion, and returns the
  state snapshot.
- `await frame(type = 'image/png', quality = 0.95)` — waits for submitted GPU
  work and returns base64-encoded image bytes without a data-URL prefix.

Automation can wait for
`document.documentElement.dataset.blackholeReady === 'true'`. Startup failures
set `data-blackhole-error` and leave the ready marker absent.

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
