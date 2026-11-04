# Interactive Kerr black-hole showcase

This package provides two browser experiences around the simulator assets.

## Start locally

From `web/`:

```bash
python -m http.server 8080
```

## Recommended: cinematic WebGPU renderer

Start with the real WebGPU renderer in its cinematic Explore interface:

```text
http://localhost:8080/webgpu/
```

Open the advanced Lab controls without changing canvas or renderer state:

```text
http://localhost:8080/webgpu/?ui=lab
```

The WebGPU path includes Kerr thin-disk, GRRT-volume, and Stokes-brick modes.
It requires a WebGPU-capable browser served from localhost.

## Guided presentation canvas

The guided session remains available at:

```text
http://localhost:8080/showcase/
```

It provides narrated controls, hardware profiles, and a broadly compatible
canvas fallback for demonstrations. This guided canvas is explicitly a
presentational layer; use the WebGPU renderer, Python CLIs, regression tests,
and native gates for implementation validation.

The simulation hot loops live in:

```text
web/webgpu/src/stokes_brick_compute.wgsl
native/cuda/kerr_stokes_kernel.cu
native/metal/kerr_stokes_kernel.metal
native/rocm/kerr_stokes_kernel.hip
native/opencl/kerr_stokes_kernel.cl
```

## CLI smoke tests

From `python/`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest
```

Render a small accelerated preview:

```bash
blackhole-render-accelerated --width 64 --height 36 --max-steps 256 --preview out/demo.png --output out/demo.npz
```
