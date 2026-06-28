# Interactive Kerr Black-Hole Simulation Showcase

This package wraps the v0.7 simulator in a guided browser session.

## Start the session

From this folder:

```bash
python -m http.server 8080
```

Open:

```text
http://localhost:8080/showcase/
```

## What the showcase provides

- Guided controls for Kerr spin, inclination, camera distance, ray steps, density, magnetic-field scale, Faraday rotation, and Faraday conversion.
- Hardware profiles:
  - Surface Snapdragon
  - Surface Intel Core Ultra 7 268V
  - desktop preview
  - workstation CUDA
- A fast canvas fallback session that opens on any browser.
- A link into the actual v0.7 WebGPU Stokes renderer:

```text
http://localhost:8080/simulator/webgpu/index.html?shader=stokes
```

## Important boundary

The guided canvas is a presentation/demo layer. The actual hot-loop implementation is in:

```text
simulator/webgpu/src/stokes_brick_compute.wgsl
simulator/native/cuda/kerr_stokes_kernel.cu
simulator/native/metal/kerr_stokes_kernel.metal
simulator/native/rocm/kerr_stokes_kernel.hip
simulator/native/opencl/kerr_stokes_kernel.cl
```

Use the guided session to explain the physics and performance tradeoffs. Use the WebGPU renderer and Python CLIs for implementation validation.

## CLI smoke tests

From `simulator/`:

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
