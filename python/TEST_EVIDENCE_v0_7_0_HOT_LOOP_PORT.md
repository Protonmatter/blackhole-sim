# v0.7.0 hot-loop port test evidence

## Full test suite

Command:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/time -f 'elapsed=%E' python -m pytest -q --durations=60
```

Result:

```text
52 passed
elapsed=0:16.84
```

## Smoke render

Command:

```bash
python -m blackhole_sim.accelerated_cli \
  --width 4 \
  --height 3 \
  --max-steps 30 \
  --step 0.2 \
  --output out/accel_tiny.npz \
  --preview out/accel_tiny.png
```

Result:

```text
out/accel_tiny.npz
out/accel_tiny.png
```

## What is validated

- CPU-emulated coefficient-brick hot loop renders finite Stokes output.
- Trilinear coefficient-brick sampler returns finite 11-coefficient vectors.
- Progressive scheduler produces a final full-resolution frame.
- CPU reference and coefficient-brick renderer compare at small resolution.
- WGSL/CUDA/Metal/HIP/OpenCL kernel assets contain the required hot-loop components:
  - Kerr geodesic integration hooks,
  - trilinear coefficient sampling,
  - Stokes RK2 stepping,
  - `kerr_stokes_render_kernel` entry points.

## What is not validated in this runtime

- Vendor SDK compilation for CUDA, Metal, HIP, or OpenCL.
- Browser execution of WebGPU on an actual GPU.
- Full-HD runtime performance on real hardware.
