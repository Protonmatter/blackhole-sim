# v0.6.0 accelerator suite evidence

Command:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/time -f 'elapsed=%E' python -m pytest -q --durations=20
```

Result:

```text
47 passed
elapsed=0:22.85
CODE=0
```

New test coverage:

- accelerator backend detection and full-HD planning
- coefficient-brick memory estimation and NPZ roundtrip
- coefficient-brick finite-value generation
- tiled and progressive render scheduling
- native kernel asset presence for CUDA, Metal, OpenCL, ROCm, and WebGPU compute

Notes:

- The CPU renderer is still the correctness path.
- The native GPU files define stable launch/buffer contracts in v0.6.0; kernel feature parity is the next implementation step.
- OpenVINO is treated as an inference/surrogate backend rather than a general Kerr geodesic kernel runtime.
