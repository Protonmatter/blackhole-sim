# BlackHole Sim

Kerr GRMHD / GRRT black-hole simulator with polarized transfer, public dump validation, WebGPU/native kernel assets, and v0.9 native hot-loop parity groundwork.

This repository is intentionally split into portable orchestration, native hot-loop groundwork, and browser showcase assets:

```text
python/       Python package, CLIs, tests, examples, public dump tooling
native/       CUDA/Metal/HIP/OpenCL kernel assets plus Rust/PyO3 core scaffold
web/          WebGPU renderer and guided browser showcase
docs/         build state, native roadmap, data validation, performance notes
```

## Install

```bash
cd python
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
```

On Linux/macOS, activate with `source .venv/bin/activate`.

## Core Checks

```bash
cd python
python -m compileall blackhole_sim
python -m pytest -q
blackhole-accelerators list --json
blackhole-accelerators doctor --json --fail-on-emulation
blackhole-render-accelerated --width 32 --height 18 --max-steps 64 --output out/stokes_smoke.npz
blackhole-benchmark --json --nr 8 --ntheta 6 --nphi 8 --points 64 --iterations 1
```

Rust scaffold checks:

```bash
cargo fmt --check --manifest-path native/core/Cargo.toml
cargo test --manifest-path native/core/Cargo.toml
maturin build --manifest-path native/core/Cargo.toml --release
```

## Native Boundary

The v0.9 milestone keeps Python as the reference path while proving small native CPU parity targets behind regression gates:

- `blackhole_native.stokes_rk2_brick()`
- `blackhole_native.sample_brick_trilinear()`
- `blackhole_native.sample_and_step_stokes()`

Python exposes `sample_brick_valid_mask(...)` plus `sample_and_step_stokes(..., invalid_policy=...)` so downstream renderers can handle nonperiodic out-of-domain samples deterministically. These are not full renderer parity or GPU physics parity claims. Native wheels still must pass architecture and parity gates on the target platform before any performance claim is made.

The release gate starts with:

```bash
blackhole-accelerators doctor --json --fail-on-emulation
```

## Showcase

Run the guided showcase from the repo root:

```bash
cd web
python -m http.server 8080
```

Open `http://localhost:8080/showcase/`.

## License

No license has been added. Do not redistribute or publish packaged artifacts under an assumed license.
