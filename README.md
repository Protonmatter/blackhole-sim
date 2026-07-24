# BlackHole Sim

Kerr GRMHD / GRRT black-hole simulator with polarized transfer, public dump validation, WebGPU/native kernel assets, and v0.9 native hot-loop parity groundwork.

This repository is intentionally split into portable orchestration, native hot-loop groundwork, and browser showcase assets:

```text
python/       Python package, CLIs, tests, examples, public dump tooling
native/       CUDA/Metal/HIP/OpenCL kernel assets plus Rust/PyO3 core scaffold
web/          WebGPU renderer and guided browser showcase
docs/         build state, native roadmap, data validation, performance notes
```

## Physics Contract

Python is the authoritative physics reference path. The repo models Kerr
spacetime, local ZAMO camera rays, null geodesics, GRMHD snapshot sampling,
invariant redshift hooks, and polarized Stokes transfer, but it is still a
research reference implementation. WebGPU and native kernels are acceleration
targets and must pass deterministic parity gates before any physics-equivalence
claim is made.

The unpolarized preview path defaults to `educational_proxy` mode. `validated`
mode requires metric-aware local plasma calculations plus a non-proxy
coefficient model or external baseline evidence.

See `docs/PHYSICS_VALIDATION.md` for the current grounding, approximation
boundaries, validation commands, and the WebGPU-versus-game-engine policy. See
`docs/SCIENTIFIC_ROADMAP.md` for the release gates that turn the project into a
validated GRRT platform.

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

## Direct GPU Path

The browser renderer uses WebGPU compute and storage buffers directly through
the platform GPU backend exposed by the browser, such as D3D12 on Windows,
Metal on macOS, or Vulkan where supported. On this Windows ARM64 checkout,
`blackhole-accelerators list --json` reports the OS video adapter when Windows
exposes it through `Win32_VideoController`.

Append `&diagnostics=1` to the WebGPU URL to run a bounded GPU readback sample
for local adapter diagnostics.

The CUDA, Metal, HIP, OpenCL, and WGSL kernel assets share the same staged
contract: invalid nonperiodic brick samples are guarded before Stokes transfer
updates, and no GPU physics parity claim is made until a small deterministic
render passes against the CPU reference envelope on the target hardware. The
Stokes kernel source now derives camera rays from a ZAMO launch helper rather
than fixed photon momenta, but readback parity is still required before claiming
GPU renderer parity.

## Showcase

Run the guided showcase from the repo root:

```bash
cd web
python -m http.server 8080
```

Open `http://localhost:8080/showcase/`.

## License

No license has been added. Do not redistribute or publish packaged artifacts under an assumed license.
