# BlackHole Sim

Kerr GRMHD / GRRT black-hole simulator with polarized transfer, public dump validation, WebGPU/native kernel assets, and a v0.8 native-foundation scaffold.

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
blackhole-benchmark --json --nr 8 --ntheta 6 --nphi 8 --iterations 1
```

Rust scaffold checks:

```bash
cargo fmt --check --manifest-path native/core/Cargo.toml
cargo test --manifest-path native/core/Cargo.toml
maturin build --manifest-path native/core/Cargo.toml --release
```

## Native Boundary

The v0.8 milestone does not claim full compiled native physics parity. It adds the architecture probe and native module scaffold needed to prove future wheels are native for Windows ARM64, Windows x86_64, macOS arm64/x86_64, Linux x86_64, and Linux aarch64.

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
