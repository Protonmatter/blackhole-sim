# Build State

Date: 2026-06-28

## Current Milestone

`v0.8.0 Native Foundation`

## Source Baseline

- Imported from `blackhole_interactive_showcase_v0_7.zip`.
- Used `blackhole_sim_project_v0_7_hot_loop_kernels.zip` as the authoritative source for `python/blackhole_sim/__init__.py` because it exports the accelerated renderer symbols.
- Generated sample outputs from the archive were excluded from the repository.

## Implemented

- Unified repository layout under `python/`, `native/`, `web/`, and `docs/`.
- Optional accelerator detection no longer fails when optional parent packages such as `numba` are absent.
- HDF5 tests skip when optional `h5py` is unavailable.
- Added `blackhole-accelerators doctor`.
- Added Python platform probe and native loader.
- Added Rust/PyO3 `blackhole_native` scaffold.
- Added CI workflows for Python, Rust, native wheel smoke builds, and architecture reports.

## Release Boundary

This milestone does not publish PyPI packages, signed apps, notarized macOS builds, or full native physics parity claims.

## Required Validation

```bash
cd python
python -m compileall blackhole_sim
python -m pytest -q
blackhole-accelerators list --json
blackhole-accelerators doctor --json --fail-on-emulation
blackhole-render-accelerated --width 32 --height 18 --max-steps 64 --output out/stokes_smoke.npz
cd ..
cargo fmt --check --manifest-path native/core/Cargo.toml
cargo test --manifest-path native/core/Cargo.toml
maturin build --manifest-path native/core/Cargo.toml --release
```

## Local Validation Evidence

Run on Windows 11 ARM64 with Python 3.14.3 ARM64:

- `python -m compileall blackhole_sim`: passed.
- `python -m pytest -q`: passed with optional HDF5/browser-dependent skips.
- `python -m blackhole_sim.accelerator_cli list --json`: passed; WebGPU shader assets detected, ARM SIMD fallback detected.
- `python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`: passed; `process_arch=arm64`, `python_arch=arm64`, `emulation_detected=false`, `native_core_loaded=false`.
- `python -m blackhole_sim.accelerated_cli --width 32 --height 18 --max-steps 64 --output out/stokes_smoke.npz`: passed; output is ignored and not committed.

Blocked locally:

- `cargo fmt --check --manifest-path native/core/Cargo.toml`: `cargo` is not installed or not on PATH.
- `cargo test --manifest-path native/core/Cargo.toml`: `cargo` is not installed or not on PATH.
- `maturin build --manifest-path native/core/Cargo.toml --release`: `maturin` is not installed in the active Python environment.
