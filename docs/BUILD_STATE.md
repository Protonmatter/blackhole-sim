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
- `python -m pip install -e .[dev]`: passed after adding explicit setuptools package discovery for `blackhole_sim`.
- `blackhole-accelerators doctor --json --fail-on-emulation`: passed from PATH after installing the editable package and native wheel; `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.8.0`.
- `cargo fmt --check --manifest-path native/core/Cargo.toml`: passed using the repo-local `stable-aarch64-pc-windows-gnullvm` override.
- `cargo test --manifest-path native/core/Cargo.toml`: passed using the repo-local `stable-aarch64-pc-windows-gnullvm` override.
- `maturin build --manifest-path native/core/Cargo.toml --release --out native/core/target/wheels-local`: passed and produced `blackhole_native-0.8.0-cp310-abi3-win_arm64.whl`.

Blocked locally:

- MSVC-native local builds are still unavailable because Visual Studio Build Tools failed to install non-interactively with exit code `1602`.
- `gh auth status`: GitHub CLI is installed, but no GitHub host is authenticated yet.

## Local Toolchain Notes

- Installed GitHub CLI `2.95.0`.
- Installed Rustup and Rust `1.96.0`.
- Installed `maturin 1.14.1`.
- Installed `blackhole-sim` editable into the Python 3.14 ARM64 user environment.
- Added user PATH entries for Cargo, winget links, and Python 3.14 ARM64 scripts.
- Set a Rustup override for this checkout to `stable-aarch64-pc-windows-gnullvm`; `.cargo/config.toml` configures `rust-lld.exe` and static CRT flags for that target.
