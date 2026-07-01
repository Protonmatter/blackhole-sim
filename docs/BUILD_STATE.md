# Build State

Date: 2026-07-01

## Current Milestone

`v0.9.0 Native Hot-Loop Parity` in progress

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
- Hardened PyO3 feature selection so plain `cargo test` does not build the extension-module flavor on macOS.
- Native wheel CI now installs the built wheel and runs the architecture doctor gate.
- Added public-data selected-dump evidence gates and deterministic hot-loop benchmark probes.
- Updated Intel macOS CI coverage to use the current `macos-15-intel` runner label.
- Added the first native CPU hot-loop parity target: Rust `blackhole_native.stokes_rk2_brick()` for RK2 Stokes stepping over coefficient-brick cells, with Python reference fallback.
- Extended `blackhole-benchmark --json` to report Python reference and native timing for the Stokes RK2 brick workload.
- Added Rust `blackhole_native.sample_brick_trilinear()` for deterministic trilinear coefficient-brick sampling, with Python reference fallback.
- Added Rust `blackhole_native.sample_and_step_stokes()` as the first composed sampler-plus-RK2 micro-kernel.
- Out-of-range nonperiodic sampler points now return explicit `NaN` coefficient vectors rather than valid-looking zero vectors; periodic `phi` wrapping remains over a `2*pi` domain.
- Added Python `sample_brick_valid_mask(...)` and `sample_and_step_stokes(..., invalid_policy=...)` so downstream renderer integration can keep, replace with initial Stokes values, or zero invalid nonperiodic rows intentionally.
- Extended `blackhole-benchmark --json` to report Python reference and native timing for the coefficient sampler workload, including backend, architecture, grid size, point count, and parity error.
- Native wheel CI now runs sampler parity tests after wheel install, alongside Stokes RK2 parity.
- WebGPU accelerator discovery now reports OS-visible GPU adapters on Windows so direct browser GPU execution can be audited alongside shader availability.
- The WebGPU browser renderer now reports the granted adapter and labels compute/fragment paths explicitly.
- CUDA, Metal, HIP, OpenCL, and WebGPU Stokes kernel assets now guard invalid nonperiodic coefficient-brick samples before applying Stokes transfer updates.
- Added a science-grade validation direction: explicit `educational_proxy` versus `validated` transfer mode, metric-aware local plasma magnetic-field and pitch-angle helpers, and a release-gate roadmap for GRRT validation.
- Coefficient-brick precomputation and polarized transfer now use the shared metric-aware local plasma helper when snapshot spin is available.
- WebGPU Stokes compute and static CUDA/Metal/HIP/OpenCL kernel assets now derive launch state and conserved photon momenta through a ZAMO camera helper instead of fixed `p_t`/`p_phi` constants.
- Installed seven local Codex science skills under `C:\Users\mkang\.codex\skills` for Kerr validation, GRMHD data validation, GRRT transfer, numerical methods, GPU parity, scientific Python quality, and scientific visualization review.
- Replaced separate Python/native/architecture smoke workflows with a unified CI pipeline plus a manual/tag artifact-delivery workflow and Dependabot update configuration. The artifact workflow uploads review artifacts only; it does not publish packages or releases.

## Release Boundary

This milestone does not publish PyPI packages, signed apps, notarized macOS builds, GPU physics parity claims, or full native renderer parity claims.

## Required Validation

```bash
cd python
python -m compileall blackhole_sim
python -m pytest -q
blackhole-accelerators list --json
blackhole-accelerators doctor --json --fail-on-emulation
blackhole-benchmark --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1
blackhole-render-accelerated --width 32 --height 18 --max-steps 64 --output out/stokes_smoke.npz
cd ..
cargo fmt --check --manifest-path native/core/Cargo.toml
cargo test --manifest-path native/core/Cargo.toml
maturin build --manifest-path native/core/Cargo.toml --release
python -m pytest -q python/tests/test_native_stokes_parity.py python/tests/test_native_sampler_parity.py
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
- `cargo +stable-aarch64-pc-windows-msvc test --manifest-path native/core/Cargo.toml`: passed inside the VS 2022 ARM64 developer environment.
- `maturin build --manifest-path native/core/Cargo.toml --release --target aarch64-pc-windows-msvc --out native/core/target/wheels-msvc`: passed and produced `blackhole_native-0.8.0-cp310-abi3-win_arm64.whl`.
- `blackhole-render-accelerated --width 32 --height 18 --max-steps 64 --output out/stokes_review_smoke.npz`: passed; output is ignored and not committed.

v0.9.0 local parity evidence:

- `python -m pip install -e .\python`: passed and installed `blackhole-sim 0.9.0`.
- `maturin build --manifest-path native/core/Cargo.toml --release`: passed and produced `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`.
- `python -m pip install --force-reinstall native/core/target/wheels-v090/blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`: passed; `blackhole_native.core_version()` returned `0.9.0` and `stokes_rk2_brick` is present.
- `python -m pytest -q tests/test_native_stokes_parity.py tests/test_benchmark.py`: passed; native parity test executed with the installed wheel.
- `python -m pytest -q`: passed with optional skips.
- `python -m blackhole_sim.benchmark_cli --json --nr 3 --ntheta 3 --nphi 3 --iterations 1`: passed; schema is `blackhole_sim.benchmark.v2`, native Stokes RK2 parity reported `allclose=true`, `max_abs_diff=0.0`, and `native_available=true`.
- `python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`: passed; `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, and `emulation_detected=false`.
- `blackhole-render-accelerated --width 32 --height 18 --max-steps 64 --output out/stokes_v090_smoke.npz`: passed; output is ignored and not committed.

v0.9.0 sampler/composed micro-kernel local evidence:

- `python -m compileall blackhole_sim`: passed.
- `cargo fmt --check --manifest-path native/core/Cargo.toml`: passed.
- `cargo test --manifest-path native/core/Cargo.toml`: passed; 6 Rust unit tests passed.
- `maturin build --manifest-path native/core/Cargo.toml --release --out native/core/target/wheels-v090-sampler`: passed and produced `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`.
- `python -m pip install --force-reinstall native\core\target\wheels-v090-sampler\blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`: passed.
- `python -m pytest -q tests/test_native_sampler_parity.py tests/test_native_stokes_parity.py tests/test_benchmark.py`: passed with installed native sampler and sample-and-step functions.
- `python -m pytest -q`: passed with optional skips.
- `python -m blackhole_sim.benchmark_cli --target sample-brick-trilinear --json --nr 4 --ntheta 3 --nphi 5 --points 16 --iterations 2`: passed; native sampler parity reported `allclose=true`, `max_abs_diff=0.0`, `max_rel_diff=0.0`, `native_available=true`, and `process_arch=arm64`.
- `python -m blackhole_sim.benchmark_cli --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1`: passed; schema is `blackhole_sim.benchmark.v2` and includes sampler plus Stokes native parity payloads.
- `python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`: passed; `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, and `emulation_detected=false`.
- `python -m blackhole_sim.accelerated_cli --width 32 --height 18 --max-steps 64 --output out\stokes_sampler_smoke.npz`: passed; output is ignored and not committed.

v0.9.0 sampler invalid-domain fix local evidence:

- `python -m compileall blackhole_sim`: passed.
- `cargo fmt --check --manifest-path native/core/Cargo.toml`: passed.
- `cargo test --manifest-path native/core/Cargo.toml`: passed; Rust tests now require explicit `NaN` output for out-of-range nonperiodic sampler points.
- `maturin build --manifest-path native/core/Cargo.toml --release --out native/core/target/wheels-v090-sampler-nan`: passed and produced `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`.
- `python -m pip install --force-reinstall native\core\target\wheels-v090-sampler-nan\blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`: passed.
- `python -m pytest -q tests/test_native_sampler_parity.py tests/test_native_stokes_parity.py tests/test_benchmark.py`: passed with installed native sampler and sample-and-step functions.
- `python -m pytest -q`: passed with optional skips.
- `python -m blackhole_sim.benchmark_cli --target sample-brick-trilinear --json --nr 4 --ntheta 3 --nphi 5 --points 16 --iterations 2`: passed; native sampler parity reported `allclose=true`, `max_abs_diff=0.0`, `max_rel_diff=0.0`, `native_available=true`, and `process_arch=arm64`.
- `python -m blackhole_sim.benchmark_cli --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1`: passed; schema is `blackhole_sim.benchmark.v2` and includes sampler plus Stokes native parity payloads.
- `python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`: passed; `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, and `emulation_detected=false`.
- `python -m blackhole_sim.accelerated_cli --width 32 --height 18 --max-steps 64 --output out\stokes_sampler_nan_smoke.npz`: passed; output is ignored and not committed.

v0.9.0 sampler invalid-row policy local evidence:

- `python -m compileall blackhole_sim`: passed.
- `cargo fmt --check --manifest-path native/core/Cargo.toml`: passed.
- `cargo test --manifest-path native/core/Cargo.toml`: passed.
- `python -m pytest -q tests/test_native_sampler_parity.py tests/test_native_stokes_parity.py tests/test_benchmark.py`: passed; sampler tests cover `sample_brick_valid_mask(...)`, default `nan`, `initial`, and `zero` invalid-sample policies.
- `maturin build --manifest-path native/core/Cargo.toml --release --out native/core/target/wheels-v090-invalid-policy`: passed and produced `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`.
- `python -m pip install --force-reinstall native\core\target\wheels-v090-invalid-policy\blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`: passed.
- `python -m pytest -q`: passed with optional skips.
- `python -m blackhole_sim.benchmark_cli --target sample-brick-trilinear --json --nr 4 --ntheta 3 --nphi 5 --points 16 --iterations 2`: passed; native sampler parity reported `allclose=true`, `max_abs_diff=0.0`, `max_rel_diff=0.0`, `native_available=true`, and `process_arch=arm64`.
- `python -m blackhole_sim.benchmark_cli --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1`: passed; schema is `blackhole_sim.benchmark.v2` and includes sampler plus Stokes native parity payloads.
- `python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`: passed; `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, and `emulation_detected=false`.
- `python -m blackhole_sim.accelerated_cli --width 32 --height 18 --max-steps 64 --output out\stokes_invalid_policy_smoke.npz`: passed; output is ignored and not committed.

v0.9.0 direct GPU contract local evidence:

- `python -m compileall blackhole_sim`: passed.
- `python -m pytest -q`: passed with optional skips.
- `cargo fmt --check --manifest-path native/core/Cargo.toml`: passed.
- `cargo test --manifest-path native/core/Cargo.toml`: passed.
- `python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`: passed; `gpu_backend=webgpu`, `native_core_loaded=true`, `native_core_arch=arm64`, and `emulation_detected=false`.
- `python -m blackhole_sim.accelerator_cli list --json`: passed; WebGPU reported `Qualcomm(R) Adreno(TM) X1-85 GPU` from Windows video-controller discovery.
- `python -m blackhole_sim.benchmark_cli --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1`: passed; native sampler and Stokes RK2 parity both reported `allclose=true`, `max_abs_diff=0.0`, and `max_rel_diff=0.0`.
- `python -m blackhole_sim.accelerated_cli --width 32 --height 18 --max-steps 64 --output out\stokes_gpu_contract_smoke.npz`: passed; output is ignored and not committed.
- `maturin build --manifest-path native/core/Cargo.toml --release --out native/core/target/wheels-gpu-contract`: passed and produced `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl`.
- In-app browser smoke against `http://127.0.0.1:8800/index.html?shader=stokes`: loaded the default WebGPU page, status reported `WebGPU direct compute: Stokes coefficient bricks on qualcomm / adreno-7xx`, and console warnings/errors were empty. Follow-up automation for the opt-in `diagnostics=1` readback path timed out before DOM readback, so GPU readback verification is not claimed here.

v0.9.0 science-grade redirection local evidence:

- `python -m pytest -q tests/test_synchrotron_polarized_transfer.py tests/test_radiative_transfer.py tests/test_native_kernel_assets.py tests/test_kerr.py`: passed.
- Local Codex skill validation with `quick_validate.py`: passed for `blackhole-gr-kerr-validation`, `blackhole-grmhd-data-validation`, `blackhole-grrt-polarized-transfer`, `blackhole-numerical-methods`, `blackhole-hpc-gpu-parity`, `scientific-python-quality`, and `scientific-visualization-review`.
- `python -m compileall blackhole_sim`: passed.
- `python -m pytest -q`: passed with 3 optional skips.
- `cargo fmt --check --manifest-path native/core/Cargo.toml`: passed.
- `cargo test --manifest-path native/core/Cargo.toml`: passed; 6 Rust unit tests passed.
- `python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation`: passed; `process_arch=arm64`, `python_arch=arm64`, `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, `gpu_backend=webgpu`, and `emulation_detected=false`.
- `python -m blackhole_sim.benchmark_cli --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1`: passed; native sampler and Stokes RK2 parity both reported `allclose=true`, `max_abs_diff=0.0`, and `max_rel_diff=0.0`.
- `python -m blackhole_sim.accelerated_cli --width 32 --height 18 --max-steps 64 --output out\science_redirection_smoke.npz`: passed; output is ignored and not committed.
- Direct WebGPU browser readback and concrete Illinois v3 selected-dump/ipole validation were not run in this pass, so GPU image parity and external GRRT parity are still not claimed.

v0.9.0 CI/CD pipeline local evidence:

- `.github/workflows/ci.yml` added as the pull-request and `main` validation pipeline, including hygiene, Python, science-contract, Rust, native-wheel, and final required aggregation jobs.
- `.github/workflows/release-artifacts.yml` added for manual or `v*` tag artifact builds; it uploads artifacts but intentionally does not publish PyPI packages or GitHub releases.
- `.github/dependabot.yml` added for weekly GitHub Actions, Python, and Cargo update PRs.
- Local workflow YAML parsing with Python/PyYAML: passed for `.github/workflows/ci.yml`, `.github/workflows/release-artifacts.yml`, and `.github/dependabot.yml`.
- Tracked-artifact policy check using `git ls-files`: passed; no tracked `.npz`, `.h5`, `.hdf5`, `.png`, `.tiff`, or `.whl` artifacts.
- `git diff --check`: passed; Windows line-ending warnings are non-blocking.
- `actionlint` was not available on PATH, so GitHub Actions execution is the next validation layer after push.

## GitHub CI Evidence

Validated on commit `46f20630f41556dce75cb6142a54ff816185c54a`:

- `Python` run `28345975404`: passed.
- `Native Core` run `28345975402`: passed.
- `Architecture Report` run `28345986411`: passed.
- CI artifact smoke: downloaded `native-wheel-windows-11-arm` from run `28345975402`, installed `blackhole_native-0.8.0-cp310-abi3-win_arm64.whl` into a clean temporary venv, and ran `blackhole-accelerators doctor --json --fail-on-emulation`; passed with `native_core_loaded=true`, `native_core_arch=arm64`, and `emulation_detected=false`.
- Earlier queued `macos-13` runs were cancelled after replacing that retired label with `macos-15-intel`.
- Architecture Report is now configured as an automatic `push` gate on `main`, matching the Python and Native Core workflows.

v0.9.0 parity target validated on commit `215769ef09682f423f02a0c8be51ebc40736e47f`:

- `Architecture Report` run `28365345475`: passed.
- `Python` run `28365345461`: passed.
- `Native Core` run `28365345460`: passed, including post-wheel `python/tests/test_native_stokes_parity.py`.
- CI artifact smoke: downloaded `native-wheel-windows-11-arm` from run `28365345460`, installed `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl` into a clean temporary venv, and ran `blackhole-accelerators doctor --json --fail-on-emulation`; passed with `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, and `emulation_detected=false`.
- CI artifact benchmark smoke: `blackhole-benchmark --target stokes-rk2-brick --json --nr 3 --ntheta 3 --nphi 3 --iterations 1` passed with `native_available=true`, `allclose=true`, `max_abs_diff=0.0`, and `max_rel_diff=0.0`.

v0.9.0 sampler/composed micro-kernel validated on commit `5fae11d5968c1a8097da7eb9724b2d6c71fdac94`:

- `Architecture Report` run `28378064114`: passed.
- `Python` run `28378064041`: passed.
- `Native Core` run `28378063931`: passed, including post-wheel `python/tests/test_native_stokes_parity.py python/tests/test_native_sampler_parity.py`.
- CI artifact `native-wheel-windows-11-arm` from run `28378063931`: artifact id `7954360318`, digest `sha256:2ffda8e8025b9aeeeec13f55fe9dce18bf96765244495cbd1a1100bad5f6df02`.
- CI artifact smoke: downloaded `native-wheel-windows-11-arm`, installed `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl` into a clean temporary venv, and ran `blackhole-accelerators doctor --json --fail-on-emulation`; passed with `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, and `emulation_detected=false`.
- CI artifact sampler benchmark smoke: `blackhole-benchmark --target sample-brick-trilinear --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1` passed with `native_available=true`, `allclose=true`, `max_abs_diff=0.0`, and `max_rel_diff=0.0`.

v0.9.0 invalid-domain sampler contract validated on commit `1ba99bbd71b29177dd5789aca4052bf828cc0540`:

- `Architecture Report` run `28379479253`: passed.
- `Python` run `28379479067`: passed.
- `Native Core` run `28379479168`: passed, including post-wheel `python/tests/test_native_stokes_parity.py python/tests/test_native_sampler_parity.py`.
- CI artifact `native-wheel-windows-11-arm` from run `28379479168`: artifact id `7954966621`, digest `sha256:33e7374454f6f6669b7801d140eb80dbf1005ab448184649e2d9914a95cf0e25`.
- CI artifact smoke: downloaded `native-wheel-windows-11-arm`, installed `blackhole_native-0.9.0-cp310-abi3-win_arm64.whl` into a clean temporary venv, and ran `blackhole-accelerators doctor --json --fail-on-emulation`; passed with `native_core_loaded=true`, `native_core_arch=arm64`, `native_core_version=0.9.0`, and `emulation_detected=false`.
- CI artifact sampler parity smoke: `python -m pytest -q python/tests/test_native_sampler_parity.py` passed with the downloaded wheel installed.
- CI artifact sampler benchmark smoke: `blackhole-benchmark --target sample-brick-trilinear --json --nr 3 --ntheta 3 --nphi 4 --points 7 --iterations 1` passed with `native_available=true`, `allclose=true`, `max_abs_diff=0.0`, and `max_rel_diff=0.0`.

Blocked locally:

- No local native-toolchain blocker is currently known for Windows ARM64.
- No current GitHub CI blocker is known for this milestone.

## Local Toolchain Notes

- Installed GitHub CLI `2.95.0`.
- Authenticated `gh` as `Protonmatter` for HTTPS Git operations.
- Created and pushed public GitHub repo `Protonmatter/blackhole-sim`.
- Installed Visual Studio Build Tools 2022 `17.14.35` with ARM64 MSVC and Windows 11 SDK 26100 components.
- Installed Rustup and Rust `1.96.0`.
- Installed `maturin 1.14.1`.
- Installed `blackhole-sim` editable into the Python 3.14 ARM64 user environment.
- Added user PATH entries for Cargo, winget links, and Python 3.14 ARM64 scripts.
- Set a Rustup override for this checkout to `stable-aarch64-pc-windows-gnullvm`; `.cargo/config.toml` configures `rust-lld.exe` and static CRT flags for that target.
