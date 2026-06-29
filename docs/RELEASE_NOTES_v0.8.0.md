# v0.8.0 Native Foundation Release Notes

Date: 2026-06-29

## Summary

`v0.8.0 Native Foundation` establishes the repository, validation gates, and first native-extension scaffold for future native black-hole simulator work. The release keeps Python as the orchestration and correctness path while adding architecture detection, CI wheel smoke builds, and a Rust/PyO3 `blackhole_native` module.

## Shipped

- Imported the v0.7 simulator into the unified `python/`, `native/`, `web/`, and `docs/` layout.
- Added `blackhole-accelerators doctor --json --fail-on-emulation` for architecture and emulation reporting.
- Added Rust/PyO3 native core scaffold with `blackhole_native.core_version()` and `blackhole_native.detect_arch()`.
- Added GitHub Actions for Python tests, Rust tests, native wheel smoke builds, and architecture reports across Windows, macOS, Linux, x64, and ARM64 targets.
- Hardened optional dependency behavior, HDF5 test skips, WebGPU Stokes routing, public-data gates, and benchmark probes.

## Non-Goals

- No PyPI publication.
- No GitHub Release or tag is required for this stabilization pass.
- No signed or notarized desktop apps.
- No claim of full native hot-loop parity or GPU physics parity.

## Validation Evidence

- Baseline commit: `46f20630f41556dce75cb6142a54ff816185c54a`.
- GitHub `Python` run `28345975404`: passed.
- GitHub `Native Core` run `28345975402`: passed.
- GitHub `Architecture Report` run `28345986411`: passed.
- Windows ARM64 CI wheel artifact installed in a clean venv and passed `blackhole-accelerators doctor --json --fail-on-emulation`.

## Next Direction

The next milestone should be v0.9 native hot-loop parity: move one deterministic coefficient/Stokes workload into Rust CPU code, compare it against the Python reference envelope, and require benchmark evidence before any performance claim.
