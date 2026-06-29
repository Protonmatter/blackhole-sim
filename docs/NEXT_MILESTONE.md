# Next Milestone: v0.9 Native Hot-Loop Parity

## Goal

Move the first deterministic hot-loop workload from Python into Rust native CPU code without changing public simulator behavior. The first target should be the coefficient-brick or small Stokes workload because it already has benchmark and smoke-test coverage.

## Required Work

- Add a Rust native CPU implementation behind the existing Python orchestration path.
- Keep the Python implementation as the reference and fallback.
- Add regression tests that compare native output against the accepted Python envelope.
- Extend `blackhole-benchmark --json` so it reports reference versus native timings on the same workload.
- Keep `blackhole-accelerators doctor --fail-on-emulation` as a release gate for native validation.

## Release Gates

- Native and Python outputs match within documented deterministic tolerances.
- Benchmark output includes platform, architecture, backend, workload size, iteration count, and elapsed time.
- CI installs the native wheel and runs doctor plus the parity regression.
- Surface Pro 11 ARM64 validation uses ARM64 Python and ARM64 native modules with no x64 emulation.

## Non-Goals

- Do not claim GPU parity in v0.9.
- Do not replace the Python reference path.
- Do not publish signed desktop apps or PyPI artifacts until native parity and release policy are explicit.
