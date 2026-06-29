# Active Milestone: v0.9 Native Hot-Loop Parity

## Goal

Move deterministic hot-loop workloads from Python into Rust native CPU code without changing public simulator behavior. The accepted targets are intentionally small: RK2 Stokes stepping, trilinear coefficient sampling, and a composed sample-and-step Stokes micro-kernel.

## Required Work

- Add Rust native CPU implementations behind the existing Python orchestration path.
- Keep the Python implementation as the reference and fallback.
- Add regression tests that compare native output against the accepted Python envelope.
- Extend `blackhole-benchmark --json` so it reports reference versus native timings on the same workload.
- Keep `blackhole-accelerators doctor --fail-on-emulation` as a release gate for native validation.
- Next target after sampler plus RK2 parity is coefficient-brick precompute loop migration, not full renderer replacement.

## Release Gates

- Native and Python outputs match within documented deterministic tolerances.
- Benchmark output includes platform, architecture, backend, workload size, iteration count, and elapsed time.
- CI installs the native wheel and runs doctor plus the parity regression.
- Surface Pro 11 ARM64 validation uses ARM64 Python and ARM64 native modules with no x64 emulation.

## Non-Goals

- Do not claim GPU parity in v0.9.
- Do not replace the Python reference path.
- Do not publish signed desktop apps or PyPI artifacts until native parity and release policy are explicit.
