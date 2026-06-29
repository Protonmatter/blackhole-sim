# Performance Plan

## Objective

Move hot loops into native code only when correctness and measurement are both in place. The Python path remains the reference implementation until a native path matches its regression envelope.

## Baseline Probe

Run:

```bash
cd python
blackhole-benchmark --json --nr 8 --ntheta 6 --nphi 8 --iterations 1
```

The benchmark records coefficient-brick cell throughput, process architecture, and emulation status. Treat this as a local comparison baseline, not a cross-machine performance claim.

The default JSON payload also includes the first native parity probe:

- `stokes_rk2_brick_reference`: vectorized Python reference timing.
- `stokes_rk2_brick_native`: Rust CPU timing when `blackhole_native.stokes_rk2_brick()` is installed.
- `stokes_rk2_brick_parity`: max absolute/relative error and explicit tolerances.

## Native Acceptance Gates

- Native output must match the Python coefficient-brick or Stokes regression envelope before speedups are claimed.
- Native Stokes RK2 parity uses `rtol=1.0e-12` and `atol=1.0e-12` for the deterministic coefficient-brick fixture.
- `blackhole-accelerators doctor --json --fail-on-emulation` must pass on the target machine.
- Benchmarks must report the target architecture and whether emulation was detected.
- GPU kernels must have a CPU reference comparison for a small deterministic render before being promoted beyond prototype status.

## First Native Targets

1. Stokes RK2 transfer step over coefficient-brick cells.
2. Trilinear coefficient sampling.
3. Coefficient-brick precompute loops.
4. Full tiled ray/Stokes integration after the above have parity tests.
