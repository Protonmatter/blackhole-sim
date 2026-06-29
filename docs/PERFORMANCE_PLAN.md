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

## Native Acceptance Gates

- Native output must match the Python coefficient-brick or Stokes regression envelope before speedups are claimed.
- `blackhole-accelerators doctor --json --fail-on-emulation` must pass on the target machine.
- Benchmarks must report the target architecture and whether emulation was detected.
- GPU kernels must have a CPU reference comparison for a small deterministic render before being promoted beyond prototype status.

## First Native Targets

1. Coefficient-brick precompute loops.
2. Trilinear coefficient sampling.
3. Stokes RK2/exponential transfer step.
4. Full tiled ray/Stokes integration after the above have parity tests.
