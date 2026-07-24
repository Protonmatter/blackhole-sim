# Physics Validation Contract

## Purpose

This repository should be treated as a research reference implementation, not as
a production EHT/GRRT pipeline. The code is grounded in known physics where the
model and tests explicitly say so, and it must not claim scientific accuracy for
paths that are only visual, approximate, or parity-unproven.

## Physics Anchors

The authoritative physics path is the Python reference implementation.

- Kerr spacetime is represented in Boyer-Lindquist coordinates with
  geometrized units `G = c = M = 1`.
- Photon rays are integrated as null geodesics with Hamiltonian equations,
  `H = 0.5 g^{mu nu} p_mu p_nu = 0`.
- Camera rays are launched from a local ZAMO orthonormal tetrad so screen-space
  directions map to local physical directions before converting to coordinate
  momenta.
- The code includes Kerr horizon, static-limit, ISCO, Keplerian orbit, circular
  four-velocity, and redshift helpers.
- GRMHD snapshots are represented as explicit `r, theta, phi` grids with
  density, electron temperature, pressure, magnetic-field four-vector, and
  fluid four-velocity fields.
- Radiative transfer uses invariant redshift hooks and Stokes `I, Q, U, V`
  transport with exact matrix stepping when SciPy is available and RK2 fallback
  otherwise.
- Validated plasma-frame helpers compute magnetic-field magnitude with the Kerr
  metric invariant `sqrt(b_mu b^mu)` when snapshot spin is available, and compute
  photon/B pitch angle from the fluid-frame invariant
  `(p_mu b^mu)/(E_fluid |B|)`.

## Current Validation Gates

The minimum physics checks are:

```bash
cd python
python -m pytest tests/test_geodesics.py tests/test_kerr.py tests/test_grmhd.py tests/test_radiative_transfer.py tests/test_synchrotron_polarized_transfer.py -q
```

These cover:

- Schwarzschild critical impact parameter and capture/escape behavior.
- Kerr metric inverse identity and Schwarzschild-limit metric components.
- Local ZAMO tetrad orthonormality.
- Null camera-ray launch and short-run Hamiltonian conservation.
- Kerr horizon and ISCO limit checks.
- Circular orbit four-velocity normalization.
- GRMHD fixture schema, interpolation, and four-velocity normalization.
- Basic radiative-transfer and polarized-transfer invariants.

The broader repository gate remains:

```bash
cd python
python -m compileall blackhole_sim
python -m pytest -q
cd ..
cargo fmt --check --manifest-path native/core/Cargo.toml
cargo test --manifest-path native/core/Cargo.toml
python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation
```

## Approximation Boundaries

The following are intentionally not full scientific validation:

- The analytic torus generator is a deterministic fixture. It is not GRMHD
  solver output.
- The unpolarized thermal coefficient model is a compact testing fit. Absolute
  Jy-scale interpretation requires pinned unit conversion and calibration. It is
  the default `educational_proxy` mode and is rejected when `validated` mode is
  requested without a non-proxy coefficient model.
- Coefficient-brick precomputation now uses the same metric-aware local plasma
  helper as polarized transfer, but image-level scientific validation still
  requires a selected dump and external baseline.
- The public-data manifest is currently a collection-level discovery anchor.
  Release-grade validation requires one concrete selected dump, SHA-256,
  expected field map, accepted numeric ranges, and an external-code comparison.
- The `ipole` path provides comparison plumbing, but parity is not established
  until the same dump, camera, frequency, mass, unit normalization, and image
  convention are pinned and the comparison report passes.

## WebGPU And Native Policy

WebGPU, CUDA, Metal, HIP, OpenCL, and Rust native paths are acceleration targets,
not alternate sources of physics truth. They must reproduce the Python reference
within an accepted regression envelope before any physics parity claim.

Current policy:

- Keep Python as the reference model and validation oracle.
- Use WebGPU compute for browser-interactive kernels where direct buffer access,
  shader auditability, and deterministic readback are needed.
- Use native CPU/GPU kernels only after small hot-loop parity tests pass.
- Do not move geodesic integration or Stokes transport into a game engine's
  built-in physics system.
- A game engine may be used later as a presentation shell only if the physical
  state is still produced by the validated reference/native/WebGPU kernels.

The current WebGPU Stokes compute shader and static CUDA/Metal/HIP/OpenCL source
assets derive camera rays through a ZAMO launch helper instead of fixed photon
momenta. They still must not be described as full physics-equivalent renderers
until geodesic stepping, coefficient sampling, Stokes transport, and readback
image regression pass against the CPU reference on target hardware.

## Next Accuracy Milestones

1. Select one public GRMHD dump and record checksum, field map, unit convention,
   camera convention, and accepted numeric ranges.
2. Run the same selected dump through this renderer and a local `ipole` baseline.
3. Add a signed comparison report for Stokes image shape, orientation, flux
   scale, and tolerance envelope.
4. Add GPU/native readback tests that compare sampled rays and small rendered
   Stokes images against the Python reference before claiming parity.
