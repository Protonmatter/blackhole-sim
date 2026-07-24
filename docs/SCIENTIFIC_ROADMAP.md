# Scientific Roadmap

## Direction

BlackHole Sim should move from an accelerator-backed visualization prototype to
a validated GRRT research platform. Python remains the canonical physics
implementation. Native Rust and WebGPU are acceleration layers that must prove
parity before they can support correctness, performance, or visual claims.

## Release Gates

- `educational_proxy`: fast visual and fixture paths are allowed, but UI, docs,
  and CLI output must identify them as approximations.
- `validated`: requires metric-aware plasma-frame calculations, pinned data
  provenance, non-proxy coefficient models or coefficient tables, and regression
  evidence.
- GPU/native parity: requires Python reference comparison for ray launch,
  geodesic stepping, coefficient sampling, Stokes stepping, and direct readback
  or compact comparison reports.
- Public-data validation: requires one selected dump with SHA-256, field map,
  accepted ranges, unit/camera/frequency conventions, and external baseline
  comparison.
- External baseline: `ipole` comparison reports must record shape, orientation,
  flux scale, Stokes component metrics, and tolerance envelope.

## Near-Term Work

1. Finish selected Illinois v3 SANE `a=+0.5` dump acquisition metadata without
   committing the dump.
2. Add a small ipole baseline report for the selected dump and a matching local
   Stokes cube report.
3. Add WebGPU diagnostics automation that proves readback for a tiny Stokes
   render on the local adapter.
4. Port the ZAMO launch numeric parity test from static source checks to runtime
   shader/native checks once vendor/WebGPU test harnesses are available.
5. Keep game engines out of the physics core. Revisit them only as presentation
   shells after validated kernels produce the physical state.
