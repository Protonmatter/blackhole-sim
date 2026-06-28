# WebGPU interactive Kerr/GRRT renderer

This browser renderer mirrors the CPU reference model closely enough to use as an
interactive porting target:

- full Kerr metric in Boyer-Lindquist coordinates,
- local ZAMO camera launch,
- fixed-step Hamiltonian ray marching per pixel,
- thin-disk shader path, and
- GRRT-volume shader path with an in-shader torus fixture.

Run from this directory:

```bash
python -m http.server 8080
```

Open one of:

```text
http://localhost:8080
http://localhost:8080/?shader=volume
```

Controls:

- Spin slider: Kerr spin `a/M`.
- Inclination slider: camera polar angle.
- Camera radius slider: observer radius in `GM/c^2`.
- Step count slider: quality/performance tradeoff.
- Density / temp / absorb sliders: GRRT-volume coefficient scaling.

The `grrt_volume.wgsl` shader currently keeps the test volume analytic so the
browser path has no external dataset dependency. The CPU path is authoritative
for HDF5/NPZ GRMHD snapshot ingestion. A production browser path should upload
validated snapshot bricks into WebGPU storage buffers or 3D textures and sample
those instead of the in-shader fixture.
