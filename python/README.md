# Kerr GRMHD / GRRT / WebGPU Black-Hole Simulator

Version `0.5.0` adds the external validation layer requested after the polarized GRMHD build:

1. public GRMHD dump manifest and acquisition helpers,
2. downloaded-dump schema verification,
3. adapter validation reports with SHA-256 evidence,
4. `ipole` parameter-file generation,
5. Stokes `I,Q,U,V` comparison against external `ipole` HDF5 image output,
6. a notebook-driven public-dump validation workflow.

The project remains a research reference implementation. It now has the correct seams for validating against public GRMHD dumps and external polarized GRRT codes, but it is still not a substitute for a production EHT pipeline unless a selected dump, coordinate convention, unit normalization, camera convention, and external-code baseline are all pinned.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,hdf5]
```

## Public GRMHD dump workflow

Write the bundled public-data manifest:

```bash
blackhole-public-dump \
  --write-manifest \
  --manifest data/public/illinois_v3_manifest.json
```

The bundled manifest targets the Illinois Simulation Data Products v3 GRMHD collection and a SANE, `a*=+0.5` selected-dump workflow. Select a concrete HDF5 dump from the public portal, then download it with a pinned SHA-256 when available:

```bash
blackhole-public-dump \
  --download-url "SELECTED_PUBLIC_HDF5_URL" \
  --download-output data/public/selected_dump.h5 \
  --sha256 "EXPECTED_SHA256"
```

Verify that the selected dump can be mapped through one of the adapters:

```bash
blackhole-public-dump \
  --verify data/public/selected_dump.h5 \
  --adapter harm \
  --report out/public_dump_verification.json
```

The verification report records:

- SHA-256,
- HDF5 dataset inventory,
- inferred grid shape,
- spin/time metadata,
- density and electron-temperature ranges,
- adapter field map,
- adapter warnings.

## Render our polarized image from the selected dump

```bash
blackhole-render-stokes \
  --snapshot data/public/selected_dump.h5 \
  --adapter harm \
  --width 64 \
  --height 64 \
  --mass m87 \
  --rho-scale 1e-18 \
  --b-scale 30 \
  --output out/ours_stokes.npz \
  --preview out/ours_stokes_preview.png
```

## External `ipole` validation workflow

Generate an `ipole` parameter file for the same dump/camera/frequency/scaling:

```bash
blackhole-external-validation write-ipole-par \
  --dump data/public/selected_dump.h5 \
  --outfile out/ipole_image.h5 \
  --par out/ipole_validation.par \
  --nx 64 \
  --ny 64 \
  --freq-hz 230e9 \
  --mbh-msun 6.2e9 \
  --m-unit-g 1e25 \
  --thetacam 17 \
  --fov-muas 160
```

Run a locally built `ipole` binary:

```bash
blackhole-external-validation run-ipole \
  --ipole /path/to/ipole \
  --par out/ipole_validation.par
```

Compare our Stokes cube against the `ipole` HDF5 image output:

```bash
blackhole-external-validation compare \
  --ours out/ours_stokes.npz \
  --ipole out/ipole_image.h5 \
  --report out/ipole_comparison.json \
  --rtol-l1 0.2
```

`read_ipole_stokes_hdf5` handles the `ipole` image convention where `pol` is stored as `(NX, NY, 5)` containing `I,Q,U,V,tauF`, and converts it to row-major `(NY, NX, 4)`.

## Notebook

Use this notebook for the full evidence trail:

```text
notebooks/public_dump_ipole_validation.ipynb
```

It covers:

1. install,
2. manifest creation,
3. selected public dump download,
4. HDF5 adapter verification,
5. our Stokes render,
6. `ipole` parameter file generation,
7. external run and Stokes image comparison.

## Existing renderer capabilities retained

### Kerr + GRMHD unpolarized GRRT render

```bash
blackhole-render-grrt \
  --width 320 \
  --height 180 \
  --spin 0.85 \
  --inclination 68 \
  --camera-radius 55 \
  --output out/grrt_kerr.png \
  --progress
```

### Generate a deterministic local torus fixture

```bash
blackhole-make-snapshot \
  --spin 0.85 \
  --nr 72 \
  --ntheta 40 \
  --nphi 48 \
  --output out/torus_fixture.npz
```

### CPU/WebGPU regression baseline

```bash
blackhole-regression --create --reference out/cpu_reference.npz
blackhole-regression --reference out/cpu_reference.npz
```

## Test status in this package

The current validation suite covers Kerr geometry, GRMHD interpolation and adapters, public-dump provenance gates, polarized transfer, synchrotron coefficients, WebGPU asset hooks, native platform probes, and the accelerated coefficient-brick renderer. Use the root `docs/BUILD_STATE.md` file for the latest local validation evidence.

## v0.6.0 GPU/WebGPU/native acceleration layer

This build adds the hardware execution boundary needed for full-HD and interactive rendering. The CPU renderer remains the authoritative reference path; production-speed rendering should use precomputed transfer coefficient bricks plus one of the GPU backends.

### New commands

```bash
blackhole-accelerators list
blackhole-accelerators plan --width 1920 --height 1080 --backend auto
blackhole-accelerators plan --width 1920 --height 1080 --backend interactive
```

Precompute coefficient bricks for GPU upload:

```bash
blackhole-precompute-bricks \
  --snapshot data/public/selected_dump.h5 \
  --dtype float32 \
  --stride 1 \
  --format hdf5 \
  --output out/coefficient_bricks.h5
```

Quick memory estimate without computing coefficients:

```bash
blackhole-precompute-bricks --estimate-only --stride 2 --dtype float16
blackhole-benchmark --json --nr 8 --ntheta 6 --nphi 8 --iterations 1
```

### Backend policy

| Backend | Intended use | Status in this package |
|---|---|---|
| WebGPU/WGSL | Browser/native-portable interactive preview | WGSL compute skeleton + existing browser shader path |
| NVIDIA CUDA | Offline high-quality full-HD/4K renders | Detection + native kernel skeleton |
| Apple Metal | Apple Silicon interactive/offline native path | Detection + MSL kernel skeleton |
| AMD ROCm/HIP | Offline native AMD GPU path | Detection + HIP/CUDA-style skeleton |
| Intel | Prefer SYCL/WebGPU for physics kernels; OpenVINO for learned surrogates | OpenVINO detection + warning path |
| ARM NEON/SVE | CPU-side SIMD coefficient precompute/fallback | Detection + scheduling path |
| CPU | Correctness/reference/regression | Fully retained |

### Why coefficient bricks

The full polarized path is expensive because live rendering requires GRMHD interpolation, synchrotron emission/absorption/Faraday coefficient evaluation, Kerr geodesic integration, and Stokes transport. The v0.6 design moves the heavy coefficient calculation to a preprocessing phase:

```text
CPU/native preprocessing
  GRMHD dump -> calibrated local plasma state -> j/alpha/rho coefficient bricks

GPU render loop
  one thread per pixel -> Kerr ray integration -> trilinear coefficient sampling -> Stokes integration
```

This keeps the GPU hot loop mostly arithmetic and texture/buffer sampling. It also lets WebGPU, CUDA, Metal, and ROCm share the same physical input tensors.

### Native kernel assets

```text
native/cuda/kerr_stokes_kernel.cu
native/metal/kerr_stokes_kernel.metal
native/opencl/kerr_stokes_kernel.cl
native/rocm/kerr_stokes_kernel.hip
webgpu/src/stokes_brick_compute.wgsl
```

The kernels are intentionally skeletal in v0.6.0. They define the stable buffer layouts and launch model but do not yet claim feature parity with the CPU reference renderer. The next implementation pass should port the Kerr Hamiltonian integrator, brick interpolation, and Stokes RK2/exponential transfer step into WGSL/CUDA/Metal/HIP, then add image regression against the CPU reference.

### v0.6.0 full test result

```text
47 passed
elapsed=0:22.85
```

## v0.7.0 accelerator hot-loop port

This build moves the actual render hot loop out of pure Python reference code and into backend kernel assets:

- `webgpu/src/stokes_brick_compute.wgsl`
- `native/cuda/kerr_stokes_kernel.cu`
- `native/metal/kerr_stokes_kernel.metal`
- `native/rocm/kerr_stokes_kernel.hip`
- `native/opencl/kerr_stokes_kernel.cl`

The ported hot loop includes:

1. Kerr Hamiltonian geodesic stepping in the kernel.
2. Trilinear coefficient-brick sampling from flattened `[nr][ntheta][nphi][11]` grids.
3. Stokes `I,Q,U,V` RK2 transport.
4. One-thread / one-invocation-per-pixel execution layout.
5. Progressive WebGPU rendering hooks through `shader=stokes`.

A CPU-emulated version of the same coefficient-brick kernel path is available for deterministic regression:

```bash
blackhole-render-accelerated \
  --width 64 \
  --height 36 \
  --max-steps 700 \
  --output out/accelerated_stokes.npz \
  --preview out/accelerated_preview.png
```

Run the WebGPU progressive renderer:

```bash
cd webgpu
python -m http.server 8080
```

Open:

```text
http://localhost:8080/?shader=stokes
```

Boundary: the CUDA/Metal/HIP/OpenCL files are source-level kernel ports included in this package. They were not compiled against local vendor SDKs in this runtime. The mandatory regression path uses the CPU-emulated coefficient-brick renderer plus static kernel-asset checks.
