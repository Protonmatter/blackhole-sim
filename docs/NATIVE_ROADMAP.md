# Native Roadmap

## v0.8.0 Native Foundation

- Keep Python as orchestration and validation.
- Add native architecture detection and emulation checks.
- Add a Rust/PyO3 native module scaffold.
- Build CI coverage for the target operating systems and architectures.
- Preserve CPU/WebGPU reference behavior while native parity work is staged.

## Target Artifacts

```text
Windows ARM64     Surface Pro 11 Snapdragon X Elite
Windows x86_64    Intel / AMD PCs
macOS arm64       Apple Silicon
macOS x86_64      Intel Macs
Linux x86_64
Linux aarch64
```

## Backend Policy

- Tier 1: native CPU, WebGPU browser, future Rust `wgpu` path.
- Tier 2: CUDA, Metal, HIP, OpenCL/SYCL fast paths.
- Tier 3: inference accelerators only for denoising, super-resolution, coefficient surrogates, or quality selection.

Do not move the Kerr geodesic integrator into inference runtimes first. The core hot loop belongs in native CPU code or GPU compute shaders.

## Release Gates

- Wheel names and binary headers must match the target architecture.
- `blackhole-accelerators doctor --fail-on-emulation` must pass on native hardware.
- Native core architecture must match the process architecture when the native module is installed.
- A small deterministic Stokes render must match the accepted regression envelope.
- Surface Pro 11 ARM64 validation must use ARM64 Python, ARM64 native modules, and no x64 runtime.
