# Citations and validation anchors

This package is structured around public and open-source GRMHD/GRRT validation targets.

## Public GRMHD data

- Illinois Simulation Data Products, v3 GRMHD Output. The portal advertises SANE/MAD flux-state filters and spin filters `-0.94, -0.5, 0, +0.5, +0.94`, and states that the data are licensed under Creative Commons Attribution 4.0 International License.
- Dhruv, V.; Prather, B.; Wong, G.; Gammie, C. F. “A Survey of General Relativistic Magnetohydrodynamic Models for Black Hole Accretion Systems,” ApJS 277, 16 / arXiv:2411.12647. The paper describes ten non-radiative ideal GRMHD simulations used in EHT Sgr A* analysis and states that simulation data are publicly released at the Illinois Simulation Data Products site.

## Analysis / adapter reference

- AFD-Illinois `pyharm`: Python tools for HARM simulation analysis. The README describes analysis of GRMHD simulations, HARM-family support, lazy fluid-state calculations, coordinates in Kerr spacetime, and support for conversion paths through EHT-babel.

## External GRRT reference

- AFD-Illinois `ipole`: Polarized covariant radiative transfer code for imaging black-hole accretion systems. Its README documents HDF5/GSL dependencies, the default `iharm` model for most GRMHD fluid data, and command-line parameter usage such as `--freqcgs`, `--MBH`, `--M_unit`, `--thetacam`, `--dump`, and `--outfile`.
- AFD-Illinois image-format documentation: `ipole` image files store `unpol` and `pol` arrays; `pol` has shape `(NX, NY, 5)` with Stokes `I,Q,U,V` and `tauF`; `/header/scale` converts stored cgs intensity to Jy per pixel.

## Polarized transfer and coefficients

- Mościbrodzka & Gammie 2018: `ipole` polarized radiative transfer reference.
- Dexter 2016; Pandya et al. 2016/2018; Marszewski et al. 2021: analytic coefficient fits used by production polarized GRRT workflows.
- Shcherbakov 2008/2010: synchrotron emission, absorption, Faraday rotation, and Faraday conversion coefficient references.

## v0.6.0 GPU/WebGPU/native acceleration references

- W3C WebGPU specification, 2026 working draft: WebGPU exposes GPU hardware capabilities to the Web and includes compute shader execution through WGSL-capable pipelines.
- W3C WGSL specification: WGSL defines shader-stage access rules including storage buffers and storage textures used by compute pipelines.
- NVIDIA Numba-CUDA documentation: Numba-CUDA provides a CUDA target for writing SIMT kernels in Python.
- NVIDIA CUDA Tensor Core documentation: Tensor Cores are exposed through CUDA WMMA APIs for warp-level matrix multiply-accumulate workloads.
- Intel OpenVINO documentation: OpenVINO Runtime supports CPU/GPU/NPU inference devices; in this project it is scoped to learned coefficient surrogates rather than generic geodesic kernels.
- Apple Metal documentation: Metal is Apple’s graphics and compute API; Metal Performance Shaders provide kernels tuned for Apple GPU families.


## v0.7.0 note

The v0.7.0 package adds implementation artifacts and local regression evidence; no new external scientific claims are introduced beyond the prior Kerr/GRRT/WebGPU/CUDA/Metal/OpenVINO/ROCm references already listed above.
