from pathlib import Path

REQUIRED_TOKENS = [
    "kerr_stokes_render_kernel",
    "sample_brick_trilinear",
    "stokes_step_rk2",
]
ONE_OF = ["kerr_rhs", "metric_contravariant", "ray_initial_state"]


def test_native_kernel_assets_exist_and_contain_hot_loop_parts():
    root = Path(__file__).resolve().parents[2]
    expected = [
        root / "native/cuda/kerr_stokes_kernel.cu",
        root / "native/metal/kerr_stokes_kernel.metal",
        root / "native/opencl/kerr_stokes_kernel.cl",
        root / "native/rocm/kerr_stokes_kernel.hip",
        root / "web/webgpu/src/stokes_brick_compute.wgsl",
    ]
    for path in expected:
        assert path.exists(), path
        text = path.read_text()
        for tok in REQUIRED_TOKENS:
            assert tok in text, f"{tok} missing from {path}"
        assert any(tok in text for tok in ONE_OF), f"geodesic integration token missing from {path}"
