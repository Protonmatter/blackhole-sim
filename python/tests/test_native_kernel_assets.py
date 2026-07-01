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


def test_gpu_kernel_assets_skip_invalid_brick_samples_before_stokes_step():
    root = Path(__file__).resolve().parents[2]
    expected = [
        root / "native/cuda/kerr_stokes_kernel.cu",
        root / "native/metal/kerr_stokes_kernel.metal",
        root / "native/opencl/kerr_stokes_kernel.cl",
        root / "native/rocm/kerr_stokes_kernel.hip",
        root / "web/webgpu/src/stokes_brick_compute.wgsl",
    ]
    invalid_sample_tokens = ("return 0", "return false", "invalid_brick_sample")
    valid_guard_tokens = ("if(valid)", "if (valid)", "sample.valid > 0.5")

    for path in expected:
        text = path.read_text()
        assert any(tok in text for tok in invalid_sample_tokens), f"invalid sample sentinel missing from {path}"
        assert any(tok in text for tok in valid_guard_tokens), f"Stokes validity guard missing from {path}"


def test_webgpu_stokes_invalid_samples_use_finite_shader_values():
    root = Path(__file__).resolve().parents[2]
    text = (root / "web/webgpu/src/stokes_brick_compute.wgsl").read_text()
    assert "fn invalid_brick_sample()" in text
    assert "out.valid = 0.0" in text
    assert "out.coeffs[c] = 0.0" in text
    assert "bitcast<f32>" not in text
    assert ")) / (2.0 * e)" not in text
    assert "* (1.0 / (2.0 * e))" in text
    assert "@compute @workgroup_size(8, 8, 1)\nfn main" in text
    assert "@compute @workgroup_size(8, 8, 1)\nfn kerr_stokes_render_kernel" not in text
    assert "fn kerr_stokes_render_kernel(gid: vec3<u32>)" in text
