import numpy as np
from pathlib import Path

from blackhole_sim.webgpu_regression import deterministic_cpu_reference, image_metrics, save_reference_npz, compare_with_reference, ensure_shader_contains_regression_hooks


def test_cpu_reference_regression_roundtrip(tmp_path):
    img = deterministic_cpu_reference(width=5, height=3)
    path = tmp_path / 'ref.npz'
    save_reference_npz(path, img)
    metrics = compare_with_reference(img, path)
    assert metrics.max_abs == 0.0
    assert metrics.passed_default


def test_shader_has_regression_hooks():
    root = Path(__file__).resolve().parents[2]
    ensure_shader_contains_regression_hooks(root / "web/webgpu/src/grrt_volume.wgsl")
