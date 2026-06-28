"""CPU-vs-WebGPU regression helpers.

The mandatory path creates deterministic CPU reference images and validates image
error metrics. The optional browser path can be wired to Playwright/WebGPU in an
environment with a GPU-enabled browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .grmhd import generate_analytic_grmhd_torus
from .grrt_renderer import GRRTRenderConfig, render_grrt_image
from .kerr import LocalCamera


@dataclass(frozen=True)
class ImageRegressionMetrics:
    max_abs: float
    mean_abs: float
    rmse: float
    psnr_db: float
    sha256: str

    @property
    def passed_default(self) -> bool:
        return self.max_abs <= 0.035 and self.mean_abs <= 0.006


def image_metrics(candidate: np.ndarray, reference: np.ndarray) -> ImageRegressionMetrics:
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    psnr = float(20.0 * np.log10(1.0 / max(rmse, 1.0e-300)))
    digest = hashlib.sha256(np.asarray(candidate, dtype=np.float32).tobytes()).hexdigest()
    return ImageRegressionMetrics(float(np.max(diff)), float(np.mean(diff)), rmse, psnr, digest)


def deterministic_cpu_reference(width: int = 24, height: int = 14, spin: float = 0.72) -> np.ndarray:
    snap = generate_analytic_grmhd_torus(spin_a=spin, nr=18, ntheta=12, nphi=10)
    cfg = GRRTRenderConfig(
        width=width,
        height=height,
        camera=LocalCamera.from_degrees(r=42.0, inclination_degrees=63.0, fov_y_degrees=30.0),
        step=0.12,
        max_steps=700,
        workers=1,
        exposure=1.0,
    )
    return render_grrt_image(cfg, snap, progress=False)


def save_reference_npz(path: str | Path, image: np.ndarray | None = None, metadata: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = deterministic_cpu_reference() if image is None else np.asarray(image, dtype=np.float32)
    np.savez_compressed(p, image=img, metadata=json.dumps(metadata or {}, sort_keys=True))


def compare_with_reference(candidate: np.ndarray, reference_path: str | Path) -> ImageRegressionMetrics:
    with np.load(reference_path) as z:
        ref = z["image"]
    return image_metrics(candidate, ref)


def ensure_shader_contains_regression_hooks(shader_path: str | Path) -> None:
    text = Path(shader_path).read_text(encoding="utf-8")
    required = ["metricinv", "ray", "density", "thetae", "alpha"]
    missing = [token for token in required if token not in text.lower()]
    if missing:
        raise AssertionError(f"shader missing regression-visible hooks: {missing}")


def try_capture_webgpu_canvas(url: str, output_png: str | Path, timeout_s: float = 20.0) -> bool:
    """Optional Playwright capture hook.

    Returns False if Playwright or a browser is unavailable. This keeps CI green
    on CPU-only runners while allowing GPU agents to run the same regression with
    a browser-rendered candidate image.
    """
    try:
        import playwright.sync_api  # type: ignore  # noqa: F401
    except Exception:
        return False
    script = f"""
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(args=['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'])
    page = browser.new_page(viewport={{'width': 640, 'height': 360}})
    page.goto({url!r}, wait_until='networkidle', timeout={int(timeout_s*1000)})
    page.wait_for_timeout(1500)
    page.locator('canvas').screenshot(path={str(output_png)!r})
    browser.close()
"""
    proc = subprocess.run([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s + 10.0)
    return proc.returncode == 0 and Path(output_png).exists()
