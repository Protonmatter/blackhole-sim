"""Reference polarized Kerr+GRMHD renderer producing Stokes I,Q,U,V images."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .calibration import PhysicalScaling
from .grmhd import GRMHDSnapshot
from .kerr import LocalCamera, camera_ray_initial_state, trace_kerr_null_geodesic
from .kerr_renderer import save_png
from .polarized_transfer import PolarizedTransferConfig, integrate_polarized_kerr_grrt
from .synchrotron import HybridSynchrotronCoefficients


@dataclass(frozen=True)
class PolarizedRenderConfig:
    width: int = 160
    height: int = 90
    camera: LocalCamera = LocalCamera.from_degrees(r=55.0, inclination_degrees=65.0, fov_y_degrees=32.0)
    step: float = 0.06
    max_steps: int = 5000
    escape_radius: float = 220.0
    workers: int = 1
    transfer: PolarizedTransferConfig = PolarizedTransferConfig(use_exact_matrix_step=False)
    tone_map: bool = True


def shade_polarized_ray(
    cfg: PolarizedRenderConfig,
    snapshot: GRMHDSnapshot,
    scaling: PhysicalScaling,
    ndc_x: float,
    ndc_y: float,
    coeffs: HybridSynchrotronCoefficients | None = None,
) -> np.ndarray:
    aspect = cfg.width / cfg.height
    y0, p_t, p_phi = camera_ray_initial_state(cfg.camera, snapshot.spin_a, ndc_x, ndc_y, aspect)
    trace = trace_kerr_null_geodesic(y0, p_t, p_phi, snapshot.spin_a, step=cfg.step, max_steps=cfg.max_steps, escape_radius=cfg.escape_radius, store_stride=1)
    result = integrate_polarized_kerr_grrt(trace, snapshot, scaling, coeff_model=coeffs, cfg=cfg.transfer)
    return result.stokes


def _row(args: tuple[PolarizedRenderConfig, GRMHDSnapshot, PhysicalScaling, int]) -> tuple[int, np.ndarray]:
    cfg, snapshot, scaling, j = args
    row = np.zeros((cfg.width, 4), dtype=np.float64)
    ndc_y = 1.0 - 2.0 * (j + 0.5) / cfg.height
    coeffs = HybridSynchrotronCoefficients()
    for i in range(cfg.width):
        ndc_x = 2.0 * (i + 0.5) / cfg.width - 1.0
        row[i] = shade_polarized_ray(cfg, snapshot, scaling, ndc_x, ndc_y, coeffs)
    return j, row


def render_stokes_image(cfg: PolarizedRenderConfig, snapshot: GRMHDSnapshot, scaling: PhysicalScaling, progress: bool = False) -> np.ndarray:
    img = np.zeros((cfg.height, cfg.width, 4), dtype=np.float64)
    workers = int(cfg.workers)
    if workers == 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1 or cfg.height < 8:
        for j in range(cfg.height):
            if progress and j % max(1, cfg.height // 20) == 0:
                print(f"row {j + 1}/{cfg.height}")
            jj, row = _row((cfg, snapshot, scaling, j))
            img[jj] = row
        return img
    with ProcessPoolExecutor(max_workers=workers) as ex:
        iterator = ((cfg, snapshot, scaling, j) for j in range(cfg.height))
        for n, (j, row) in enumerate(ex.map(_row, iterator), start=1):
            img[j] = row
            if progress and n % max(1, cfg.height // 20) == 0:
                print(f"rows {n}/{cfg.height}")
    return img


def stokes_to_rgb(stokes_img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    S = np.asarray(stokes_img, dtype=float)
    I = np.maximum(S[..., 0], 0.0)
    Q = S[..., 1]
    U = S[..., 2]
    V = S[..., 3]
    lin = np.sqrt(Q * Q + U * U)
    # Diagnostic colorization: I controls brightness; polarization alters tint.
    rgb = np.zeros(S.shape[:2] + (3,), dtype=float)
    rgb[..., 0] = I + 0.35 * np.maximum(Q, 0.0) + 0.15 * np.maximum(V, 0.0)
    rgb[..., 1] = I + 0.25 * lin
    rgb[..., 2] = I + 0.35 * np.maximum(-Q, 0.0) + 0.15 * np.maximum(-V, 0.0)
    m = float(np.percentile(rgb, 99.5)) if np.any(rgb > 0.0) else 1.0
    rgb = rgb / max(m, 1.0e-300)
    rgb = rgb / (1.0 + rgb)
    return np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)


def save_stokes_npz(stokes_img: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, stokes=np.asarray(stokes_img, dtype=np.float64))


def save_stokes_preview_png(stokes_img: np.ndarray, path: str | Path) -> None:
    save_png(stokes_to_rgb(stokes_img), path)
