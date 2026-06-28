"""Kerr + GRMHD snapshot volume renderer.

This CPU renderer is the reference implementation for high-fidelity work. It
launches local ZAMO camera rays, traces Kerr null geodesics, samples a GRMHD
volume, and integrates unpolarized general-relativistic radiative transfer.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
import math
from pathlib import Path

import numpy as np

from .grmhd import GRMHDSnapshot
from .kerr import LocalCamera, camera_ray_initial_state, trace_kerr_null_geodesic
from .radiative_transfer import CoefficientModel, ThermalSynchrotronFit, TransferConfig, integrate_kerr_grrt
from .kerr_renderer import save_png


@dataclass(frozen=True)
class GRRTRenderConfig:
    width: int = 320
    height: int = 180
    camera: LocalCamera = LocalCamera.from_degrees(r=55.0, inclination_degrees=65.0, fov_y_degrees=32.0)
    step: float = 0.045
    max_steps: int = 6500
    escape_radius: float = 220.0
    exposure: float = 1.0
    gamma: float = 2.2
    workers: int = 1
    transfer: TransferConfig = TransferConfig()


def shade_grrt_ray(
    cfg: GRRTRenderConfig,
    snapshot: GRMHDSnapshot,
    ndc_x: float,
    ndc_y: float,
    coeffs: CoefficientModel | None = None,
) -> np.ndarray:
    aspect = cfg.width / cfg.height
    y0, p_t, p_phi = camera_ray_initial_state(cfg.camera, snapshot.spin_a, ndc_x, ndc_y, aspect)
    trace = trace_kerr_null_geodesic(
        y0,
        p_t,
        p_phi,
        snapshot.spin_a,
        step=cfg.step,
        max_steps=cfg.max_steps,
        escape_radius=cfg.escape_radius,
        store_stride=1,
    )
    tr = integrate_kerr_grrt(trace, snapshot, coeffs=coeffs or ThermalSynchrotronFit(cfg.transfer.observing_frequency_hz), cfg=cfg.transfer)
    # Sparse star-field background for empty rays.
    r2 = ndc_x * ndc_x + ndc_y * ndc_y
    background = np.array([0.0012, 0.0014, 0.0020]) * max(0.0, 1.0 - 0.25 * r2)
    if trace.status == "captured":
        background *= 0.02
    color = tr.intensity_rgb + background
    color *= cfg.exposure
    color = color / (1.0 + color)
    return np.clip(color, 0.0, 1.0) ** (1.0 / cfg.gamma)


def _render_row(args: tuple[GRRTRenderConfig, GRMHDSnapshot, int]) -> tuple[int, np.ndarray]:
    cfg, snapshot, j = args
    row = np.zeros((cfg.width, 3), dtype=np.float32)
    ndc_y = 1.0 - 2.0 * (j + 0.5) / cfg.height
    coeffs = ThermalSynchrotronFit(cfg.transfer.observing_frequency_hz)
    for i in range(cfg.width):
        ndc_x = 2.0 * (i + 0.5) / cfg.width - 1.0
        row[i] = shade_grrt_ray(cfg, snapshot, ndc_x, ndc_y, coeffs=coeffs)
    return j, row


def render_grrt_image(cfg: GRRTRenderConfig, snapshot: GRMHDSnapshot, progress: bool = False) -> np.ndarray:
    img = np.zeros((cfg.height, cfg.width, 3), dtype=np.float32)
    workers = int(cfg.workers)
    if workers == 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1 or cfg.height < 8:
        for j in range(cfg.height):
            if progress and (j % max(1, cfg.height // 20) == 0):
                print(f"row {j + 1}/{cfg.height}")
            jj, row = _render_row((cfg, snapshot, j))
            img[jj] = row
        return img
    with ProcessPoolExecutor(max_workers=workers) as ex:
        iterator = ((cfg, snapshot, j) for j in range(cfg.height))
        for n, (j, row) in enumerate(ex.map(_render_row, iterator), start=1):
            img[j] = row
            if progress and (n % max(1, cfg.height // 20) == 0):
                print(f"rows {n}/{cfg.height}")
    return img


def save_grrt_png(img: np.ndarray, path: str | Path) -> None:
    save_png(img, path)
