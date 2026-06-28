"""Physically grounded Kerr black-hole renderer.

The renderer performs backward ray tracing from a local ZAMO camera through the
full Kerr metric, detects equatorial disk crossings, and applies redshift from
an equatorial circular-orbit emitter four-velocity. It is designed as a CPU
reference implementation and as the validation target for the WebGPU port.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .kerr import (
    LocalCamera,
    camera_ray_initial_state,
    circular_orbit_four_velocity,
    horizon_radius,
    isco_radius,
    kerr_metric_contravariant,
    redshift_factor_to_observer,
    trace_kerr_null_geodesic,
)


@dataclass(frozen=True)
class KerrDiskModel:
    inner_radius: float | None = None
    outer_radius: float = 40.0
    emissivity_power: float = 2.6
    density_power: float = 0.6
    prograde: bool = True
    monochromatic: bool = True
    max_crossings: int = 3


@dataclass(frozen=True)
class KerrRenderConfig:
    width: int = 640
    height: int = 360
    spin_a: float = 0.7
    camera: LocalCamera = LocalCamera.from_degrees(r=55.0, inclination_degrees=65.0, fov_y_degrees=34.0)
    disk: KerrDiskModel = KerrDiskModel()
    step: float = 0.05
    max_steps: int = 5000
    escape_radius: float = 220.0
    exposure: float = 1.0
    gamma: float = 2.2
    background: bool = True
    workers: int = 1


def _hash01(x: float, y: float) -> float:
    return float(np.mod(np.sin(x * 127.1 + y * 311.7) * 43758.5453123, 1.0))


def _background(ndc_x: float, ndc_y: float, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.zeros(3)
    r2 = ndc_x * ndc_x + ndc_y * ndc_y
    base = np.array([0.0015, 0.0018, 0.0025]) * (1.0 + 0.8 * max(0.0, 1.0 - r2))
    h = _hash01(math.floor((ndc_x + 1.0) * 900.0), math.floor((ndc_y + 1.0) * 520.0))
    if h > 0.9975:
        return base + np.array([0.5, 0.54, 0.62]) * ((h - 0.9975) / 0.0025) ** 3
    return base


def _disk_spectrum_color(r: float, g_shift: float, disk: KerrDiskModel, inner: float) -> np.ndarray:
    # Analytic emissivity remains a source model. Gravity/orbits/redshift are Kerr.
    radial = (max(r, inner) / inner) ** (-disk.emissivity_power)
    taper_outer = max(0.0, min(1.0, (disk.outer_radius - r) / max(1.0, 0.15 * disk.outer_radius)))
    ring = math.exp(-0.5 * ((r - 1.4 * inner) / max(0.8, 0.35 * inner)) ** 2)
    emissivity = radial * (0.35 + ring) * taper_outer
    boost_power = 3.0 if disk.monochromatic else 4.0
    intensity = emissivity * max(g_shift, 0.0) ** boost_power
    # Temperature proxy: blueshift makes inner disk whiter/bluer, redshift warmer.
    warmth = np.array([1.0, 0.54, 0.20])
    white = np.array([0.9, 0.95, 1.0])
    mix = max(0.0, min(1.0, (g_shift - 0.55) / 0.95))
    return intensity * ((1.0 - mix) * warmth + mix * white)


def _crossings_with_equator(states: np.ndarray) -> list[tuple[float, int, float]]:
    crossings: list[tuple[float, int, float]] = []
    target = math.pi / 2.0
    theta = states[:, 2]
    for i in range(1, len(states)):
        a0 = theta[i - 1] - target
        a1 = theta[i] - target
        if a0 == 0.0:
            f = 0.0
        elif a0 * a1 > 0.0:
            continue
        else:
            f = abs(a0) / max(abs(a0 - a1), 1.0e-15)
        r = (1.0 - f) * states[i - 1, 1] + f * states[i, 1]
        crossings.append((float(r), i, float(f)))
    return crossings


def shade_kerr_ray(cfg: KerrRenderConfig, ndc_x: float, ndc_y: float) -> np.ndarray:
    aspect = cfg.width / cfg.height
    y0, p_t, p_phi = camera_ray_initial_state(cfg.camera, cfg.spin_a, ndc_x, ndc_y, aspect)
    trace = trace_kerr_null_geodesic(
        y0,
        p_t,
        p_phi,
        cfg.spin_a,
        step=cfg.step,
        max_steps=cfg.max_steps,
        escape_radius=cfg.escape_radius,
        store_stride=1,
    )
    if trace.status == "captured":
        color = np.zeros(3)
    else:
        color = _background(ndc_x, ndc_y, cfg.background)

    inner = cfg.disk.inner_radius if cfg.disk.inner_radius is not None else isco_radius(cfg.spin_a, cfg.disk.prograde)
    crossings = _crossings_with_equator(trace.states)
    alpha_remaining = 1.0
    used = 0
    for r, idx, _ in crossings:
        if r < inner or r > cfg.disk.outer_radius or r <= horizon_radius(cfg.spin_a):
            continue
        state = trace.states[idx]
        p_cov = np.array([p_t, state[4], state[5], p_phi], dtype=float)
        try:
            u_emit = circular_orbit_four_velocity(r, cfg.spin_a, prograde=cfg.disk.prograde)
            g_shift = redshift_factor_to_observer(p_cov, u_emit, observed_energy=1.0)
        except ValueError:
            continue
        if g_shift <= 0.0:
            continue
        local = _disk_spectrum_color(r, g_shift, cfg.disk, inner)
        opacity = min(0.72, 0.28 + 0.22 * (inner / max(r, inner)) ** cfg.disk.density_power)
        color = color + alpha_remaining * opacity * local
        alpha_remaining *= (1.0 - opacity)
        used += 1
        if used >= cfg.disk.max_crossings or alpha_remaining < 0.04:
            break

    color *= cfg.exposure
    color = color / (1.0 + color)  # filmic compression
    color = np.clip(color, 0.0, 1.0) ** (1.0 / cfg.gamma)
    return color


def _render_row(args: tuple[KerrRenderConfig, int]) -> tuple[int, np.ndarray]:
    cfg, j = args
    row = np.zeros((cfg.width, 3), dtype=np.float32)
    ndc_y = 1.0 - 2.0 * (j + 0.5) / cfg.height
    for i in range(cfg.width):
        ndc_x = 2.0 * (i + 0.5) / cfg.width - 1.0
        row[i] = shade_kerr_ray(cfg, ndc_x, ndc_y)
    return j, row


def render_kerr_image(cfg: KerrRenderConfig, progress: bool = False) -> np.ndarray:
    img = np.zeros((cfg.height, cfg.width, 3), dtype=np.float32)
    workers = int(cfg.workers)
    if workers == 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1 or cfg.height < 8:
        for j in range(cfg.height):
            if progress and (j % max(1, cfg.height // 20) == 0):
                print(f"row {j + 1}/{cfg.height}")
            jj, row = _render_row((cfg, j))
            img[jj] = row
        return img

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (j, row) in enumerate(ex.map(_render_row, ((cfg, j) for j in range(cfg.height))), start=1):
            img[j] = row
            if progress and (n % max(1, cfg.height // 20) == 0):
                print(f"rows {n}/{cfg.height}")
    return img


def save_png(img: np.ndarray, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(p, np.clip(img, 0.0, 1.0))
