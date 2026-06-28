"""Accelerated coefficient-brick renderer and CPU kernel emulation.

This module provides the concrete hot-loop structure shared by the GPU/WebGPU
ports:

1. launch one Kerr ray per pixel,
2. integrate Kerr Hamiltonian null geodesics,
3. trilinearly sample precomputed polarized transfer coefficient bricks,
4. advance Stokes I,Q,U,V with RK2 or the exact matrix step,
5. schedule work in tiles / progressive levels.

The actual native kernels live under ``native/`` and ``webgpu/src/``. In the
pytest/CPU environment we use the same algorithmic layout as a deterministic
CPU emulation so small-image regression can run without requiring a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coefficient_bricks import COEFF_NAMES, CoefficientBrickGrid
from .grmhd import GRMHDSnapshot, _bracket_linear, _bracket_periodic_phi
from .kerr import LocalCamera, camera_ray_initial_state, trace_kerr_null_geodesic
from .polarized_renderer import PolarizedRenderConfig, save_stokes_npz, save_stokes_preview_png, stokes_to_rgb
from .polarized_transfer import PolarizedTransferConfig, stokes_step_exact, stokes_step_rk2
from .synchrotron import PolarizedCoefficients
from .tiles import RenderTile, generate_tiles, progressive_levels
from .webgpu_regression import ImageRegressionMetrics, image_metrics

COEFF_INDEX = {name: i for i, name in enumerate(COEFF_NAMES)}


@dataclass(frozen=True)
class AcceleratedRenderConfig:
    width: int = 160
    height: int = 90
    camera: LocalCamera = LocalCamera.from_degrees(r=55.0, inclination_degrees=65.0, fov_y_degrees=32.0)
    step: float = 0.06
    max_steps: int = 5000
    escape_radius: float = 220.0
    transfer: PolarizedTransferConfig = PolarizedTransferConfig(use_exact_matrix_step=False)
    tile_size: int = 64
    progressive_min_width: int = 480


@dataclass(frozen=True)
class ProgressiveFrame:
    level: int
    width: int
    height: int
    stokes: np.ndarray


@dataclass(frozen=True)
class SmallRegressionResult:
    metrics: ImageRegressionMetrics
    reference: np.ndarray
    candidate: np.ndarray


def _make_coeff(arr: np.ndarray) -> PolarizedCoefficients:
    a = np.asarray(arr, dtype=float)
    return PolarizedCoefficients(
        a[COEFF_INDEX["j_i"]], a[COEFF_INDEX["j_q"]], a[COEFF_INDEX["j_u"]], a[COEFF_INDEX["j_v"]],
        a[COEFF_INDEX["alpha_i"]], a[COEFF_INDEX["alpha_q"]], a[COEFF_INDEX["alpha_u"]], a[COEFF_INDEX["alpha_v"]],
        a[COEFF_INDEX["rho_v"]], a[COEFF_INDEX["rho_q"]], a[COEFF_INDEX["rho_u"]],
    )


def sample_coefficient_brick(bricks: CoefficientBrickGrid, r: float, theta: float, phi: float) -> np.ndarray | None:
    """Trilinearly sample one coefficient vector from a precomputed brick grid."""
    rb = _bracket_linear(bricks.r, float(r))
    tb = _bracket_linear(bricks.theta, float(theta))
    if rb is None or tb is None:
        return None
    r0, r1, wr = rb
    t0, t1, wt = tb
    p0, p1, wp = _bracket_periodic_phi(bricks.phi, float(phi))
    v = bricks.coeffs
    c000 = v[r0, t0, p0]
    c001 = v[r0, t0, p1]
    c010 = v[r0, t1, p0]
    c011 = v[r0, t1, p1]
    c100 = v[r1, t0, p0]
    c101 = v[r1, t0, p1]
    c110 = v[r1, t1, p0]
    c111 = v[r1, t1, p1]
    c00 = (1.0 - wp) * c000 + wp * c001
    c01 = (1.0 - wp) * c010 + wp * c011
    c10 = (1.0 - wp) * c100 + wp * c101
    c11 = (1.0 - wp) * c110 + wp * c111
    c0 = (1.0 - wt) * c00 + wt * c01
    c1 = (1.0 - wt) * c10 + wt * c11
    return np.asarray((1.0 - wr) * c0 + wr * c1, dtype=float)


def integrate_stokes_from_bricks_trace(
    trace_states: np.ndarray,
    bricks: CoefficientBrickGrid,
    affine_to_length_cm: float,
    use_exact_matrix_step: bool = False,
    max_optical_depth: float = 30.0,
) -> np.ndarray:
    states = np.asarray(trace_states, dtype=float)
    if states.shape[0] < 2:
        return np.zeros(4, dtype=float)
    S = np.zeros(4, dtype=float)
    tau = 0.0
    for i in range(1, states.shape[0]):
        prev = states[i - 1]
        cur = states[i]
        mid = 0.5 * (prev + cur)
        coeff_arr = sample_coefficient_brick(bricks, float(mid[1]), float(mid[2]), float(mid[3]))
        if coeff_arr is None:
            continue
        coeff = _make_coeff(coeff_arr)
        dlambda = float(np.linalg.norm(cur[1:4] - prev[1:4]))
        ds_cm = max(affine_to_length_cm * max(dlambda, 1.0e-14), 0.0)
        S = stokes_step_exact(S, coeff, ds_cm) if use_exact_matrix_step else stokes_step_rk2(S, coeff, ds_cm)
        tau += max(coeff.alpha_i, 0.0) * ds_cm
        if tau >= max_optical_depth:
            break
    return np.nan_to_num(np.asarray(S, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def render_tile_stokes_bricks(
    cfg: AcceleratedRenderConfig,
    snapshot: GRMHDSnapshot,
    bricks: CoefficientBrickGrid,
    tile: RenderTile,
) -> np.ndarray:
    out = np.zeros((tile.height, tile.width, 4), dtype=np.float64)
    aspect = cfg.width / cfg.height
    for tj, j in enumerate(range(tile.y0, tile.y1)):
        ndc_y = 1.0 - 2.0 * (j + 0.5) / cfg.height
        for ti, i in enumerate(range(tile.x0, tile.x1)):
            ndc_x = 2.0 * (i + 0.5) / cfg.width - 1.0
            y0, p_t, p_phi = camera_ray_initial_state(cfg.camera, snapshot.spin_a, ndc_x, ndc_y, aspect)
            trace = trace_kerr_null_geodesic(y0, p_t, p_phi, snapshot.spin_a, step=cfg.step, max_steps=cfg.max_steps, escape_radius=cfg.escape_radius, store_stride=1)
            out[tj, ti] = integrate_stokes_from_bricks_trace(
                trace.states,
                bricks,
                affine_to_length_cm=cfg.transfer.affine_to_length_cm or 1.0,
                use_exact_matrix_step=cfg.transfer.use_exact_matrix_step,
                max_optical_depth=cfg.transfer.max_optical_depth,
            )
    return out


def render_stokes_image_bricks(cfg: AcceleratedRenderConfig, snapshot: GRMHDSnapshot, bricks: CoefficientBrickGrid) -> np.ndarray:
    img = np.zeros((cfg.height, cfg.width, 4), dtype=np.float64)
    for tile in generate_tiles(cfg.width, cfg.height, tile_size=cfg.tile_size, level=0):
        block = render_tile_stokes_bricks(cfg, snapshot, bricks, tile)
        img[tile.y0:tile.y1, tile.x0:tile.x1] = block
    return img


def render_progressive_stokes_bricks(cfg: AcceleratedRenderConfig, snapshot: GRMHDSnapshot, bricks: CoefficientBrickGrid) -> list[ProgressiveFrame]:
    frames: list[ProgressiveFrame] = []
    for level, w, h in progressive_levels(cfg.width, cfg.height, min_width=cfg.progressive_min_width):
        level_cfg = AcceleratedRenderConfig(width=w, height=h, camera=cfg.camera, step=cfg.step, max_steps=cfg.max_steps, escape_radius=cfg.escape_radius, transfer=cfg.transfer, tile_size=cfg.tile_size, progressive_min_width=cfg.progressive_min_width)
        frames.append(ProgressiveFrame(level=level, width=w, height=h, stokes=render_stokes_image_bricks(level_cfg, snapshot, bricks)))
    return frames


def compare_reference_vs_bricks(reference: np.ndarray, candidate: np.ndarray) -> SmallRegressionResult:
    metrics = image_metrics(stokes_to_rgb(candidate), stokes_to_rgb(reference))
    return SmallRegressionResult(metrics=metrics, reference=np.asarray(reference), candidate=np.asarray(candidate))


def save_progressive_preview(frame: ProgressiveFrame, path: str | Path) -> None:
    save_stokes_preview_png(frame.stokes, path)


def save_progressive_npz(frame: ProgressiveFrame, path: str | Path) -> None:
    save_stokes_npz(frame.stokes, path)
