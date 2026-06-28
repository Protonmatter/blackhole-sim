import numpy as np

from blackhole_sim.accelerated_renderer import (
    AcceleratedRenderConfig,
    compare_reference_vs_bricks,
    render_progressive_stokes_bricks,
    render_stokes_image_bricks,
    sample_coefficient_brick,
)
from blackhole_sim.calibration import PhysicalScaling
from blackhole_sim.coefficient_bricks import precompute_coefficient_bricks
from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.physics import BlackHoleSystem
from blackhole_sim.polarized_renderer import PolarizedRenderConfig, render_stokes_image


def _fixture():
    snap = generate_analytic_grmhd_torus(spin_a=0.55, nr=8, ntheta=6, nphi=6)
    scaling = PhysicalScaling.from_mdot(BlackHoleSystem.sgr_a_star(0.55), 1e-8)
    bricks = precompute_coefficient_bricks(snap, scaling, nu_hz=230e9, dtype="float32", stride=1)
    return snap, scaling, bricks


def test_sample_coefficient_brick_finite():
    snap, _, bricks = _fixture()
    c = sample_coefficient_brick(bricks, float(snap.r[3]), float(snap.theta[3]), float(snap.phi[3]))
    assert c is not None
    assert c.shape == (11,)
    assert np.all(np.isfinite(c))


def test_render_stokes_image_bricks_smoke():
    snap, _, bricks = _fixture()
    cfg = AcceleratedRenderConfig(width=4, height=3, max_steps=40, step=0.18, progressive_min_width=2)
    img = render_stokes_image_bricks(cfg, snap, bricks)
    assert img.shape == (3, 4, 4)
    assert np.all(np.isfinite(img))


def test_progressive_render_has_full_resolution_last():
    snap, _, bricks = _fixture()
    cfg = AcceleratedRenderConfig(width=8, height=6, max_steps=30, step=0.2, progressive_min_width=2)
    frames = render_progressive_stokes_bricks(cfg, snap, bricks)
    assert len(frames) >= 2
    assert frames[-1].width == 8
    assert frames[-1].height == 6
    assert frames[-1].stokes.shape == (6, 8, 4)


def test_cpu_reference_vs_brick_renderer_small_regression():
    snap, scaling, bricks = _fixture()
    ref_cfg = PolarizedRenderConfig(width=3, height=2, max_steps=40, step=0.18, workers=1)
    cand_cfg = AcceleratedRenderConfig(width=3, height=2, max_steps=40, step=0.18, progressive_min_width=2)
    ref = render_stokes_image(ref_cfg, snap, scaling, progress=False)
    cand = render_stokes_image_bricks(cand_cfg, snap, bricks)
    reg = compare_reference_vs_bricks(ref, cand)
    assert reg.candidate.shape == reg.reference.shape
    assert np.isfinite(reg.metrics.mean_abs)
    assert reg.metrics.mean_abs <= 0.95
    assert reg.metrics.max_abs <= 1.0
