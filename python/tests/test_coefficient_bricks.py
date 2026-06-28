from pathlib import Path
import numpy as np

from blackhole_sim.calibration import PhysicalScaling
from blackhole_sim.coefficient_bricks import COEFF_NAMES, estimate_brick_memory, precompute_coefficient_bricks, CoefficientBrickGrid
from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.physics import BlackHoleSystem


def test_estimate_brick_memory_positive():
    est = estimate_brick_memory(8, 6, 4, 'float32')
    assert est['total_bytes'] > 8 * 6 * 4 * len(COEFF_NAMES) * 3
    assert est['total_mib'] > 0


def test_precompute_coefficient_bricks_npz_roundtrip(tmp_path: Path):
    snap = generate_analytic_grmhd_torus(spin_a=0.5, nr=5, ntheta=4, nphi=3)
    scaling = PhysicalScaling.from_mdot(BlackHoleSystem.m87_star(0.5), 1e-4)
    bricks = precompute_coefficient_bricks(snap, scaling, nu_hz=230e9, dtype='float32', stride=1)
    assert bricks.shape == (5, 4, 3, len(COEFF_NAMES))
    assert bricks.coeffs.dtype == np.float32
    assert np.all(np.isfinite(bricks.coeffs))
    out = tmp_path / 'bricks.npz'
    bricks.save_npz(out)
    loaded = CoefficientBrickGrid.load_npz(out)
    assert loaded.shape == bricks.shape
    assert loaded.spin_a == bricks.spin_a
    assert loaded.nu_hz == bricks.nu_hz


def test_precompute_stride_reduces_grid():
    snap = generate_analytic_grmhd_torus(spin_a=0.3, nr=6, ntheta=5, nphi=4)
    scaling = PhysicalScaling.from_mdot(BlackHoleSystem.sgr_a_star(0.3), 1e-8)
    bricks = precompute_coefficient_bricks(snap, scaling, dtype='float16', stride=2)
    assert bricks.shape[:3] == (3, 3, 2)
    assert bricks.coeffs.dtype == np.float16
