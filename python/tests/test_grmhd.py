import math

import numpy as np

from blackhole_sim.grmhd import (
    GRMHDSnapshot,
    assert_four_velocity_normalization,
    generate_analytic_grmhd_torus,
)


def test_generate_torus_schema_and_four_velocity_normalization():
    snap = generate_analytic_grmhd_torus(spin_a=0.75, nr=10, ntheta=8, nphi=7)
    assert snap.shape == (10, 8, 7)
    assert snap.rho.shape == snap.shape
    assert snap.u_con.shape == snap.shape + (4,)
    assert assert_four_velocity_normalization(snap, samples=40) < 1e-10


def test_periodic_phi_interpolation_matches_wrapped_sample():
    snap = generate_analytic_grmhd_torus(spin_a=0.5, nr=12, ntheta=9, nphi=11)
    r = float(snap.r[5])
    th = float(snap.theta[4])
    s0 = snap.sample(r, th, 0.37)
    s1 = snap.sample(r, th, 0.37 + 2.0 * math.pi)
    assert s0.valid and s1.valid
    assert np.isclose(s0.rho, s1.rho)
    assert np.allclose(s0.u_con, s1.u_con)


def test_npz_roundtrip(tmp_path):
    snap = generate_analytic_grmhd_torus(spin_a=0.6, nr=9, ntheta=7, nphi=6)
    path = tmp_path / "snap.npz"
    snap.to_npz(path)
    loaded = GRMHDSnapshot.from_npz(path)
    assert loaded.shape == snap.shape
    assert loaded.spin_a == snap.spin_a
    assert np.allclose(loaded.rho, snap.rho)
