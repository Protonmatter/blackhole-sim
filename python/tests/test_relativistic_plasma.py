import numpy as np

from blackhole_sim.relativistic_plasma import (
    RelativisticParticleState,
    gamma_from_u,
    push_particles,
    relativistic_boris_step,
    relativistic_guiding_center_from_particle,
    push_guiding_center,
    vay_step,
)


def test_relativistic_boris_preserves_gamma_in_static_b():
    qom = 1.0
    B = np.array([0.0, 0.0, 6.0])
    E = np.zeros(3)
    x = np.zeros(3)
    u = np.array([0.7, 0.1, 0.2])
    g0 = gamma_from_u(u)
    for _ in range(200):
        x, u = relativistic_boris_step(x, u, qom, E, B, 0.005)
    assert np.isclose(gamma_from_u(u).item(), g0.item(), rtol=1e-11, atol=1e-11)


def test_vay_preserves_gamma_in_static_b():
    qom = 1.0
    B = np.array([0.0, 0.0, 6.0])
    E = np.zeros(3)
    x = np.zeros(3)
    u = np.array([0.7, 0.1, 0.2])
    g0 = gamma_from_u(u)
    for _ in range(200):
        x, u = vay_step(x, u, qom, E, B, 0.005)
    assert np.isclose(gamma_from_u(u).item(), g0.item(), rtol=1e-6, atol=1e-8)


def test_guiding_center_matches_parallel_motion_in_uniform_b():
    qom = 1.0
    B0 = np.array([0.0, 0.0, 10.0])
    E0 = np.zeros(3)

    def E(x):
        return np.broadcast_to(E0, np.shape(x))

    def B(x):
        return np.broadcast_to(B0, np.shape(x))

    x0 = np.zeros(3)
    u0 = np.array([0.4, 0.0, 0.3])
    gc0 = relativistic_guiding_center_from_particle(x0, u0, qom, B0)
    out = push_guiding_center(gc0, qom, E, B, dt=0.1, steps=100)
    expected_vz = u0[2] / np.sqrt(1.0 + u0[2] ** 2)
    assert np.isclose(out.R[2] - gc0.R[2], expected_vz * 10.0, rtol=5e-3)
    assert np.allclose(out.mu, gc0.mu)


def test_vay_and_boris_agree_for_pure_b_small_dt():
    qom = 1.0
    B0 = np.array([0.0, 0.0, 5.0])
    E0 = np.zeros(3)

    def E(x):
        return np.broadcast_to(E0, np.shape(x))

    def B(x):
        return np.broadcast_to(B0, np.shape(x))

    state = RelativisticParticleState(np.zeros(3), np.array([0.3, 0.1, 0.05]))
    boris = push_particles(state, qom, E, B, dt=0.002, steps=500, method="boris")
    vay = push_particles(state, qom, E, B, dt=0.002, steps=500, method="vay")
    assert np.linalg.norm(boris.u - vay.u) < 0.04
