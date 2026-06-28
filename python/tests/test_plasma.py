import numpy as np

from blackhole_sim.plasma import (
    ParticleState,
    boris_push,
    guiding_center_push,
    initial_guiding_center_from_particle,
    recommended_boris_dt,
)


def test_boris_preserves_speed_in_static_b_field():
    qom = 1.0
    B0 = np.array([0.0, 0.0, 5.0])
    E0 = np.array([0.0, 0.0, 0.0])

    def E(x):
        return np.broadcast_to(E0, np.shape(x))

    def B(x):
        return np.broadcast_to(B0, np.shape(x))

    x0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([1.0, 0.1, 0.2])
    dt = recommended_boris_dt(qom, np.linalg.norm(B0), 64)
    out = boris_push(ParticleState(x=x0, v=v0), qom, E, B, dt, 500)
    assert np.isclose(np.linalg.norm(out.v), np.linalg.norm(v0), rtol=1e-10, atol=1e-10)


def test_guiding_center_advances_parallel_motion():
    qom = 1.0
    B0 = np.array([0.0, 0.0, 10.0])
    E0 = np.array([0.0, 0.0, 0.0])

    def E(x):
        return np.broadcast_to(E0, np.shape(x))

    def B(x):
        return np.broadcast_to(B0, np.shape(x))

    x0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([1.0, 0.0, 0.25])
    gc0 = initial_guiding_center_from_particle(x0, v0, qom, B0)
    out = guiding_center_push(gc0, qom, E, B, dt=0.5, steps=20)
    assert np.isclose(out.R[2] - gc0.R[2], 2.5, atol=1e-12)
    assert np.isclose(out.mu, gc0.mu)
