"""Validate guiding-center drift against explicit Boris/Vay particle pushers."""

import numpy as np

from blackhole_sim.relativistic_plasma import (
    RelativisticParticleState,
    push_particles,
    push_guiding_center,
    relativistic_guiding_center_from_particle,
    relativistic_gyro_dt,
)

qom = 1.0
B0 = np.array([0.0, 0.0, 8.0])
E0 = np.array([0.15, 0.0, 0.0])


def E(x):
    return np.broadcast_to(E0, np.shape(x))


def B(x):
    return np.broadcast_to(B0, np.shape(x))

x0 = np.array([0.0, 0.0, 0.0])
u0 = np.array([0.45, 0.0, 0.25])
dt = relativistic_gyro_dt(qom, np.linalg.norm(B0), gamma=1.2, points_per_gyration=128)
T = 25.0
steps = int(T / dt)

boris = push_particles(RelativisticParticleState(x0, u0), qom, E, B, dt, steps, method="boris")
vay = push_particles(RelativisticParticleState(x0, u0), qom, E, B, dt, steps, method="vay")
gc0 = relativistic_guiding_center_from_particle(x0, u0, qom, B0)
gc = push_guiding_center(gc0, qom, E, B, dt=0.05, steps=int(T / 0.05))

print("Boris final x:", boris.x)
print("Vay final x:  ", vay.x)
print("GC final R:   ", gc.R)
print("Expected E x B drift velocity:", np.cross(E0, B0) / np.dot(B0, B0))
