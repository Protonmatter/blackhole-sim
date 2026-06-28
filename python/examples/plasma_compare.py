"""Compare explicit Boris gyromotion vs gyro-averaged guiding center."""

import numpy as np

from blackhole_sim.plasma import (
    ParticleState,
    boris_push,
    recommended_boris_dt,
    initial_guiding_center_from_particle,
    guiding_center_push,
)

qom = 1.0
B0 = np.array([0.0, 0.0, 10.0])
E0 = np.array([0.0, 0.0, 0.0])

def E(x):
    return np.broadcast_to(E0, np.shape(x))

def B(x):
    return np.broadcast_to(B0, np.shape(x))

x0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([1.0, 0.0, 0.2])

# Explicit algorithm: many tiny steps to resolve the gyro-orbit.
dt_boris = recommended_boris_dt(qom, np.linalg.norm(B0), points_per_gyration=96)
steps = int(50.0 / dt_boris)
particle = boris_push(ParticleState(x=x0, v=v0), qom, E, B, dt_boris, steps)

# Averaged algorithm: large steps because gyro-motion is not explicitly resolved.
gc0 = initial_guiding_center_from_particle(x0, v0, qom, B0)
gc = guiding_center_push(gc0, qom, E, B, dt=0.25, steps=int(50.0 / 0.25))

print("Boris final particle position:", particle.x)
print("Guiding-center final gyro-center:", gc.R)
print("Guiding-center magnetic moment:", gc.mu)
