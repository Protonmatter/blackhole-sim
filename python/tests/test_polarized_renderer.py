import numpy as np

from blackhole_sim.calibration import PhysicalScaling
from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.physics import BlackHoleSystem
from blackhole_sim.polarized_renderer import PolarizedRenderConfig, render_stokes_image


def test_polarized_renderer_smoke():
    snap = generate_analytic_grmhd_torus(spin_a=0.4, nr=10, ntheta=8, nphi=6)
    scaling = PhysicalScaling.from_mdot(BlackHoleSystem.sgr_a_star(0.4), 1e-8)
    cfg = PolarizedRenderConfig(width=3, height=2, step=0.2, max_steps=80, workers=1)
    img = render_stokes_image(cfg, snap, scaling)
    assert img.shape == (2,3,4)
    assert np.isfinite(img).all()
