import numpy as np

from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.grrt_renderer import GRRTRenderConfig, render_grrt_image


def test_grrt_renderer_smoke_low_resolution():
    snap = generate_analytic_grmhd_torus(spin_a=0.55, nr=12, ntheta=9, nphi=8)
    cfg = GRRTRenderConfig(width=8, height=5, step=0.12, max_steps=500, workers=1, exposure=1.0)
    img = render_grrt_image(cfg, snap, progress=False)
    assert img.shape == (5, 8, 3)
    assert np.all(np.isfinite(img))
    assert np.all((img >= 0.0) & (img <= 1.0))
