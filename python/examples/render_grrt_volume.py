from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.grrt_renderer import GRRTRenderConfig, render_grrt_image, save_grrt_png
from blackhole_sim.kerr import LocalCamera

snap = generate_analytic_grmhd_torus(spin_a=0.85, nr=56, ntheta=32, nphi=36)
cfg = GRRTRenderConfig(
    width=240,
    height=135,
    camera=LocalCamera.from_degrees(r=55, inclination_degrees=68, fov_y_degrees=32),
    step=0.055,
    max_steps=5200,
    exposure=1.4,
    workers=1,
)
img = render_grrt_image(cfg, snap, progress=True)
save_grrt_png(img, "out/grrt_volume.png")
print("wrote out/grrt_volume.png")
