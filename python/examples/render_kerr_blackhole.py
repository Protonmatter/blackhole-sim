from blackhole_sim import LocalCamera, KerrDiskModel, KerrRenderConfig, render_kerr_image
from blackhole_sim.kerr import isco_radius
from blackhole_sim.kerr_renderer import save_png

spin = 0.85
prograde = True
inner = isco_radius(spin, prograde=prograde)

cfg = KerrRenderConfig(
    width=480,
    height=270,
    spin_a=spin,
    camera=LocalCamera.from_degrees(r=60.0, inclination_degrees=68.0, fov_y_degrees=32.0),
    disk=KerrDiskModel(inner_radius=inner, outer_radius=42.0, prograde=prograde),
    step=0.045,
    max_steps=5200,
    exposure=1.15,
)

img = render_kerr_image(cfg, progress=True)
save_png(img, "out/kerr_blackhole_480x270.png")
