from blackhole_sim import Camera, DiskModel, RenderConfig, render_image
from blackhole_sim.renderer import save_png

cfg = RenderConfig(
    width=640,
    height=360,
    camera=Camera(radius=36.0, inclination_degrees=72.0, fov_degrees=40.0),
    disk=DiskModel(inner_radius=6.0, outer_radius=26.0, ring_radius=8.0),
    dphi=0.0028,
)

img = render_image(cfg, progress=True)
save_png(img, "out/blackhole_640x360.png")
