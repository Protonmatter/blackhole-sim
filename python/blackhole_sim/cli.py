"""Command line interface for Kerr black-hole rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

from .kerr import LocalCamera, isco_radius
from .kerr_renderer import KerrDiskModel, KerrRenderConfig, render_kerr_image, save_png
from .physics import BlackHoleSystem


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a Kerr black-hole image with physical spin/orbit parameters")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--spin", type=float, default=0.7, help="Dimensionless Kerr spin a/M, |a| < 1")
    p.add_argument("--inclination", type=float, default=65.0, help="Degrees from spin axis; 0=pole-on, 90=edge-on")
    p.add_argument("--fov", type=float, default=34.0)
    p.add_argument("--camera-radius", type=float, default=55.0, help="Observer radius in GM/c^2 units")
    p.add_argument("--inner-radius", type=float, default=None, help="Disk inner radius in GM/c^2. Default: Kerr ISCO")
    p.add_argument("--outer-radius", type=float, default=40.0)
    p.add_argument("--retrograde", action="store_true", help="Use retrograde disk orbital motion")
    p.add_argument("--step", type=float, default=0.05, help="Affine step for geodesic integration")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--output", type=Path, default=Path("out/kerr_blackhole.png"))
    p.add_argument("--workers", type=int, default=1, help="CPU worker processes; 0 uses cpu_count-1")
    p.add_argument("--system", choices=["m87", "sgr-a", "unit"], default="unit", help="Print physical scale metadata")
    p.add_argument("--progress", action="store_true")
    return p


def _system(name: str, spin: float) -> BlackHoleSystem | None:
    if name == "m87":
        return BlackHoleSystem.m87_star(spin_a=spin)
    if name == "sgr-a":
        return BlackHoleSystem.sgr_a_star(spin_a=spin)
    return None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prograde = not args.retrograde
    inner = args.inner_radius if args.inner_radius is not None else isco_radius(args.spin, prograde=prograde)
    cfg = KerrRenderConfig(
        width=args.width,
        height=args.height,
        spin_a=args.spin,
        camera=LocalCamera.from_degrees(
            r=args.camera_radius,
            inclination_degrees=args.inclination,
            fov_y_degrees=args.fov,
        ),
        disk=KerrDiskModel(inner_radius=inner, outer_radius=args.outer_radius, prograde=prograde),
        step=args.step,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    img = render_kerr_image(cfg, progress=args.progress)
    save_png(img, args.output)
    print(f"wrote {args.output}")
    print(f"spin a/M={args.spin:.4f}; disk inner radius={inner:.6f} GM/c^2")
    sys = _system(args.system, args.spin)
    if sys is not None:
        print(f"r_g={sys.gravitational_radius_m:.6e} m; t_g={sys.gravitational_time_s:.6e} s")
        print(f"1 r_g angular size={sys.angular_rg_microarcsec:.6f} microarcsec")


if __name__ == "__main__":
    main()
