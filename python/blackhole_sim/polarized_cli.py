"""CLI for polarized Kerr+GRMHD Stokes rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import PhysicalScaling
from .grmhd import GRMHDSnapshot, generate_analytic_grmhd_torus, load_grmhd_hdf5
from .grmhd_adapters import load_bhac_hdf5, load_harm_hdf5, load_koral_hdf5
from .kerr import LocalCamera
from .physics import BlackHoleSystem
from .polarized_renderer import PolarizedRenderConfig, render_stokes_image, save_stokes_npz, save_stokes_preview_png
from .polarized_transfer import PolarizedTransferConfig


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render Stokes I,Q,U,V image from Kerr+GRMHD snapshot")
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--adapter", choices=["native", "harm", "koral", "bhac"], default="native")
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--height", type=int, default=54)
    p.add_argument("--spin", type=float, default=0.85)
    p.add_argument("--inclination", type=float, default=65.0)
    p.add_argument("--fov", type=float, default=32.0)
    p.add_argument("--camera-radius", type=float, default=55.0)
    p.add_argument("--step", type=float, default=0.08)
    p.add_argument("--max-steps", type=int, default=3500)
    p.add_argument("--nu", type=float, default=230e9)
    p.add_argument("--mass", choices=["sgr-a", "m87"], default="m87")
    p.add_argument("--mdot-msun-year", type=float, default=1e-4)
    p.add_argument("--rho-scale", type=float, default=None, help="Override rho code unit in g/cm^3")
    p.add_argument("--b-scale", type=float, default=None, help="Override B code unit in Gauss")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("out/stokes.npz"))
    p.add_argument("--preview", type=Path, default=Path("out/stokes_preview.png"))
    p.add_argument("--progress", action="store_true")
    return p


def load_snapshot(path: Path | None, adapter: str, spin: float) -> GRMHDSnapshot:
    if path is None:
        return generate_analytic_grmhd_torus(spin_a=spin, nr=48, ntheta=28, nphi=32)
    if adapter == "harm":
        return load_harm_hdf5(path).snapshot
    if adapter == "koral":
        return load_koral_hdf5(path).snapshot
    if adapter == "bhac":
        return load_bhac_hdf5(path).snapshot
    if path.suffix.lower() == ".npz":
        return GRMHDSnapshot.from_npz(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_grmhd_hdf5(path)
    raise ValueError("Unsupported snapshot file")


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    snap = load_snapshot(args.snapshot, args.adapter, args.spin)
    system = BlackHoleSystem.m87_star(snap.spin_a) if args.mass == "m87" else BlackHoleSystem.sgr_a_star(snap.spin_a)
    if args.rho_scale is not None and args.b_scale is not None:
        scaling = PhysicalScaling.from_black_hole_system(system, args.rho_scale, args.b_scale)
    else:
        scaling = PhysicalScaling.from_mdot(system, args.mdot_msun_year)
    cfg = PolarizedRenderConfig(
        width=args.width,
        height=args.height,
        camera=LocalCamera.from_degrees(args.camera_radius, args.inclination, fov_y_degrees=args.fov),
        step=args.step,
        max_steps=args.max_steps,
        workers=args.workers,
        transfer=PolarizedTransferConfig(observing_frequency_hz=args.nu),
    )
    img = render_stokes_image(cfg, snap, scaling, args.progress)
    save_stokes_npz(img, args.output)
    save_stokes_preview_png(img, args.preview)
    print(f"wrote {args.output} and {args.preview}; stokes_shape={img.shape}; spin={snap.spin_a:.4f}")


if __name__ == "__main__":
    main()
