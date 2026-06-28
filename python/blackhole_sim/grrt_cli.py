"""CLI for rendering a Kerr black hole from a GRMHD snapshot volume."""

from __future__ import annotations

import argparse
from pathlib import Path

from .grmhd import GRMHDSnapshot, generate_analytic_grmhd_torus, load_grmhd_hdf5
from .grrt_renderer import GRRTRenderConfig, render_grrt_image, save_grrt_png
from .kerr import LocalCamera
from .radiative_transfer import TransferConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render Kerr GRRT image from a GRMHD snapshot")
    p.add_argument("--snapshot", type=Path, default=None, help="Input .npz or .h5/.hdf5 snapshot. Omit to use fixture torus.")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--spin", type=float, default=0.85, help="Used only when generating the built-in fixture")
    p.add_argument("--inclination", type=float, default=65.0)
    p.add_argument("--fov", type=float, default=32.0)
    p.add_argument("--camera-radius", type=float, default=55.0)
    p.add_argument("--step", type=float, default=0.045)
    p.add_argument("--max-steps", type=int, default=6500)
    p.add_argument("--nu", type=float, default=230.0e9, help="Observing frequency in Hz")
    p.add_argument("--path-length-scale", type=float, default=0.045)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("out/grrt_kerr.png"))
    p.add_argument("--progress", action="store_true")
    return p


def _load_snapshot(path: Path | None, spin: float) -> GRMHDSnapshot:
    if path is None:
        return generate_analytic_grmhd_torus(spin_a=spin, nr=64, ntheta=36, nphi=40)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return GRMHDSnapshot.from_npz(path)
    if suffix in {".h5", ".hdf5"}:
        return load_grmhd_hdf5(path)
    raise ValueError(f"Unsupported snapshot suffix {suffix!r}; use .npz, .h5, or .hdf5")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    snap = _load_snapshot(args.snapshot, args.spin)
    cfg = GRRTRenderConfig(
        width=args.width,
        height=args.height,
        camera=LocalCamera.from_degrees(r=args.camera_radius, inclination_degrees=args.inclination, fov_y_degrees=args.fov),
        step=args.step,
        max_steps=args.max_steps,
        exposure=args.exposure,
        workers=args.workers,
        transfer=TransferConfig(observing_frequency_hz=args.nu, path_length_scale=args.path_length_scale),
    )
    img = render_grrt_image(cfg, snap, progress=args.progress)
    save_grrt_png(img, args.output)
    print(f"wrote {args.output}; snapshot shape={snap.shape}; spin a/M={snap.spin_a:.4f}")


if __name__ == "__main__":
    main()
