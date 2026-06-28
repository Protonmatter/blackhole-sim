"""CLI for the v0.7 coefficient-brick hot-loop renderer.

This is a CPU-emulated execution path for the same algorithmic kernel used by
WGSL/CUDA/Metal/HIP. It is intended for small regression images and offline
validation on machines without a GPU runtime installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .accelerated_renderer import AcceleratedRenderConfig, render_progressive_stokes_bricks, render_stokes_image_bricks, save_progressive_npz, save_progressive_preview
from .calibration import PhysicalScaling
from .coefficient_bricks import CoefficientBrickGrid, precompute_coefficient_bricks
from .grmhd import GRMHDSnapshot, generate_analytic_grmhd_torus
from .kerr import LocalCamera
from .physics import BlackHoleSystem
from .polarized_renderer import save_stokes_npz, save_stokes_preview_png


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render with the v0.7 coefficient-brick hot loop")
    p.add_argument("--snapshot", type=str, default="", help="optional native GRMHD .npz snapshot")
    p.add_argument("--bricks", type=str, default="", help="optional precomputed coefficient brick .npz")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=36)
    p.add_argument("--spin", type=float, default=0.85)
    p.add_argument("--inclination", type=float, default=65.0)
    p.add_argument("--camera-radius", type=float, default=55.0)
    p.add_argument("--step", type=float, default=0.08)
    p.add_argument("--max-steps", type=int, default=700)
    p.add_argument("--progressive", action="store_true")
    p.add_argument("--output", type=str, default="out/accelerated_stokes.npz")
    p.add_argument("--preview", type=str, default="")
    args = p.parse_args(argv)

    if args.snapshot:
        snap = GRMHDSnapshot.from_npz(args.snapshot)
    else:
        snap = generate_analytic_grmhd_torus(spin_a=args.spin, nr=18, ntheta=12, nphi=12)
    if args.bricks:
        bricks = CoefficientBrickGrid.load_npz(args.bricks)
    else:
        scaling = PhysicalScaling.from_mdot(BlackHoleSystem.sgr_a_star(snap.spin_a), 1e-8)
        bricks = precompute_coefficient_bricks(snap, scaling, dtype="float32", stride=1)

    cfg = AcceleratedRenderConfig(
        width=args.width,
        height=args.height,
        camera=LocalCamera.from_degrees(r=args.camera_radius, inclination_degrees=args.inclination),
        step=args.step,
        max_steps=args.max_steps,
    )
    if args.progressive:
        frames = render_progressive_stokes_bricks(cfg, snap, bricks)
        for frame in frames:
            stem = Path(args.output).with_suffix("")
            save_progressive_npz(frame, f"{stem}_level{frame.level}.npz")
        final = frames[-1]
        save_progressive_npz(final, args.output)
        if args.preview:
            save_progressive_preview(final, args.preview)
    else:
        img = render_stokes_image_bricks(cfg, snap, bricks)
        save_stokes_npz(img, args.output)
        if args.preview:
            save_stokes_preview_png(img, args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
