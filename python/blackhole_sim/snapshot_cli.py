"""CLI for generating a local GRMHD-like fixture snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from .grmhd import assert_four_velocity_normalization, generate_analytic_grmhd_torus


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a deterministic GRMHD-schema torus fixture")
    p.add_argument("--output", type=Path, default=Path("out/torus_fixture.npz"))
    p.add_argument("--format", choices=["npz", "hdf5"], default="npz")
    p.add_argument("--spin", type=float, default=0.85)
    p.add_argument("--nr", type=int, default=72)
    p.add_argument("--ntheta", type=int, default=40)
    p.add_argument("--nphi", type=int, default=48)
    p.add_argument("--r-max", type=float, default=60.0)
    p.add_argument("--time", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    snap = generate_analytic_grmhd_torus(
        spin_a=args.spin,
        nr=args.nr,
        ntheta=args.ntheta,
        nphi=args.nphi,
        r_max=args.r_max,
        time_m=args.time,
    )
    err = assert_four_velocity_normalization(snap, samples=256)
    if args.format == "hdf5":
        snap.to_hdf5(args.output)
    else:
        snap.to_npz(args.output)
    print(f"wrote {args.output}; shape={snap.shape}; max |u_mu u^mu + 1|={err:.3e}")


if __name__ == "__main__":
    main()
