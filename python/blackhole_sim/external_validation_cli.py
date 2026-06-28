"""CLI to prepare and evaluate external ipole validation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .external_validation import (
    IpoleRunConfig,
    compare_stokes_images,
    read_ipole_stokes_hdf5,
    read_our_stokes_npz,
    run_ipole,
    write_comparison_report,
    write_ipole_parameter_file,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate blackhole-sim Stokes output against ipole output")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("write-ipole-par")
    prep.add_argument("--dump", required=True)
    prep.add_argument("--outfile", default="out/ipole_image.h5")
    prep.add_argument("--par", type=Path, default=Path("out/ipole_validation.par"))
    prep.add_argument("--nx", type=int, default=64)
    prep.add_argument("--ny", type=int, default=64)
    prep.add_argument("--freq-hz", type=float, default=230e9)
    prep.add_argument("--mbh-msun", type=float, default=6.2e9)
    prep.add_argument("--m-unit-g", type=float, default=1e25)
    prep.add_argument("--thetacam", type=float, default=17.0)
    prep.add_argument("--fov-muas", type=float, default=160.0)

    run = sub.add_parser("run-ipole")
    run.add_argument("--ipole", required=True)
    run.add_argument("--par", required=True, type=Path)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--ours", required=True, type=Path)
    cmp_p.add_argument("--ipole", required=True, type=Path)
    cmp_p.add_argument("--jy-per-pixel", action="store_true")
    cmp_p.add_argument("--rtol-l1", type=float, default=0.2)
    cmp_p.add_argument("--report", type=Path, default=Path("out/ipole_comparison.json"))

    args = p.parse_args(argv)
    if args.cmd == "write-ipole-par":
        cfg = IpoleRunConfig(
            dump_path=args.dump,
            output_path=args.outfile,
            freq_hz=args.freq_hz,
            mbh_msun=args.mbh_msun,
            m_unit_g=args.m_unit_g,
            camera_theta_deg=args.thetacam,
            fov_muas=args.fov_muas,
            nx=args.nx,
            ny=args.ny,
        )
        path = write_ipole_parameter_file(cfg, args.par)
        print(f"wrote {path}")
    elif args.cmd == "run-ipole":
        proc = run_ipole(args.ipole, args.par)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
    elif args.cmd == "compare":
        ours = read_our_stokes_npz(args.ours)
        ref = read_ipole_stokes_hdf5(args.ipole, jy_per_pixel=args.jy_per_pixel)
        metrics = compare_stokes_images(ours, ref, rtol_l1=args.rtol_l1)
        write_comparison_report(metrics, args.report)
        print(metrics)
        if not metrics.passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
