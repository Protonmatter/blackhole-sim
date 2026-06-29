"""CLI for deterministic hot-loop benchmark probes."""

from __future__ import annotations

import argparse
import json

from .benchmark import (
    benchmark_suite,
    coefficient_brick_benchmark,
    sample_brick_trilinear_parity_benchmark,
    stokes_rk2_brick_parity_benchmark,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run deterministic blackhole-sim hot-loop benchmark probes")
    p.add_argument("--nr", type=int, default=8)
    p.add_argument("--ntheta", type=int, default=6)
    p.add_argument("--nphi", type=int, default=8)
    p.add_argument("--points", type=int, default=64)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--dtype", choices=("float32", "float16", "float64"), default="float32")
    p.add_argument(
        "--target",
        choices=("all", "coefficient-brick", "sample-brick-trilinear", "stokes-rk2-brick"),
        default="all",
    )
    p.add_argument("--json", action="store_true")
    return p


def _print_parity_payload(payload: dict) -> None:
    ref = payload["reference"]
    native = payload["native"]
    print(f"{ref['name']}: {ref['items']} items in {ref['seconds']:.6f}s")
    if native is None:
        print(f"{payload['metadata']['workload']}_native: unavailable; Python fallback active")
    else:
        print(f"{native['name']}: {native['items']} items in {native['seconds']:.6f}s")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.target == "all":
        payload = benchmark_suite(
            nr=args.nr,
            ntheta=args.ntheta,
            nphi=args.nphi,
            point_count=args.points,
            iterations=args.iterations,
            dtype=args.dtype,
        )
    elif args.target == "coefficient-brick":
        result = coefficient_brick_benchmark(
            nr=args.nr,
            ntheta=args.ntheta,
            nphi=args.nphi,
            iterations=args.iterations,
            dtype=args.dtype,
        )
        payload = result.to_json_dict()
    elif args.target == "sample-brick-trilinear":
        payload = sample_brick_trilinear_parity_benchmark(
            nr=args.nr,
            ntheta=args.ntheta,
            nphi=args.nphi,
            point_count=args.points,
            iterations=args.iterations,
        )
    else:
        payload = stokes_rk2_brick_parity_benchmark(
            nr=args.nr,
            ntheta=args.ntheta,
            nphi=args.nphi,
            iterations=args.iterations,
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if args.target == "all":
            for item in payload["benchmarks"]:
                if item["name"] in {"sample_brick_trilinear_parity", "stokes_rk2_brick_parity"}:
                    _print_parity_payload(item)
                else:
                    print(f"{item['name']}: {item['items']} cells in {item['seconds']:.6f}s")
        elif args.target in {"sample-brick-trilinear", "stokes-rk2-brick"}:
            _print_parity_payload(payload)
        else:
            print(
                f"{payload['name']}: {payload['items']} cells in {payload['seconds']:.6f}s "
                f"({payload['items_per_second']:.2f} cells/s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
