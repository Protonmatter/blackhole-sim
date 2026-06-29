"""CLI for deterministic hot-loop benchmark probes."""

from __future__ import annotations

import argparse
import json

from .benchmark import benchmark_suite, coefficient_brick_benchmark, stokes_rk2_brick_parity_benchmark


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run deterministic blackhole-sim hot-loop benchmark probes")
    p.add_argument("--nr", type=int, default=8)
    p.add_argument("--ntheta", type=int, default=6)
    p.add_argument("--nphi", type=int, default=8)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--dtype", choices=("float32", "float16", "float64"), default="float32")
    p.add_argument("--target", choices=("all", "coefficient-brick", "stokes-rk2-brick"), default="all")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.target == "all":
        payload = benchmark_suite(
            nr=args.nr,
            ntheta=args.ntheta,
            nphi=args.nphi,
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
                if item["name"] == "stokes_rk2_brick_parity":
                    ref = item["reference"]
                    native = item["native"]
                    print(f"{ref['name']}: {ref['items']} cells in {ref['seconds']:.6f}s")
                    if native is None:
                        print("stokes_rk2_brick_native: unavailable; Python fallback active")
                    else:
                        print(f"{native['name']}: {native['items']} cells in {native['seconds']:.6f}s")
                else:
                    print(f"{item['name']}: {item['items']} cells in {item['seconds']:.6f}s")
        elif args.target == "stokes-rk2-brick":
            ref = payload["reference"]
            native = payload["native"]
            print(f"{ref['name']}: {ref['items']} cells in {ref['seconds']:.6f}s")
            if native is None:
                print("stokes_rk2_brick_native: unavailable; Python fallback active")
            else:
                print(f"{native['name']}: {native['items']} cells in {native['seconds']:.6f}s")
        else:
            print(
                f"{payload['name']}: {payload['items']} cells in {payload['seconds']:.6f}s "
                f"({payload['items_per_second']:.2f} cells/s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
