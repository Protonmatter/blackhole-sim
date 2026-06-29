"""CLI for deterministic hot-loop benchmark probes."""

from __future__ import annotations

import argparse
import json

from .benchmark import coefficient_brick_benchmark


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run deterministic blackhole-sim hot-loop benchmark probes")
    p.add_argument("--nr", type=int, default=8)
    p.add_argument("--ntheta", type=int, default=6)
    p.add_argument("--nphi", type=int, default=8)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--dtype", choices=("float32", "float16", "float64"), default="float32")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = coefficient_brick_benchmark(
        nr=args.nr,
        ntheta=args.ntheta,
        nphi=args.nphi,
        iterations=args.iterations,
        dtype=args.dtype,
    )
    payload = result.to_json_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"{result.name}: {result.items} cells in {result.seconds:.6f}s "
            f"({result.items_per_second:.2f} cells/s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
