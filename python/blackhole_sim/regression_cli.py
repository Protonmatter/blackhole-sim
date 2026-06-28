"""CLI for deterministic CPU/WebGPU regression artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .webgpu_regression import compare_with_reference, deterministic_cpu_reference, ensure_shader_contains_regression_hooks, save_reference_npz


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Create or compare black-hole renderer regression images")
    p.add_argument("--reference", type=Path, default=Path("out/cpu_reference.npz"))
    p.add_argument("--create", action="store_true")
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--height", type=int, default=14)
    p.add_argument("--shader", type=Path, default=Path("webgpu/src/grrt_volume.wgsl"))
    args = p.parse_args(argv)
    ensure_shader_contains_regression_hooks(args.shader)
    img = deterministic_cpu_reference(args.width, args.height)
    if args.create or not args.reference.exists():
        save_reference_npz(args.reference, img, {"width": args.width, "height": args.height})
        print(f"wrote reference {args.reference}")
        return
    metrics = compare_with_reference(img, args.reference)
    print(metrics)
    if not metrics.passed_default:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
