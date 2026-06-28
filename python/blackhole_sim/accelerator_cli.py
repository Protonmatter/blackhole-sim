"""CLI for accelerator discovery and render planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .accelerator import detect_backends, make_render_plan
from .platform_probe import doctor_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect GPU/native accelerator options for blackhole-sim")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="list detected accelerator backends")
    s.add_argument("--project-root", type=Path, default=Path.cwd())
    s.add_argument("--json", action="store_true")
    q = sub.add_parser("plan", help="build a render execution plan")
    q.add_argument("--width", type=int, default=1920)
    q.add_argument("--height", type=int, default=1080)
    q.add_argument("--backend", default="auto", choices=["auto", "interactive", "cpu", "webgpu", "cuda", "metal", "rocm", "openvino", "arm-simd"])
    q.add_argument("--tile-size", type=int, default=64)
    q.add_argument("--precision", default="float32", choices=["float32", "float16", "float64"])
    q.add_argument("--project-root", type=Path, default=Path.cwd())
    q.add_argument("--json", action="store_true")
    d = sub.add_parser("doctor", help="report architecture, native-core, backend, and emulation status")
    d.add_argument("--project-root", type=Path, default=Path.cwd())
    d.add_argument("--json", action="store_true")
    d.add_argument("--fail-on-emulation", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "list":
        backends = detect_backends(args.project_root)
        if args.json:
            print(json.dumps([b.to_dict() for b in backends], indent=2))
            return 0
        for b in backends:
            status = "available" if b.available else "missing"
            print(f"{b.name:10s} {status:9s} {b.api} :: {b.reason}")
            if b.devices:
                print(f"  devices: {', '.join(b.devices)}")
            for note in b.notes:
                print(f"  - {note}")
        return 0
    if args.cmd == "plan":
        plan = make_render_plan(args.width, args.height, args.backend, args.project_root, args.precision, args.tile_size)
        if args.json:
            print(json.dumps(plan.to_dict(), indent=2))
            return 0
        print(f"backend={plan.backend} pixels={plan.pixels} tiles={plan.tiles} tile_size={plan.tile_size}")
        print(f"precision={plan.precision} coefficient_bricks={plan.coefficient_bricks} progressive={plan.progressive}")
        print(f"path={plan.expected_path}")
        for w in plan.warnings:
            print(f"warning: {w}")
        return 0
    if args.cmd == "doctor":
        report = doctor_report(args.project_root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"process_arch: {report['process_arch']}")
            print(f"python_arch: {report['python_arch']}")
            print(f"native_core_arch: {report['native_core_arch']}")
            print(f"native_core_loaded: {str(report['native_core_loaded']).lower()}")
            print(f"gpu_backend: {report['gpu_backend']}")
            print(f"emulation_detected: {str(report['emulation_detected']).lower()}")
            for warning in report["warnings"]:
                print(f"warning: {warning}")
        if args.fail_on_emulation and report["emulation_detected"]:
            return 2
        native_arch = report.get("native_core_arch")
        if args.fail_on_emulation and report["native_core_loaded"] and native_arch not in {None, "unknown", report["process_arch"]}:
            return 3
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
