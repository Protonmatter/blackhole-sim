"""Runtime architecture and emulation probes for release gates."""

from __future__ import annotations

import os
import platform
import struct
import sys
from pathlib import Path
from typing import Any


def normalize_arch(value: str | None) -> str:
    text = (value or "").strip().lower().replace("-", "_")
    if text in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if text in {"arm64", "aarch64"}:
        return "arm64"
    if text in {"x86", "i386", "i686"}:
        return "x86"
    if text.startswith("armv7") or text == "arm":
        return "arm"
    return text or "unknown"


def _windows_env_arch() -> tuple[str, str]:
    return (
        normalize_arch(os.environ.get("PROCESSOR_ARCHITECTURE")),
        normalize_arch(os.environ.get("PROCESSOR_ARCHITEW6432")),
    )


def runtime_arch_report() -> dict[str, Any]:
    machine = normalize_arch(platform.machine())
    processor_architecture, processor_architew6432 = _windows_env_arch()
    pointer_bits = struct.calcsize("P") * 8
    process_arch = normalize_arch(platform.machine())

    if platform.system().lower() == "windows" and processor_architecture != "unknown":
        process_arch = processor_architecture

    emulation_reasons: list[str] = []
    if machine == "arm64" and process_arch in {"x86", "x86_64"}:
        emulation_reasons.append("ARM64 host appears to be running an x86/x64 process")
    if machine == "arm64" and processor_architew6432 in {"x86", "x86_64"}:
        emulation_reasons.append("PROCESSOR_ARCHITEW6432 indicates x86/x64 compatibility mode")
    if processor_architecture in {"x86", "x86_64"} and processor_architew6432 == "arm64":
        emulation_reasons.append("Windows ARM64 compatibility process detected")

    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "process_arch": process_arch,
        "python_arch": normalize_arch(platform.machine()),
        "pointer_bits": pointer_bits,
        "processor_architecture": os.environ.get("PROCESSOR_ARCHITECTURE"),
        "processor_architew6432": os.environ.get("PROCESSOR_ARCHITEW6432"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cwd": str(Path.cwd()),
        "emulation_detected": bool(emulation_reasons),
        "emulation_reasons": tuple(emulation_reasons),
    }


def doctor_report(project_root: str | Path | None = None) -> dict[str, Any]:
    from .accelerator import detect_backends, choose_backend
    from .native_loader import native_core_status

    arch = runtime_arch_report()
    native = native_core_status()
    backends = detect_backends(project_root)
    chosen = choose_backend(backends)
    warnings: list[str] = []

    if not native["native_core_loaded"]:
        warnings.append("native core extension is not installed; Python fallback remains active")
    elif native.get("native_core_arch") not in {None, "unknown", arch["process_arch"]}:
        warnings.append("native core architecture does not match the current process architecture")
    if arch["emulation_detected"]:
        warnings.extend(str(r) for r in arch["emulation_reasons"])

    return {
        "process_arch": arch["process_arch"],
        "python_arch": arch["python_arch"],
        "native_core_arch": native.get("native_core_arch"),
        "native_core_loaded": native["native_core_loaded"],
        "native_core_version": native.get("native_core_version"),
        "gpu_backend": chosen.name,
        "emulation_detected": arch["emulation_detected"],
        "runtime": arch,
        "native": native,
        "backends": [b.to_dict() for b in backends],
        "warnings": tuple(warnings),
    }
