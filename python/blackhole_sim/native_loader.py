"""Optional native-core loader for architecture-gated releases."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from .platform_probe import normalize_arch


def _load_native_module() -> ModuleType | None:
    try:
        return importlib.import_module("blackhole_native")
    except Exception:
        return None


def load_native_module() -> ModuleType | None:
    """Return the optional native module when importable, otherwise ``None``."""

    return _load_native_module()


def _module_path(module: ModuleType) -> str:
    value = getattr(module, "__file__", None)
    return str(Path(value).resolve()) if value else ""


def native_core_status() -> dict[str, Any]:
    module = _load_native_module()
    if module is None:
        return {
            "native_core_loaded": False,
            "native_core_arch": None,
            "native_core_version": None,
            "module_path": None,
            "reason": "blackhole_native module not importable",
        }

    arch_info: dict[str, Any] = {}
    try:
        raw = module.detect_arch()
        if isinstance(raw, dict):
            arch_info = raw
    except Exception as exc:
        arch_info = {"error": str(exc)}

    try:
        version = str(module.core_version())
    except Exception:
        version = None

    native_arch = normalize_arch(str(arch_info.get("arch", "")))
    return {
        "native_core_loaded": True,
        "native_core_arch": native_arch,
        "native_core_version": version,
        "module_path": _module_path(module),
        "raw_arch": arch_info,
        "reason": "loaded",
    }
