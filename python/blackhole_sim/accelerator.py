"""Hardware accelerator discovery and execution planning.

This module is intentionally conservative. It does not require CUDA, Metal,
OpenVINO, ROCm, or WebGPU Python packages to be installed. Instead it detects
what appears to be available, returns an auditable execution plan, and exposes
stable names used by CLIs/tests. The native backends are optional plugins; the
CPU and browser WebGPU reference paths remain portable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    vendor: str
    api: str
    available: bool
    priority: int
    execution_mode: str
    supports_interactive: bool = False
    supports_offline: bool = True
    supports_stokes: bool = True
    supports_coeff_bricks: bool = False
    supports_tensor_units: bool = False
    reason: str = ""
    devices: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenderExecutionPlan:
    backend: str
    width: int
    height: int
    pixels: int
    tile_size: int
    tiles: int
    coefficient_bricks: bool
    precision: str
    progressive: bool
    expected_path: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _run_short(cmd: list[str], timeout: float = 1.5) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception:
        return ""
    return (out.stdout or out.stderr or "").strip()


def _windows_video_devices() -> tuple[str, ...]:
    if platform.system().lower() != "windows":
        return ()
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return ()
    cmd = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -ExpandProperty Name | "
        "ConvertTo-Json -Compress",
    ]
    raw = _run_short(cmd, timeout=2.0)
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return tuple(line.strip() for line in raw.splitlines() if line.strip())
    if isinstance(payload, str):
        return (payload,) if payload.strip() else ()
    if isinstance(payload, list):
        return tuple(str(item).strip() for item in payload if str(item).strip())
    return ()


def _os_gpu_devices() -> tuple[str, ...]:
    return _windows_video_devices()


def detect_cuda() -> BackendCapabilities:
    devices: list[str] = []
    available = False
    reason = "CUDA Python stack not detected"
    notes: list[str] = []
    if _module_available("numba.cuda") or _module_available("numba_cuda") or _module_available("cupy"):
        available = True
        reason = "CUDA Python package detected"
    if shutil.which("nvidia-smi"):
        smi = _run_short(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=2.5)
        if smi:
            devices = [line.strip() for line in smi.splitlines() if line.strip()]
            available = True
            reason = "nvidia-smi detected NVIDIA GPU"
    if _module_available("numba.cuda"):
        notes.append("numba.cuda import target available")
    if _module_available("cupy"):
        notes.append("cupy import target available")
    return BackendCapabilities(
        name="cuda",
        vendor="nvidia",
        api="CUDA / CUDA Python",
        available=available,
        priority=90,
        execution_mode="native-gpu",
        supports_interactive=False,
        supports_offline=True,
        supports_stokes=True,
        supports_coeff_bricks=True,
        supports_tensor_units=True,
        reason=reason,
        devices=tuple(devices),
        notes=tuple(notes or ["best target for offline 1080p/4K Stokes renders when a discrete NVIDIA GPU is present"]),
    )


def detect_webgpu(project_root: str | Path | None = None) -> BackendCapabilities:
    root = Path(project_root or Path.cwd()).resolve()
    shader_candidates = (
        root / "webgpu" / "src" / "grrt_volume.wgsl",
        root / "web" / "webgpu" / "src" / "grrt_volume.wgsl",
        root.parent / "web" / "webgpu" / "src" / "grrt_volume.wgsl",
    )
    shader = next((p for p in shader_candidates if p.exists()), shader_candidates[0])
    available = shader.exists()
    devices = _os_gpu_devices()
    reason = "WGSL shader files present"
    if available and devices:
        reason = "WGSL shader files present; OS GPU adapter detected"
    return BackendCapabilities(
        name="webgpu",
        vendor="browser/native-cross-vendor",
        api="WebGPU / WGSL",
        available=available,
        priority=80,
        execution_mode="browser-gpu",
        supports_interactive=True,
        supports_offline=False,
        supports_stokes=True,
        supports_coeff_bricks=True,
        reason=reason if available else "webgpu shader directory not found",
        devices=devices,
        notes=(
            "direct browser GPU compute path; WGSL dispatches through the platform backend such as D3D12, Metal, or Vulkan",
            "best portable interactive path; use coefficient bricks uploaded to GPU storage buffers",
        ),
    )


def detect_metal() -> BackendCapabilities:
    is_mac = platform.system().lower() == "darwin"
    metal_tool = shutil.which("xcrun")
    available = bool(is_mac and metal_tool)
    devices = (platform.processor() or platform.machine(),) if is_mac else ()
    return BackendCapabilities(
        name="metal",
        vendor="apple",
        api="Metal / MSL",
        available=available,
        priority=75,
        execution_mode="native-gpu",
        supports_interactive=True,
        supports_offline=True,
        supports_stokes=True,
        supports_coeff_bricks=True,
        supports_tensor_units=False,
        reason="macOS xcrun detected" if available else "not running on macOS with Xcode command-line tools",
        devices=devices,
        notes=("target for Apple Silicon; use MSL compute kernels and Metal buffers/textures",),
    )


def detect_rocm() -> BackendCapabilities:
    tools = [t for t in ("rocminfo", "hipcc") if shutil.which(t)]
    available = bool(tools or _module_available("cupy"))
    reason = "ROCm/HIP tool detected" if tools else "ROCm/HIP stack not detected"
    return BackendCapabilities(
        name="rocm",
        vendor="amd",
        api="ROCm / HIP",
        available=available,
        priority=70,
        execution_mode="native-gpu",
        supports_interactive=False,
        supports_offline=True,
        supports_stokes=True,
        supports_coeff_bricks=True,
        reason=reason,
        notes=("HIP kernel path can share most CUDA kernel structure with portability guards",),
    )


def detect_openvino() -> BackendCapabilities:
    devices: list[str] = []
    available = _module_available("openvino")
    reason = "OpenVINO package not detected"
    if available:
        reason = "OpenVINO runtime detected"
        try:
            import openvino as ov  # type: ignore
            core = ov.Core()
            devices = [str(d) for d in core.available_devices]
        except Exception as exc:  # pragma: no cover - depends on local install
            reason = f"OpenVINO import detected but device query failed: {exc}"
    return BackendCapabilities(
        name="openvino",
        vendor="intel",
        api="OpenVINO Runtime",
        available=available,
        priority=50,
        execution_mode="inference-accelerator",
        supports_interactive=False,
        supports_offline=False,
        supports_stokes=False,
        supports_coeff_bricks=False,
        supports_tensor_units=True,
        reason=reason,
        devices=tuple(devices),
        notes=(
            "use for learned coefficient surrogates or denoisers, not as the primary Kerr-ray integration kernel",
            "Intel native physics kernels should target SYCL/oneAPI or WebGPU rather than OpenVINO inference graphs",
        ),
    )


def detect_arm() -> BackendCapabilities:
    machine = platform.machine().lower()
    is_arm = any(tok in machine for tok in ("arm", "aarch64"))
    return BackendCapabilities(
        name="arm-simd",
        vendor="arm",
        api="NEON/SVE via native extension or compiler auto-vectorization",
        available=is_arm,
        priority=35,
        execution_mode="native-cpu-simd",
        supports_interactive=False,
        supports_offline=True,
        supports_stokes=True,
        supports_coeff_bricks=True,
        reason=f"machine={platform.machine()}" if is_arm else f"not ARM: machine={platform.machine()}",
        notes=("best used for coefficient-brick generation and CPU fallback on ARM servers or mobile SoCs",),
    )


def detect_cpu() -> BackendCapabilities:
    return BackendCapabilities(
        name="cpu",
        vendor="portable",
        api="NumPy/SciPy reference",
        available=True,
        priority=10,
        execution_mode="reference-cpu",
        supports_interactive=False,
        supports_offline=True,
        supports_stokes=True,
        supports_coeff_bricks=True,
        reason="always available",
        devices=(platform.processor() or platform.machine(),),
        notes=("correctness path; too slow for native full-HD final renders",),
    )


def detect_backends(project_root: str | Path | None = None) -> list[BackendCapabilities]:
    return [
        detect_cuda(),
        detect_webgpu(project_root),
        detect_metal(),
        detect_rocm(),
        detect_openvino(),
        detect_arm(),
        detect_cpu(),
    ]


def choose_backend(backends: Iterable[BackendCapabilities], prefer_interactive: bool = False) -> BackendCapabilities:
    candidates = [b for b in backends if b.available]
    if prefer_interactive:
        interactive = [b for b in candidates if b.supports_interactive]
        if interactive:
            candidates = interactive
    return sorted(candidates, key=lambda b: b.priority, reverse=True)[0]


def make_render_plan(
    width: int,
    height: int,
    backend: str = "auto",
    project_root: str | Path | None = None,
    precision: str = "float32",
    tile_size: int = 64,
    progressive: bool = True,
) -> RenderExecutionPlan:
    backends = detect_backends(project_root)
    if backend == "auto":
        chosen = choose_backend(backends, prefer_interactive=False)
    elif backend == "interactive":
        chosen = choose_backend(backends, prefer_interactive=True)
    else:
        by_name = {b.name: b for b in backends}
        chosen = by_name.get(backend, BackendCapabilities(backend, "unknown", backend, False, 0, "unknown", reason="unknown backend"))
    pixels = int(width) * int(height)
    tiles_x = (int(width) + tile_size - 1) // tile_size
    tiles_y = (int(height) + tile_size - 1) // tile_size
    warnings: list[str] = []
    if not chosen.available:
        warnings.append(f"requested backend {chosen.name!r} is not currently available: {chosen.reason}")
    if chosen.name == "openvino":
        warnings.append("OpenVINO is an inference runtime path; use it for learned coefficient surrogates, not core geodesic marching")
    if chosen.name == "cpu" and pixels >= 2_000_000:
        warnings.append("CPU reference renderer is expected to be impractical at full HD except for tiny max_steps/debug runs")
    if chosen.name in {"cuda", "metal", "rocm", "webgpu"}:
        expected = "coefficient-brick precompute + tiled GPU ray/Stokes integration"
        coeff = True
    else:
        expected = "reference CPU renderer / validation"
        coeff = chosen.supports_coeff_bricks
    return RenderExecutionPlan(
        backend=chosen.name,
        width=int(width),
        height=int(height),
        pixels=pixels,
        tile_size=int(tile_size),
        tiles=tiles_x * tiles_y,
        coefficient_bricks=coeff,
        precision=precision,
        progressive=progressive,
        expected_path=expected,
        warnings=tuple(warnings),
    )


def backends_json(project_root: str | Path | None = None) -> str:
    return json.dumps([b.to_dict() for b in detect_backends(project_root)], indent=2)


def native_platform_summary(project_root: str | Path | None = None) -> dict[str, Any]:
    from .native_loader import native_core_status
    from .platform_probe import runtime_arch_report

    backends = detect_backends(project_root)
    chosen = choose_backend(backends)
    report = runtime_arch_report()
    native = native_core_status()
    return {
        "process_arch": report["process_arch"],
        "python_arch": report["python_arch"],
        "native_core_arch": native.get("native_core_arch"),
        "native_core_loaded": native["native_core_loaded"],
        "gpu_backend": chosen.name,
        "emulation_detected": report["emulation_detected"],
        "executable": sys.executable,
    }
