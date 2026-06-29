"""Public GRMHD dump manifests, acquisition helpers, and schema verification.

This module intentionally keeps public-data acquisition separate from the
renderer. Public GRMHD dump collections are large, periodically reorganized, and
usually require a human to choose a model/spin/dump-time. The manifest captures
stable landing pages and validation requirements, while the downloader supports
explicit file URLs once a concrete dump is selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.request import urlopen

import numpy as np

from .grmhd import GRMHDSnapshot
from .grmhd_adapters import AdaptedSnapshot, load_bhac_hdf5, load_harm_hdf5, load_koral_hdf5, list_hdf5_datasets

AdapterName = Literal["harm", "koral", "bhac"]
ValidationStatus = Literal["collection_only", "selected_dump_verified"]


@dataclass(frozen=True)
class PublicDumpDescriptor:
    """Descriptor for a public dump collection or a concrete selected dump."""

    id: str
    collection: str
    landing_page: str
    citation: str
    license: str
    flux_state: str | None = None
    spin_a: float | None = None
    notes: str = ""
    adapter: AdapterName = "harm"
    direct_download_url: str | None = None
    sha256: str | None = None
    expected_suffixes: tuple[str, ...] = (".h5", ".hdf5")
    validation_status: ValidationStatus = "collection_only"
    expected_field_map: dict[str, str] = field(default_factory=dict)
    accepted_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_suffixes"] = list(self.expected_suffixes)
        d["accepted_ranges"] = {k: list(v) for k, v in self.accepted_ranges.items()}
        return d

    def selected_dump_gate_issues(self) -> tuple[str, ...]:
        """Return missing evidence for descriptors that claim a concrete dump."""

        if self.direct_download_url is None:
            return ()
        issues: list[str] = []
        if self.validation_status != "selected_dump_verified":
            issues.append("validation_status must be selected_dump_verified for direct downloads")
        if not self.sha256:
            issues.append("sha256 is required for direct downloads")
        if not self.expected_field_map:
            issues.append("expected_field_map is required for direct downloads")
        required_ranges = {"rho", "theta_e"}
        missing_ranges = sorted(required_ranges.difference(self.accepted_ranges))
        if missing_ranges:
            issues.append(f"accepted_ranges missing: {', '.join(missing_ranges)}")
        return tuple(issues)


ILLINOIS_V3_DESCRIPTOR = PublicDumpDescriptor(
    id="illinois_v3_grmhd_sane_a+0.50_selected_dump",
    collection="Illinois Simulation Data Products v3 GRMHD Output",
    landing_page="https://thz.astro.illinois.edu/v3_grmhd.html",
    citation="Dhruv, Prather, Wong & Gammie, ApJS 277, 16; arXiv:2411.12647",
    license="Creative Commons Attribution 4.0 International License, as stated by the data portal",
    flux_state="SANE",
    spin_a=0.5,
    adapter="harm",
    notes=(
        "Select a concrete v3 SANE, a*=+0.5 HDF5 dump from the portal, then pass "
        "its final file URL to blackhole-public-dump --download-url. The portal is "
        "JavaScript-filtered, so direct file URLs are intentionally not hard-coded."
    ),
    metadata={
        "parameter_survey": "SANE/MAD, spins -0.94, -0.5, 0, +0.5, +0.94",
        "recommended_analysis_package": "AFD-Illinois/pyharm",
    },
)


DEFAULT_PUBLIC_MANIFEST: tuple[PublicDumpDescriptor, ...] = (ILLINOIS_V3_DESCRIPTOR,)


def write_public_manifest(path: str | Path, descriptors: Iterable[PublicDumpDescriptor] = DEFAULT_PUBLIC_MANIFEST) -> Path:
    """Write a JSON manifest for public GRMHD data targets."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "blackhole_sim.public_grmhd_manifest.v1",
        "description": "Public GRMHD collections and selected-dump acquisition metadata.",
        "entries": [d.to_json_dict() for d in descriptors],
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def load_public_manifest(path: str | Path) -> list[PublicDumpDescriptor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[PublicDumpDescriptor] = []
    for item in payload.get("entries", []):
        item = dict(item)
        item["expected_suffixes"] = tuple(item.get("expected_suffixes", (".h5", ".hdf5")))
        item["accepted_ranges"] = {k: tuple(v) for k, v in item.get("accepted_ranges", {}).items()}
        out.append(PublicDumpDescriptor(**item))
    return out


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def download_public_dump(url: str, output: str | Path, expected_sha256: str | None = None, timeout_s: int = 120) -> Path:
    """Download a concrete public dump URL selected from a manifest landing page.

    The runtime running this function must have internet access. A SHA-256 check
    is optional but strongly preferred for reproducibility.
    """

    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout_s) as resp, p.open("wb") as f:  # noqa: S310 - user-provided scientific data URL
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
    if expected_sha256:
        actual = sha256_file(p)
        if actual.lower() != expected_sha256.lower():
            p.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {p}: expected {expected_sha256}, got {actual}")
    return p


def _load_with_adapter(path: str | Path, adapter: AdapterName) -> AdaptedSnapshot:
    if adapter == "harm":
        return load_harm_hdf5(path)
    if adapter == "koral":
        return load_koral_hdf5(path)
    if adapter == "bhac":
        return load_bhac_hdf5(path)
    raise ValueError(f"unknown adapter: {adapter}")


@dataclass(frozen=True)
class DumpVerificationReport:
    path: str
    adapter: str
    sha256: str
    datasets: dict[str, tuple[int, ...]]
    shape: tuple[int, int, int]
    spin_a: float
    time_m: float
    rho_range: tuple[float, float]
    theta_e_range: tuple[float, float]
    warnings: tuple[str, ...]
    field_map: dict[str, str]

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["datasets"] = {k: list(v) for k, v in self.datasets.items()}
        return d


def verify_public_dump(path: str | Path, adapter: AdapterName = "harm") -> DumpVerificationReport:
    """Inspect, adapt, and validate a downloaded public GRMHD HDF5 dump."""

    adapted = _load_with_adapter(path, adapter)
    snap: GRMHDSnapshot = adapted.snapshot
    return DumpVerificationReport(
        path=str(path),
        adapter=adapter,
        sha256=sha256_file(path),
        datasets=list_hdf5_datasets(path),
        shape=snap.shape,
        spin_a=float(snap.spin_a),
        time_m=float(snap.time_m),
        rho_range=(float(np.nanmin(snap.rho)), float(np.nanmax(snap.rho))),
        theta_e_range=(float(np.nanmin(snap.theta_e)), float(np.nanmax(snap.theta_e))),
        warnings=tuple(adapted.report.warnings),
        field_map=dict(adapted.report.field_map),
    )


def write_verification_report(report: DumpVerificationReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
