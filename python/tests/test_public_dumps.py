from pathlib import Path

import numpy as np
import pytest

from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.public_dumps import (
    DEFAULT_PUBLIC_MANIFEST,
    PublicDumpDescriptor,
    download_public_dump,
    load_public_manifest,
    sha256_file,
    verify_public_dump,
    write_public_manifest,
)


def test_public_manifest_round_trip(tmp_path: Path):
    manifest = write_public_manifest(tmp_path / "manifest.json")
    entries = load_public_manifest(manifest)
    assert entries
    assert entries[0].collection.startswith("Illinois")
    assert entries[0].adapter == "harm"
    assert entries[0].landing_page.startswith("https://")
    assert entries[0].validation_status == "collection_only"


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_download_public_dump_sha_mismatch_file_url(tmp_path: Path):
    src = tmp_path / "src.h5"
    src.write_bytes(b"not a real dump")
    with pytest.raises(ValueError):
        download_public_dump(src.as_uri(), tmp_path / "out.h5", expected_sha256="0" * 64)


def test_verify_public_dump_canonical_hdf5(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    snap = generate_analytic_grmhd_torus(nr=6, ntheta=5, nphi=4, spin_a=0.5)
    path = tmp_path / "fixture.h5"
    snap.to_hdf5(path)
    report = verify_public_dump(path, adapter="harm")
    assert report.shape == snap.shape
    assert report.adapter == "harm"
    assert report.spin_a == pytest.approx(0.5)
    assert "rho" in report.field_map


def test_collection_only_manifest_does_not_claim_selected_dump_validation():
    for entry in DEFAULT_PUBLIC_MANIFEST:
        assert entry.direct_download_url is None
        assert entry.validation_status == "collection_only"
        assert entry.selected_dump_gate_issues() == ()


def test_selected_dump_descriptor_requires_reproducibility_evidence():
    descriptor = PublicDumpDescriptor(
        id="example",
        collection="Example",
        landing_page="https://example.com",
        citation="example",
        license="example",
        direct_download_url="https://example.com/dump.h5",
        validation_status="selected_dump_verified",
    )
    issues = descriptor.selected_dump_gate_issues()
    assert any("sha256" in issue for issue in issues)
    assert any("expected_field_map" in issue for issue in issues)
    assert any("accepted_ranges" in issue for issue in issues)

    verified = PublicDumpDescriptor(
        id="example",
        collection="Example",
        landing_page="https://example.com",
        citation="example",
        license="example",
        direct_download_url="https://example.com/dump.h5",
        sha256="a" * 64,
        validation_status="selected_dump_verified",
        expected_field_map={"rho": "rho", "theta_e": "theta_e", "u_con": "u_con", "b_con": "b_con"},
        accepted_ranges={"rho": (1.0e-12, 10.0), "theta_e": (1.0e-3, 1.0e3)},
    )
    assert verified.selected_dump_gate_issues() == ()
