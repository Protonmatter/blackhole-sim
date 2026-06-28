from pathlib import Path

import numpy as np
import pytest

from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.public_dumps import (
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
