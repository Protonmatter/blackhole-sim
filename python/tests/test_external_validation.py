from pathlib import Path

import numpy as np
import pytest

from blackhole_sim.external_validation import (
    IpoleRunConfig,
    compare_stokes_images,
    read_ipole_stokes_hdf5,
    read_our_stokes_npz,
    write_comparison_report,
    write_ipole_parameter_file,
)


def test_write_ipole_parameter_file(tmp_path: Path):
    cfg = IpoleRunConfig(dump_path="dump.h5", output_path="image.h5", nx=8, ny=6, extra_parameters={"unpol": 0})
    path = write_ipole_parameter_file(cfg, tmp_path / "ipole.par")
    text = path.read_text()
    assert "dump dump.h5" in text
    assert "outfile image.h5" in text
    assert "nx 8" in text
    assert "unpol 0" in text


def test_read_ipole_stokes_hdf5_orientation(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "ipole_image.h5"
    raw = np.zeros((3, 2, 5), dtype=float)
    raw[..., 0] = np.arange(6).reshape(3, 2)
    raw[..., 1] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("pol", data=raw)
        header = f.create_group("header")
        header.create_dataset("scale", data=2.0)
    arr = read_ipole_stokes_hdf5(path, jy_per_pixel=True)
    assert arr.shape == (2, 3, 4)
    assert arr[0, 0, 0] == pytest.approx(0.0)
    assert arr[1, 2, 0] == pytest.approx(10.0)
    assert np.all(arr[..., 1] == 2.0)


def test_compare_stokes_images_and_report(tmp_path: Path):
    a = np.ones((3, 4, 4))
    b = np.ones((3, 4, 4)) * 1.1
    metrics = compare_stokes_images(a, b, rtol_l1=0.2)
    assert metrics.passed
    assert metrics.shape == (3, 4, 4)
    out = write_comparison_report(metrics, tmp_path / "report.json")
    assert out.exists()


def test_read_our_stokes_npz(tmp_path: Path):
    path = tmp_path / "ours.npz"
    stokes = np.zeros((2, 2, 4))
    np.savez(path, stokes=stokes)
    assert read_our_stokes_npz(path).shape == (2, 2, 4)
