import numpy as np
import pytest

from blackhole_sim.grmhd import assert_four_velocity_normalization
from blackhole_sim.grmhd_adapters import load_harm_hdf5, list_hdf5_datasets


def test_harm_hdf5_primitive_adapter(tmp_path):
    h5py = pytest.importorskip("h5py")
    r = np.linspace(3.0, 8.0, 5)
    th = np.linspace(0.4, 2.7, 4)
    ph = np.linspace(0.0, 2*np.pi, 3, endpoint=False)
    prims = np.zeros((5,4,3,8))
    prims[...,0] = 1.0
    prims[...,1] = 0.2
    prims[...,2] = 0.01
    prims[...,3] = 0.0
    prims[...,4] = 0.02
    prims[...,5] = 0.01
    prims[...,6] = 0.02
    prims[...,7] = 0.03
    path = tmp_path / 'mock_harm.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['spin_a'] = 0.3
        f.create_dataset('x1', data=r)
        f.create_dataset('x2', data=th)
        f.create_dataset('x3', data=ph)
        f.create_dataset('prims', data=prims)
    assert 'prims' in list_hdf5_datasets(path)
    adapted = load_harm_hdf5(path)
    snap = adapted.snapshot
    assert snap.shape == (5,4,3)
    assert snap.spin_a == 0.3
    assert 'u_con' in adapted.report.field_map
    assert assert_four_velocity_normalization(snap, samples=10) < 1e-10
