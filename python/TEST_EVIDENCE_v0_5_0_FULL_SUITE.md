# v0.5.0 Full Suite Evidence

Date: 2026-06-21
Runtime: local container, Python 3.13.5

Command:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/time -f 'elapsed=%E' python -m pytest -q --durations=36
```

Reason for `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: isolates the project test run from globally installed pytest plugins in the execution environment. The project itself only declares `pytest` under the `dev` extra.

Result:

```text
....................................                                     [100%]
============================= slowest 36 durations =============================
11.48s call     tests/test_grrt_renderer.py::test_grrt_renderer_smoke_low_resolution
5.67s call     tests/test_webgpu_regression.py::test_cpu_reference_regression_roundtrip
2.20s call     tests/test_polarized_renderer.py::test_polarized_renderer_smoke
0.42s call     tests/test_radiative_transfer.py::test_grrt_integrator_accumulates_nonnegative_intensity
0.15s call     tests/test_relativistic_plasma.py::test_vay_and_boris_agree_for_pure_b_small_dt
0.11s call     tests/test_external_validation.py::test_read_ipole_stokes_hdf5_orientation
0.06s call     tests/test_plasma.py::test_boris_preserves_speed_in_static_b_field
0.04s call     tests/test_public_dumps.py::test_verify_public_dump_canonical_hdf5
0.04s call     tests/test_radiative_transfer.py::test_invariant_redshift_positive_for_valid_fluid_sample
0.03s call     tests/test_relativistic_plasma.py::test_vay_preserves_gamma_in_static_b
0.03s call     tests/test_public_dumps.py::test_download_public_dump_sha_mismatch_file_url
0.03s call     tests/test_grmhd.py::test_periodic_phi_interpolation_matches_wrapped_sample
0.03s call     tests/test_kerr.py::test_short_geodesic_preserves_null_hamiltonian
0.02s call     tests/test_relativistic_plasma.py::test_relativistic_boris_preserves_gamma_in_static_b
0.02s call     tests/test_grmhd.py::test_npz_roundtrip
0.02s call     tests/test_geodesics.py::test_escape_above_critical_b
0.02s call     tests/test_grmhd.py::test_generate_torus_schema_and_four_velocity_normalization
0.01s call     tests/test_geodesics.py::test_capture_below_critical_b
0.01s call     tests/test_grmhd_adapters.py::test_harm_hdf5_primitive_adapter
0.01s call     tests/test_relativistic_plasma.py::test_guiding_center_matches_parallel_motion_in_uniform_b

(16 durations < 0.005s hidden.  Use -vv to show these durations.)
elapsed=0:24.51
CODE=0
```

Summary: 36 tests passed.
