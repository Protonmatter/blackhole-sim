import json

from blackhole_sim.benchmark import (
    benchmark_suite,
    coefficient_brick_benchmark,
    sample_brick_trilinear_parity_benchmark,
    stokes_rk2_brick_parity_benchmark,
)
from blackhole_sim.benchmark_cli import main as benchmark_main


def test_coefficient_brick_benchmark_reports_architecture_metadata():
    result = coefficient_brick_benchmark(nr=3, ntheta=3, nphi=3, iterations=1)
    payload = result.to_json_dict()

    assert payload["name"] == "coefficient_brick_precompute"
    assert payload["items"] == 27
    assert payload["seconds"] > 0.0
    assert payload["items_per_second"] > 0.0
    assert payload["metadata"]["grid"] == [3, 3, 3]
    assert "process_arch" in payload["metadata"]


def test_stokes_rk2_brick_parity_benchmark_reports_reference_and_native_status():
    payload = stokes_rk2_brick_parity_benchmark(nr=3, ntheta=3, nphi=3, iterations=1)

    assert payload["name"] == "stokes_rk2_brick_parity"
    assert payload["reference"]["name"] == "stokes_rk2_brick_reference"
    assert payload["reference"]["items"] == 27
    assert payload["parity"]["rtol"] > 0.0
    assert payload["parity"]["atol"] > 0.0
    assert "native_available" in payload["parity"]


def test_sample_brick_trilinear_parity_benchmark_reports_reference_and_native_status():
    payload = sample_brick_trilinear_parity_benchmark(nr=3, ntheta=3, nphi=4, point_count=7, iterations=1)

    assert payload["name"] == "sample_brick_trilinear_parity"
    assert payload["reference"]["name"] == "sample_brick_trilinear_reference"
    assert payload["reference"]["items"] == 7
    assert payload["reference"]["metadata"]["point_count"] == 7
    assert payload["metadata"]["grid"] == [3, 3, 4]
    assert payload["parity"]["rtol"] == 1.0e-12
    assert payload["parity"]["atol"] == 1.0e-12
    assert "native_available" in payload["parity"]


def test_benchmark_suite_includes_reference_native_parity_payload():
    payload = benchmark_suite(nr=3, ntheta=3, nphi=4, point_count=7, iterations=1)

    names = [item["name"] for item in payload["benchmarks"]]
    assert payload["schema"] == "blackhole_sim.benchmark.v2"
    assert "coefficient_brick_precompute" in names
    assert "sample_brick_trilinear_parity" in names
    assert "stokes_rk2_brick_parity" in names


def test_benchmark_cli_json_reports_suite(capsys):
    rc = benchmark_main(["--json", "--nr", "3", "--ntheta", "3", "--nphi", "4", "--points", "7", "--iterations", "1"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "blackhole_sim.benchmark.v2"
    assert any(item["name"] == "sample_brick_trilinear_parity" for item in payload["benchmarks"])
    assert any(item["name"] == "stokes_rk2_brick_parity" for item in payload["benchmarks"])


def test_benchmark_cli_json_reports_sampler_target(capsys):
    rc = benchmark_main(
        [
            "--target",
            "sample-brick-trilinear",
            "--json",
            "--nr",
            "3",
            "--ntheta",
            "3",
            "--nphi",
            "4",
            "--points",
            "7",
            "--iterations",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["name"] == "sample_brick_trilinear_parity"
    assert payload["reference"]["name"] == "sample_brick_trilinear_reference"
