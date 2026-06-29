from blackhole_sim.benchmark import coefficient_brick_benchmark


def test_coefficient_brick_benchmark_reports_architecture_metadata():
    result = coefficient_brick_benchmark(nr=3, ntheta=3, nphi=3, iterations=1)
    payload = result.to_json_dict()

    assert payload["name"] == "coefficient_brick_precompute"
    assert payload["items"] == 27
    assert payload["seconds"] > 0.0
    assert payload["items_per_second"] > 0.0
    assert payload["metadata"]["grid"] == [3, 3, 3]
    assert "process_arch" in payload["metadata"]
