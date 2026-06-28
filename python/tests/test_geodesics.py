import numpy as np

from blackhole_sim.geodesics import critical_impact_parameter, normalize, trace_null_geodesic


def test_critical_impact_parameter():
    assert np.isclose(critical_impact_parameter(1.0), np.sqrt(27.0))


def _ray_from_impact(b, r0=80.0):
    cam = np.array([r0, 0.0, 0.0])
    f = 1.0 - 2.0 / r0
    sin_alpha = b * np.sqrt(f) / r0
    cos_alpha = np.sqrt(1.0 - sin_alpha * sin_alpha)
    # Mostly inward with positive tangential component.
    direction = normalize(np.array([-cos_alpha, sin_alpha, 0.0]))
    return cam, direction


def test_capture_below_critical_b():
    cam, direction = _ray_from_impact(4.8)
    result = trace_null_geodesic(cam, direction, dphi=0.003, escape_radius=80.0, max_steps=20000)
    assert result.status == "captured"


def test_escape_above_critical_b():
    cam, direction = _ray_from_impact(6.2)
    result = trace_null_geodesic(cam, direction, dphi=0.003, escape_radius=80.0, max_steps=20000)
    assert result.status == "escaped"
