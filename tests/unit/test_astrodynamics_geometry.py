import numpy as np
import pytest

from aerognc.astrodynamics.geometry import (
    SphericalBodyGeometry,
    SphericalGroundStation,
    eclipse_state,
    ground_station_access,
    line_of_sight,
    spherical_occultation,
)


def test_spherical_occultation_matches_analytical_closest_approach() -> None:
    body = SphericalBodyGeometry("planet", [0.0, 0.0, 0.0], 1.0, "TEST_I")
    blocked = spherical_occultation([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], body, frame="TEST_I")
    clear = spherical_occultation([-2.0, 2.0, 0.0], [2.0, 2.0, 0.0], body, frame="TEST_I")

    assert blocked.occulted
    assert blocked.closest_distance_m == pytest.approx(0.0)
    assert blocked.segment_fraction == pytest.approx(0.5)
    assert not clear.occulted
    assert clear.clearance_m == pytest.approx(1.0)
    aggregate = line_of_sight([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], (body,), frame="TEST_I")
    assert not aggregate.clear
    assert aggregate.occulting_body_names == ("planet",)


def test_apparent_disc_eclipse_classifies_sunlight_penumbra_and_umbra() -> None:
    luminous = SphericalBodyGeometry("star", [100.0, 0.0, 0.0], 10.0, "TEST_I")
    umbra_body = SphericalBodyGeometry("moon", [10.0, 0.0, 0.0], 2.0, "TEST_I")
    penumbra_body = SphericalBodyGeometry("moon", [10.0, 1.5, 0.0], 2.0, "TEST_I")
    clear_body = SphericalBodyGeometry("moon", [10.0, 5.0, 0.0], 2.0, "TEST_I")

    assert eclipse_state([0.0, 0.0, 0.0], luminous, umbra_body, frame="TEST_I").state == "umbra"
    penumbra = eclipse_state([0.0, 0.0, 0.0], luminous, penumbra_body, frame="TEST_I")
    assert penumbra.state == "penumbra"
    assert eclipse_state([0.0, 0.0, 0.0], luminous, clear_body, frame="TEST_I").state == "sunlit"


def test_ground_station_crossings_match_spherical_horizon_geometry() -> None:
    radius_m = 1.0e6
    station = SphericalGroundStation("equator", 0.0, 0.0, 0.0, radius_m)
    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 721)
    time_s = angles.copy()
    positions = np.column_stack(
        (
            2.0 * radius_m * np.cos(angles),
            2.0 * radius_m * np.sin(angles),
            np.zeros_like(angles),
        )
    )
    access = ground_station_access(time_s, positions, station, frame="BODY_FIXED")

    expected = np.arccos(0.5)
    assert [(item.kind) for item in access.crossings] == ["rise", "set"]
    assert access.crossings[0].time_s == pytest.approx(-expected, abs=2.0e-6)
    assert access.crossings[1].time_s == pytest.approx(expected, abs=2.0e-6)
    assert access.access_intervals_s[0] == pytest.approx((-expected, expected), abs=2.0e-6)


def test_geometry_rejects_implicit_frame_mixing() -> None:
    body = SphericalBodyGeometry("body", [0.0, 0.0, 0.0], 1.0, "A")
    with pytest.raises(ValueError, match="frame"):
        spherical_occultation([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], body, frame="B")
