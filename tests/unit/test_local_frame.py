"""Unit tests for home-referenced local-tangent geometry."""

import numpy as np
import pytest

from aerognc.mathematics.geodesy import GeodeticPosition
from aerognc.mathematics.local_frame import (
    WGS84,
    LocalTangentFrame,
    flat_earth_offset_ned_m,
    great_circle_distance_m,
    initial_bearing_rad,
    wrap_to_2pi,
    wrap_to_pi,
)

HOME = GeodeticPosition(
    latitude_rad=float(np.deg2rad(39.925)),
    longitude_rad=float(np.deg2rad(32.8369)),
    altitude_m=0.0,
)


def test_home_maps_to_origin() -> None:
    frame = LocalTangentFrame(origin=HOME)
    ned = frame.geodetic_to_ned(HOME)
    assert np.allclose(ned, np.zeros(3), atol=1e-6)


def test_geodetic_ned_round_trip() -> None:
    frame = LocalTangentFrame(origin=HOME)
    target = GeodeticPosition(
        latitude_rad=float(np.deg2rad(39.930)),
        longitude_rad=float(np.deg2rad(32.847)),
        altitude_m=180.0,
    )
    ned = frame.geodetic_to_ned(target)
    recovered = frame.ned_to_geodetic(ned)
    assert recovered.latitude_rad == pytest.approx(target.latitude_rad, abs=1e-10)
    assert recovered.longitude_rad == pytest.approx(target.longitude_rad, abs=1e-10)
    assert recovered.altitude_m == pytest.approx(target.altitude_m, abs=1e-4)


def test_north_is_positive_for_higher_latitude() -> None:
    frame = LocalTangentFrame(origin=HOME)
    north_point = GeodeticPosition(HOME.latitude_rad + 1e-3, HOME.longitude_rad, 0.0)
    ned = frame.geodetic_to_ned(north_point)
    assert ned[0] > 100.0  # North dominates (~6 km for 1e-3 rad)
    assert abs(ned[1]) < 1.0  # negligible East along a meridian
    # Down is NOT negligible: over ~6 km the point sits ~3 m below the tangent
    # plane due to Earth curvature (d^2 / 2R). This is expected, not an error.
    assert 0.0 < ned[2] < 10.0


def test_great_circle_distance_one_degree_latitude() -> None:
    a = GeodeticPosition(0.0, 0.0, 0.0)
    b = GeodeticPosition(float(np.deg2rad(1.0)), 0.0, 0.0)
    distance_m = great_circle_distance_m(a, b)
    # ~111.2 km per degree on the mean sphere.
    assert distance_m == pytest.approx(111195.0, rel=1e-3)


def test_initial_bearing_cardinal_directions() -> None:
    origin = GeodeticPosition(0.0, 0.0, 0.0)
    north = GeodeticPosition(float(np.deg2rad(1.0)), 0.0, 0.0)
    east = GeodeticPosition(0.0, float(np.deg2rad(1.0)), 0.0)
    assert initial_bearing_rad(origin, north) == pytest.approx(0.0, abs=1e-6)
    assert initial_bearing_rad(origin, east) == pytest.approx(0.5 * np.pi, abs=1e-6)


def test_flat_earth_matches_exact_frame_at_short_range() -> None:
    frame = LocalTangentFrame(origin=HOME)
    target = GeodeticPosition(
        latitude_rad=HOME.latitude_rad + float(np.deg2rad(0.01)),
        longitude_rad=HOME.longitude_rad + float(np.deg2rad(0.01)),
        altitude_m=50.0,
    )
    exact = frame.geodetic_to_ned(target)
    flat = flat_earth_offset_ned_m(HOME, target, WGS84)
    # Within ~1 m over ~1.5 km — flat-Earth is a valid short-range approximation;
    # the residual is dominated by the curvature term the flat model omits.
    assert np.allclose(exact, flat, atol=1.0)


def test_wrap_to_pi_range() -> None:
    assert wrap_to_pi(3.0 * np.pi) == pytest.approx(-np.pi, abs=1e-9) or wrap_to_pi(
        3.0 * np.pi
    ) == pytest.approx(np.pi, abs=1e-9)
    assert wrap_to_pi(0.5 * np.pi) == pytest.approx(0.5 * np.pi)
    assert -np.pi <= wrap_to_pi(10.0) < np.pi


def test_wrap_to_2pi_range() -> None:
    assert 0.0 <= wrap_to_2pi(-0.5) < 2.0 * np.pi
    assert wrap_to_2pi(2.5 * np.pi) == pytest.approx(0.5 * np.pi)


def test_non_finite_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        wrap_to_pi(np.inf)
    with pytest.raises(ValueError):
        great_circle_distance_m(HOME, HOME, radius_m=-1.0)
