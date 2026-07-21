"""Unit tests for the waypoint data model and its validation."""

import numpy as np
import pytest

from aerognc.mission.waypoint import (
    AltitudeReference,
    LoiterDirection,
    Waypoint,
    WaypointAction,
)


def _wp(**overrides: object) -> Waypoint:
    base: dict[str, object] = {
        "id": 1,
        "name": "WP1",
        "latitude_deg": 39.927,
        "longitude_deg": 32.840,
        "altitude_m": 120.0,
    }
    base.update(overrides)
    return Waypoint(**base)  # type: ignore[arg-type]


def test_minimal_waypoint_defaults() -> None:
    wp = _wp()
    assert wp.action is WaypointAction.FLY_THROUGH
    assert wp.altitude_reference is AltitudeReference.RELATIVE_HOME
    assert wp.airspeed_mps is None


def test_geodetic_conversion_to_radians() -> None:
    wp = _wp(latitude_deg=45.0, longitude_deg=-90.0, altitude_m=10.0)
    geo = wp.geodetic()
    assert geo.latitude_rad == pytest.approx(np.deg2rad(45.0))
    assert geo.longitude_rad == pytest.approx(np.deg2rad(-90.0))
    assert geo.altitude_m == pytest.approx(10.0)


@pytest.mark.parametrize("latitude", [-91.0, 90.5])
def test_latitude_bounds_rejected(latitude: float) -> None:
    with pytest.raises(ValueError, match="latitude"):
        _wp(latitude_deg=latitude)


def test_non_finite_latitude_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        _wp(latitude_deg=np.nan)


@pytest.mark.parametrize("longitude", [-181.0, 200.0])
def test_longitude_bounds_rejected(longitude: float) -> None:
    with pytest.raises(ValueError, match="longitude"):
        _wp(longitude_deg=longitude)


@pytest.mark.parametrize("bad_id", [0, -3])
def test_non_positive_id_rejected(bad_id: int) -> None:
    with pytest.raises(ValueError, match="id"):
        _wp(id=bad_id)


def test_negative_airspeed_rejected() -> None:
    with pytest.raises(ValueError, match="airspeed_mps"):
        _wp(airspeed_mps=-5.0)


def test_loiter_requires_radius() -> None:
    with pytest.raises(ValueError, match="loiter_radius_m"):
        _wp(action=WaypointAction.LOITER)


def test_loiter_waypoint_valid() -> None:
    wp = _wp(
        action=WaypointAction.LOITER,
        loiter_radius_m=100.0,
        loiter_duration_s=60.0,
        loiter_direction=LoiterDirection.COUNTERCLOCKWISE,
    )
    assert wp.loiter_direction is LoiterDirection.COUNTERCLOCKWISE


def test_to_from_dict_round_trip() -> None:
    wp = _wp(
        action=WaypointAction.LOITER,
        loiter_radius_m=120.0,
        loiter_duration_s=30.0,
        airspeed_mps=22.0,
        notes="hold here",
    )
    restored = Waypoint.from_dict(wp.to_dict())
    assert restored == wp


def test_from_dict_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        Waypoint.from_dict(
            {"id": 1, "name": "x", "latitude_deg": 0, "longitude_deg": 0, "altitude_m": 0, "z": 1}
        )


def test_from_dict_missing_field_rejected() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        Waypoint.from_dict({"id": 1, "name": "x"})


def test_from_dict_bad_enum_rejected() -> None:
    with pytest.raises(ValueError, match="WaypointAction"):
        Waypoint.from_dict(
            {
                "id": 1,
                "name": "x",
                "latitude_deg": 0,
                "longitude_deg": 0,
                "altitude_m": 0,
                "action": "nonsense",
            }
        )
