import numpy as np
import pytest

from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    ReferenceEllipsoid,
    body_rotation_rate_ned,
    dcm_ecef_to_ned,
    dcm_inertial_to_ecef,
    ecef_position_to_ned,
    ecef_to_geodetic,
    ecef_to_inertial_state,
    geodetic_to_ecef,
    inertial_to_ecef_state,
    meridian_radius_m,
    ned_position_to_ecef,
    prime_vertical_radius_m,
    transport_rate_ned,
)

ELLIPSOID = ReferenceEllipsoid(semi_major_axis_m=6_400_000.0, flattening=1.0 / 300.0)


def test_reference_ellipsoid_axes_and_curvature_are_consistent() -> None:
    assert ELLIPSOID.semi_minor_axis_m == pytest.approx(6_378_666.666666667)
    assert ELLIPSOID.first_eccentricity_squared == pytest.approx(2.0 / 300.0 - 1.0 / 300.0**2)
    assert prime_vertical_radius_m(0.0, ELLIPSOID) == pytest.approx(ELLIPSOID.semi_major_axis_m)
    assert meridian_radius_m(0.0, ELLIPSOID) == pytest.approx(
        ELLIPSOID.semi_major_axis_m * (1.0 - ELLIPSOID.first_eccentricity_squared)
    )


@pytest.mark.parametrize(
    ("latitude_rad", "longitude_rad", "altitude_m"),
    [
        (0.0, 0.0, 0.0),
        (np.deg2rad(37.4), np.deg2rad(28.2), 1265.0),
        (np.deg2rad(-61.0), np.deg2rad(145.0), 31_000.0),
        (np.deg2rad(89.999), np.deg2rad(-72.0), 250.0),
        (np.deg2rad(-22.0), np.deg2rad(205.0), -75.0),
    ],
)
def test_geodetic_ecef_round_trip(
    latitude_rad: float,
    longitude_rad: float,
    altitude_m: float,
) -> None:
    original = GeodeticPosition(latitude_rad, longitude_rad, altitude_m)
    recovered = ecef_to_geodetic(geodetic_to_ecef(original, ELLIPSOID), ELLIPSOID)
    longitude_error = np.arctan2(
        np.sin(recovered.longitude_rad - longitude_rad),
        np.cos(recovered.longitude_rad - longitude_rad),
    )
    assert recovered.latitude_rad == pytest.approx(latitude_rad, abs=2.0e-12)
    assert longitude_error == pytest.approx(0.0, abs=2.0e-12)
    assert recovered.altitude_m == pytest.approx(altitude_m, abs=2.0e-5)


def test_equator_pole_and_local_ned_directions() -> None:
    equator = GeodeticPosition(0.0, 0.0, 0.0)
    north_pole = GeodeticPosition(0.5 * np.pi, 0.0, 0.0)
    assert geodetic_to_ecef(equator, ELLIPSOID) == pytest.approx(
        [ELLIPSOID.semi_major_axis_m, 0.0, 0.0]
    )
    assert geodetic_to_ecef(north_pole, ELLIPSOID) == pytest.approx(
        [0.0, 0.0, ELLIPSOID.semi_minor_axis_m], abs=1.0e-9
    )
    dcm_ne = dcm_ecef_to_ned(0.0, 0.0)
    assert dcm_ne @ dcm_ne.T == pytest.approx(np.eye(3), abs=1.0e-15)
    assert dcm_ne @ np.array([0.0, 0.0, 1.0]) == pytest.approx([1.0, 0.0, 0.0])
    assert dcm_ne @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    assert dcm_ne @ np.array([-1.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])


def test_local_ned_position_inverse() -> None:
    origin = GeodeticPosition(np.deg2rad(42.0), np.deg2rad(31.0), 850.0)
    displacement_ned_m = np.array([1200.0, -340.0, -85.0])
    target_ecef_m = ned_position_to_ecef(displacement_ned_m, origin, ELLIPSOID)
    assert ecef_position_to_ned(target_ecef_m, origin, ELLIPSOID) == pytest.approx(
        displacement_ned_m, abs=1.0e-9
    )


def test_inertial_ecef_state_round_trip_includes_transport_velocity() -> None:
    position_i_m = np.array([7.1e6, -2.0e6, 1.4e6])
    velocity_i_mps = np.array([1300.0, 6900.0, -800.0])
    time_s = 1375.0
    rotation_rate_radps = 8.0e-5
    position_e_m, velocity_e_mps = inertial_to_ecef_state(
        position_i_m,
        velocity_i_mps,
        time_s=time_s,
        rotation_rate_radps=rotation_rate_radps,
        initial_rotation_angle_rad=0.31,
    )
    recovered_position_i_m, recovered_velocity_i_mps = ecef_to_inertial_state(
        position_e_m,
        velocity_e_mps,
        time_s=time_s,
        rotation_rate_radps=rotation_rate_radps,
        initial_rotation_angle_rad=0.31,
    )
    assert recovered_position_i_m == pytest.approx(position_i_m, abs=1.0e-9)
    assert recovered_velocity_i_mps == pytest.approx(velocity_i_mps, abs=1.0e-10)
    assert dcm_inertial_to_ecef(0.31) @ dcm_inertial_to_ecef(0.31).T == pytest.approx(
        np.eye(3), abs=1.0e-15
    )


def test_body_and_transport_rates_use_documented_ned_signs() -> None:
    latitude_rad = np.deg2rad(45.0)
    rotation_rate_radps = 7.0e-5
    assert body_rotation_rate_ned(latitude_rad, rotation_rate_radps) == pytest.approx(
        [rotation_rate_radps / np.sqrt(2.0), 0.0, -rotation_rate_radps / np.sqrt(2.0)]
    )
    geodetic = GeodeticPosition(latitude_rad, 0.0, 1200.0)
    velocity_ned_mps = np.array([210.0, 95.0, -4.0])
    transport = transport_rate_ned(geodetic, velocity_ned_mps, ELLIPSOID)
    assert transport[0] > 0.0
    assert transport[1] < 0.0
    assert transport[2] < 0.0


def test_geodesy_rejects_invalid_domains() -> None:
    with pytest.raises(ValueError, match="body centre"):
        ecef_to_geodetic(np.zeros(3), ELLIPSOID)
    with pytest.raises(ValueError, match="latitude_rad"):
        GeodeticPosition(np.pi, 0.0, 0.0)
    with pytest.raises(ValueError, match="flattening"):
        ReferenceEllipsoid(10.0, 1.0)
