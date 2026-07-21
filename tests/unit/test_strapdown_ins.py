import numpy as np
import pytest

from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.gnc.strapdown_ins import (
    ImuIncrement,
    RotatingNavigationState,
    compensate_two_sample_imu,
    gravity_ned_mps2,
    propagate_rotating_strapdown,
)
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    ReferenceEllipsoid,
    body_rotation_rate_ned,
)
from aerognc.mathematics.quaternion import rotation_vector_to_quaternion


def _planet() -> RotatingOblatePlanet:
    return RotatingOblatePlanet(
        name="Orbis-A",
        ellipsoid=ReferenceEllipsoid(6_400_000.0, 1.0 / 300.0),
        gravitational_parameter_m3ps2=4.05e14,
        rotation_rate_radps=7.5e-5,
        j2=1.1e-3,
    )


def test_rotation_vector_quaternion_small_and_quarter_turn() -> None:
    identity = rotation_vector_to_quaternion([0.0, 0.0, 0.0])
    quarter_turn = rotation_vector_to_quaternion([0.0, 0.0, 0.5 * np.pi])
    np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        quarter_turn,
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1.0e-14,
    )


def test_two_sample_coning_and_sculling_terms_have_documented_signs() -> None:
    first = ImuIncrement(0.0, 0.01, [0.02, 0.0, 0.0], [0.4, 0.0, 0.0])
    second = ImuIncrement(0.01, 0.02, [0.0, 0.03, 0.0], [0.0, 0.2, 0.0])
    combined = compensate_two_sample_imu(first, second)

    expected_angle = np.array([0.02, 0.03, 0.0004])
    summed_angle = np.array([0.02, 0.03, 0.0])
    summed_velocity = np.array([0.4, 0.2, 0.0])
    expected_velocity = (
        summed_velocity
        + 0.5 * np.cross(summed_angle, summed_velocity)
        + (2.0 / 3.0)
        * (
            np.cross(np.array([0.02, 0.0, 0.0]), np.array([0.0, 0.2, 0.0]))
            + np.cross(np.array([0.4, 0.0, 0.0]), np.array([0.0, 0.03, 0.0]))
        )
    )
    np.testing.assert_allclose(combined.delta_angle_body_rad, expected_angle)
    np.testing.assert_allclose(combined.delta_velocity_body_mps, expected_velocity)
    assert combined.duration_s == pytest.approx(0.02)


def test_stationary_rotating_planet_mechanisation_remains_bounded() -> None:
    planet = _planet()
    initial = GeodeticPosition(np.deg2rad(38.0), np.deg2rad(29.0), 850.0)
    state = RotatingNavigationState(initial, np.zeros(3), [1.0, 0.0, 0.0, 0.0])
    gravity_ned = gravity_ned_mps2(initial, planet)
    angular_rate_body = body_rotation_rate_ned(initial.latitude_rad, planet.rotation_rate_radps)

    step_s = 0.01
    for index in range(1000):
        propagate_rotating_strapdown(
            state,
            ImuIncrement(
                index * step_s,
                (index + 1) * step_s,
                angular_rate_body * step_s,
                -gravity_ned * step_s,
            ),
            planet,
        )

    displacement = np.array(
        [
            (state.geodetic.latitude_rad - initial.latitude_rad) * 6_400_000.0,
            (state.geodetic.longitude_rad - initial.longitude_rad)
            * 6_400_000.0
            * np.cos(initial.latitude_rad),
            state.geodetic.altitude_m - initial.altitude_m,
        ]
    )
    assert np.linalg.norm(displacement) < 2.0e-4
    assert np.linalg.norm(state.velocity_ned_mps) < 5.0e-5
    assert np.linalg.norm(state.quaternion_nb) == pytest.approx(1.0, abs=1.0e-14)
    np.testing.assert_allclose(state.quaternion_nb, [1.0, 0.0, 0.0, 0.0], atol=2.0e-8)


def test_noncontiguous_two_sample_increment_is_rejected() -> None:
    first = ImuIncrement(0.0, 0.01, np.zeros(3), np.zeros(3))
    second = ImuIncrement(0.02, 0.03, np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="contiguous"):
        compensate_two_sample_imu(first, second)
