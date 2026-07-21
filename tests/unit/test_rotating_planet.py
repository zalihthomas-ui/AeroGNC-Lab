import numpy as np
import pytest

from aerognc.dynamics.rotating_planet import (
    RotatingTranslationalState,
    rotating_translational_derivative,
)
from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.mathematics.geodesy import GeodeticPosition, ReferenceEllipsoid

ELLIPSOID = ReferenceEllipsoid(6_400_000.0, 1.0 / 300.0)
PLANET = RotatingOblatePlanet(
    name="Fictional Orbis",
    ellipsoid=ELLIPSOID,
    gravitational_parameter_m3ps2=4.05e14,
    rotation_rate_radps=7.5e-5,
    j2=1.1e-3,
)


def test_central_gravity_and_apparent_acceleration_signs() -> None:
    spherical = RotatingOblatePlanet(
        name="Spherical test body",
        ellipsoid=ReferenceEllipsoid(6.4e6, 0.0),
        gravitational_parameter_m3ps2=4.0e14,
        rotation_rate_radps=1.0e-4,
    )
    position_ecef_m = np.array([7.0e6, 0.0, 0.0])
    velocity_ecef_mps = np.array([0.0, 1000.0, 0.0])
    gravity = spherical.gravity_ecef_mps2(position_ecef_m)
    coriolis = spherical.coriolis_ecef_mps2(velocity_ecef_mps)
    centrifugal = spherical.centrifugal_ecef_mps2(position_ecef_m)
    assert gravity == pytest.approx([-4.0e14 / (7.0e6) ** 2, 0.0, 0.0])
    assert coriolis == pytest.approx([0.2, 0.0, 0.0])
    assert centrifugal == pytest.approx([0.07, 0.0, 0.0])
    assert spherical.apparent_acceleration_ecef_mps2(
        position_ecef_m, velocity_ecef_mps
    ) == pytest.approx(gravity + coriolis + centrifugal)


def test_j2_gravity_is_axisymmetric_and_surface_down_is_positive() -> None:
    radius_m = 7.0e6
    acceleration_x = PLANET.gravity_ecef_mps2([radius_m, 0.0, 0.0])
    acceleration_y = PLANET.gravity_ecef_mps2([0.0, radius_m, 0.0])
    assert acceleration_x[0] == pytest.approx(acceleration_y[1])
    assert acceleration_x[1:] == pytest.approx([0.0, 0.0])
    gravity_down = PLANET.surface_gravity_down_mps2(
        GeodeticPosition(np.deg2rad(38.0), np.deg2rad(29.0), 0.0)
    )
    assert 8.0 < gravity_down < 11.0


def test_rotating_derivative_matches_equatorial_circular_kinematics() -> None:
    spherical = RotatingOblatePlanet(
        name="Circular test body",
        ellipsoid=ReferenceEllipsoid(6.4e6, 0.0),
        gravitational_parameter_m3ps2=4.0e14,
        rotation_rate_radps=8.0e-5,
    )
    radius_m = 7.1e6
    inertial_rate_radps = np.sqrt(spherical.gravitational_parameter_m3ps2 / radius_m**3)
    relative_rate_radps = inertial_rate_radps - spherical.rotation_rate_radps
    state = RotatingTranslationalState(
        [radius_m, 0.0, 0.0],
        [0.0, relative_rate_radps * radius_m, 0.0],
    )
    derivative = rotating_translational_derivative(state.as_array(), spherical)
    assert derivative[:3] == pytest.approx(state.velocity_ecef_mps)
    assert derivative[3:] == pytest.approx(
        [-(relative_rate_radps**2) * radius_m, 0.0, 0.0], rel=1.0e-13, abs=1.0e-13
    )


def test_specific_force_is_added_explicitly() -> None:
    state = RotatingTranslationalState([7.0e6, 0.0, 0.0], [0.0, 100.0, 0.0])
    baseline = rotating_translational_derivative(state.as_array(), PLANET)
    forced = rotating_translational_derivative(state.as_array(), PLANET, [1.0, -2.0, 3.0])
    assert forced[3:] - baseline[3:] == pytest.approx([1.0, -2.0, 3.0])


def test_rotating_models_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        RotatingOblatePlanet("bad", ELLIPSOID, 0.0, 1.0e-4)
    with pytest.raises(ValueError, match="centre"):
        PLANET.gravity_ecef_mps2(np.zeros(3))
    with pytest.raises(ValueError, match="state"):
        RotatingTranslationalState.from_array(np.zeros(5))
