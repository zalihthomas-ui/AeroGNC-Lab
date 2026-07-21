import numpy as np
import pytest

from aerognc.astrodynamics.ephemeris import (
    EphemerisUnavailableError,
    SpiceEphemeris,
    TabulatedEphemeris,
)
from aerognc.astrodynamics.full_n_body import (
    GRAVITATIONAL_CONSTANT_M3_KG_S2,
    FullNBodyModel,
    MassiveBody,
    propagate_full_n_body,
)
from aerognc.astrodynamics.perturbations import (
    ASTRONOMICAL_UNIT_M,
    SOLAR_RADIATION_PRESSURE_AT_ONE_AU_PA,
    hill_sphere_radius_m,
    j2_acceleration_mps2,
    laplace_sphere_of_influence_radius_m,
    schwarzschild_acceleration_mps2,
    solar_radiation_pressure_acceleration_mps2,
)


def test_j2_srp_and_relativistic_force_directions_and_scales() -> None:
    equatorial = j2_acceleration_mps2([7.0e6, 0.0, 0.0], 3.986e14, 6.378e6, 1.0826e-3)
    assert equatorial[0] < 0.0
    assert equatorial[1] == pytest.approx(0.0)
    radiation = solar_radiation_pressure_acceleration_mps2(
        [ASTRONOMICAL_UNIT_M, 0.0, 0.0], area_m2=20.0, mass_kg=1_000.0, reflectivity_coefficient=1.5
    )
    assert radiation[0] == pytest.approx(
        SOLAR_RADIATION_PRESSURE_AT_ONE_AU_PA * 20.0 * 1.5 / 1_000.0
    )
    relativistic = schwarzschild_acceleration_mps2(
        [ASTRONOMICAL_UNIT_M, 0.0, 0.0], [0.0, 29_780.0, 0.0], 1.32712440018e20
    )
    assert np.all(np.isfinite(relativistic))
    assert np.linalg.norm(relativistic) < 1.0e-6


def test_hill_and_laplace_spheres_match_definitions() -> None:
    hill = hill_sphere_radius_m(1.5e11, 0.1, 6.0e24, 2.0e30)
    laplace = laplace_sphere_of_influence_radius_m(1.5e11, 6.0e24, 2.0e30)
    assert hill == pytest.approx(1.5e11 * 0.9 * (6.0e24 / (6.0e30)) ** (1.0 / 3.0))
    assert laplace == pytest.approx(1.5e11 * (6.0e24 / 2.0e30) ** 0.4)


def test_full_n_body_preserves_momentum_and_bounds_energy_error() -> None:
    primary_mass = 1.0e20
    secondary_mass = 1.0e16
    separation = 1.0e8
    total_mass = primary_mass + secondary_mass
    angular_rate = np.sqrt(GRAVITATIONAL_CONSTANT_M3_KG_S2 * total_mass / separation**3)
    primary_radius = secondary_mass / total_mass * separation
    secondary_radius = primary_mass / total_mass * separation
    model = FullNBodyModel(
        (MassiveBody("primary", primary_mass), MassiveBody("secondary", secondary_mass))
    )
    initial = np.array(
        [
            -primary_radius,
            0.0,
            0.0,
            0.0,
            -angular_rate * primary_radius,
            0.0,
            secondary_radius,
            0.0,
            0.0,
            0.0,
            angular_rate * secondary_radius,
            0.0,
        ]
    )
    period_s = 2.0 * np.pi / angular_rate

    result = propagate_full_n_body(model, initial, period_s, period_s / 500.0)
    initial_energy = model.total_energy_j(result.state[0])
    final_energy = model.total_energy_j(result.state[-1])

    np.testing.assert_allclose(
        model.total_linear_momentum_kg_mps(result.state[-1]), np.zeros(3), atol=1.0e5
    )
    assert abs((final_energy - initial_energy) / initial_energy) < 1.0e-8
    np.testing.assert_allclose(result.state[-1], initial, rtol=1.0e-8, atol=0.5)


def test_tabular_ephemeris_interpolates_and_external_adapter_fails_honestly() -> None:
    ephemeris = TabulatedEphemeris(
        np.array([0.0, 10.0]),
        {"world": np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 1.0, 2.0, 3.0]])},
    )
    position, velocity = ephemeris.state_at_time("world", 5.0)
    np.testing.assert_allclose(position, [5.0, 10.0, 15.0])
    np.testing.assert_allclose(velocity, [1.0, 2.0, 3.0])
    with pytest.raises(EphemerisUnavailableError):
        SpiceEphemeris(["definitely_missing_kernel.bsp"], observer="SUN")
