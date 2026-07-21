import numpy as np
import pytest

from aerognc.astrodynamics import (
    CircularOrbitBody,
    PrimaryBody,
    RestrictedNBodyModel,
    design_hohmann_transfer,
    evaluate_hyperbolic_flyby,
)


def _body() -> CircularOrbitBody:
    return CircularOrbitBody(
        name="Test world",
        role="assist",
        gravitational_parameter_m3_s2=10.0,
        radius_m=1.0,
        semi_major_axis_m=100.0,
        phase_at_epoch_rad=0.0,
    )


def test_circular_ephemeris_preserves_radius_speed_and_quarter_period_geometry() -> None:
    body = _body()
    primary_mu = 1_000.0
    position_0, velocity_0 = body.state_at_time(0.0, primary_mu)
    position_quarter, velocity_quarter = body.state_at_time(
        0.25 * body.orbital_period_s(primary_mu), primary_mu
    )

    np.testing.assert_allclose(position_0, [100.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(position_quarter, [0.0, 100.0, 0.0], atol=1.0e-12)
    assert np.linalg.norm(velocity_0) == pytest.approx(np.sqrt(primary_mu / 100.0))
    assert np.linalg.norm(velocity_quarter) == pytest.approx(np.linalg.norm(velocity_0))
    assert np.dot(position_quarter, velocity_quarter) == pytest.approx(0.0, abs=1.0e-12)


def test_restricted_n_body_acceleration_includes_primary_direct_and_indirect_terms() -> None:
    model = RestrictedNBodyModel(
        PrimaryBody("Test star", gravitational_parameter_m3_s2=1_000.0, radius_m=1.0),
        (_body(),),
    )
    acceleration = model.acceleration(0.0, [10.0, 0.0, 0.0])
    expected_x = -10.0 + 10.0 / 90.0**2 - 10.0 / 100.0**2
    np.testing.assert_allclose(acceleration, [expected_x, 0.0, 0.0], atol=1.0e-14)


def test_hohmann_and_hyperbolic_flyby_equations_match_closed_forms() -> None:
    primary_mu = 3.986004418e14
    radius_1 = 7.0e6
    radius_2 = 4.2164e7
    transfer = design_hohmann_transfer(primary_mu, radius_1, radius_2)
    expected_time = np.pi * np.sqrt((0.5 * (radius_1 + radius_2)) ** 3 / primary_mu)
    assert transfer.transfer_time_s == pytest.approx(expected_time)
    assert transfer.departure_delta_v_mps > 0.0
    assert transfer.arrival_delta_v_mps > 0.0

    flyby = evaluate_hyperbolic_flyby(1.26686534e17, 4.0e8, 6_000.0)
    expected_eccentricity = 1.0 + 4.0e8 * 6_000.0**2 / 1.26686534e17
    assert flyby.eccentricity == pytest.approx(expected_eccentricity)
    assert flyby.turn_angle_rad == pytest.approx(2.0 * np.arcsin(1.0 / expected_eccentricity))
    assert 0.0 < flyby.turn_angle_rad < np.pi
    assert flyby.periapsis_speed_mps > 6_000.0


def test_astrodynamics_models_reject_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="must differ"):
        design_hohmann_transfer(1.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="positive"):
        evaluate_hyperbolic_flyby(1.0, -2.0, 3.0)
    with pytest.raises(ValueError, match="inclination"):
        CircularOrbitBody(
            "bad",
            "background",
            1.0,
            1.0,
            10.0,
            0.0,
            inclination_rad=4.0,
        )
