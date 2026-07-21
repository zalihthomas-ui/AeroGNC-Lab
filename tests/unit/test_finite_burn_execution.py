import numpy as np
import pytest

from aerognc.astrodynamics.finite_burn import execute_two_body_finite_burn
from aerognc.astrodynamics.maneuvers import FiniteBurn
from aerognc.mathematics.adaptive_integrators import AdaptiveOptions


def test_finite_burn_boundaries_mass_and_rocket_equation_are_verified() -> None:
    burn = FiniteBurn(
        "trim",
        2.0,
        5.0,
        10.0,
        (1.0, 0.0, 0.0),
        frame="inertial",
        specific_impulse_s=300.0,
    )
    initial = np.array([1.0e12, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0])
    result = execute_two_body_finite_burn(
        initial,
        burn,
        gravitational_parameter_m3_s2=1.0,
        dry_mass_kg=90.0,
        end_time_s=10.0,
        integration_options=AdaptiveOptions(
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-12,
            initial_step_s=0.1,
            minimum_step_s=1.0e-12,
            maximum_step_s=0.5,
        ),
    )
    expected_propellant_kg = burn.mass_flow_rate_kg_s * burn.duration_s

    assert [(event.name, event.time_s) for event in result.events] == [
        ("burn_start", 2.0),
        ("burn_end", 7.0),
    ]
    assert result.propellant_used_kg == pytest.approx(expected_propellant_kg, abs=2.0e-12)
    assert abs(result.mass_balance_error_kg) < 2.0e-12
    velocity_increment_mps = result.state[-1, 0 + 3] - initial[3]
    assert velocity_increment_mps == pytest.approx(result.ideal_delivered_delta_v_mps, rel=2.0e-10)
    assert np.all(np.diff(result.time_s) > 0.0)


def test_finite_burn_execution_rejects_dry_mass_violation() -> None:
    burn = FiniteBurn("too-long", 0.0, 100.0, 100.0, (1.0, 0.0, 0.0), frame="inertial")
    with pytest.raises(FloatingPointError, match="propellant"):
        execute_two_body_finite_burn(
            [7.0e6, 0.0, 0.0, 0.0, 7_500.0, 0.0, 100.0],
            burn,
            gravitational_parameter_m3_s2=3.986e14,
            dry_mass_kg=99.0,
            end_time_s=101.0,
        )
