import pytest

from aerognc.configuration import load_three_dof_configuration
from aerognc.simulation.monte_carlo import (
    DispersionSample,
    MonteCarloRunResult,
    _perturbed_vehicle,
    summarise_monte_carlo,
)


def test_failed_runs_are_retained_while_successes_are_summarised() -> None:
    successful = MonteCarloRunResult(
        0,
        1,
        {"scale": 1.0},
        True,
        None,
        {"apogee_m": 100.0},
        {"minimum_apogee": 10.0},
        {"minimum_apogee": True, "overall": True},
    )
    failed = MonteCarloRunResult(
        1,
        2,
        {"scale": 2.0},
        False,
        "synthetic failure",
        {},
        {},
        {"overall": False},
    )
    summary = summarise_monte_carlo("test", 3, (successful, failed))
    assert summary.successful_count == 1
    assert summary.failed_count == 1
    assert summary.statistics["apogee_m"]["mean"] == 100.0
    assert summary.requirement_pass_rates["overall"] == 0.5


def test_tabulated_aerodynamics_accept_monte_carlo_drag_dispersion() -> None:
    base = load_three_dof_configuration("configs/three_dof_aero_database.yaml").vehicle
    sample = DispersionSample(
        run_index=0,
        random_seed=1,
        initial_speed_offset_mps=0.0,
        initial_elevation_offset_deg=0.0,
        vehicle_mass_scale=1.0,
        thrust_scale=1.0,
        thrust_misalignment_pitch_deg=0.0,
        thrust_misalignment_yaw_deg=0.0,
        aerodynamic_scale=1.25,
        wind_scale=1.0,
        sensor_noise_scale=1.0,
        sensor_bias_scale=1.0,
        actuator_delay_scale=1.0,
        controller_gain_scale=1.0,
    )

    dispersed = _perturbed_vehicle(base, sample)
    nominal_coefficients = base.aerodynamics.coefficients(0.8, 0.0, 0.0)
    dispersed_coefficients = dispersed.aerodynamics.coefficients(0.8, 0.0, 0.0)

    assert dispersed_coefficients.drag == pytest.approx(1.25 * nominal_coefficients.drag)
    assert dispersed_coefficients.pitch == pytest.approx(nominal_coefficients.pitch)
