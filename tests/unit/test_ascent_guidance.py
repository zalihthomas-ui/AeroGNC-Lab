import numpy as np
import pytest

from aerognc.configuration import load_ascent_guidance_configuration
from aerognc.gnc.ascent_guidance import (
    AscentGuidanceDecision,
    AscentGuidanceInputs,
    ConstraintAwareAscentGuidance,
)
from aerognc.simulation.guided_ascent import simulate_guided_ascent


def _inputs(**overrides: float) -> AscentGuidanceInputs:
    values = {
        "time_s": 2.0,
        "dynamic_pressure_pa": 5000.0,
        "air_flight_path_angle_rad": np.deg2rad(80.0),
        "mass_kg": 25.0,
        "nominal_thrust_n": 1000.0,
        "aerodynamic_force_magnitude_n": 50.0,
        "predicted_ballistic_apogee_m": 500.0,
    }
    values.update(overrides)
    return AscentGuidanceInputs(**values)


def test_online_guidance_applies_independent_constraint_limiters() -> None:
    configuration = load_ascent_guidance_configuration("configs/constrained_ascent_guidance.yaml")
    guidance = ConstraintAwareAscentGuidance(configuration)
    decision = AscentGuidanceDecision(0.0, 1.0)

    nominal = guidance.command(_inputs(), decision)
    assert 0.0 < nominal.throttle <= 1.0
    assert abs(nominal.elevation_rad - np.deg2rad(80.0)) <= np.deg2rad(5.0)

    max_q = guidance.command(
        _inputs(dynamic_pressure_pa=configuration.maximum_dynamic_pressure_pa), decision
    )
    assert max_q.throttle == 0.0
    assert max_q.dynamic_pressure_limited

    load_limited = guidance.command(_inputs(nominal_thrust_n=10_000.0), decision)
    assert load_limited.proper_load_limited
    assert load_limited.throttle < nominal.throttle


def test_guided_simulation_preserves_mass_and_detects_events() -> None:
    configuration = load_ascent_guidance_configuration("configs/constrained_ascent_guidance.yaml")
    run = simulate_guided_ascent(
        configuration,
        AscentGuidanceDecision(0.0, 1.0),
        governor_enabled=True,
    )
    event_names = {event.name for event in run.result.events}
    assert event_names == {"motor_window_end", "apogee", "ground_impact"}
    mass = run.result.columns["mass_kg"]
    propellant = run.result.columns["propellant_mass_kg"]
    dry_mass = configuration.base_scenario.vehicle.mass_properties.dry_mass_kg
    assert np.all(np.diff(mass) <= 1.0e-10)
    np.testing.assert_allclose(mass, dry_mass + propellant, atol=1.0e-10)
    assert mass[-1] >= dry_mass
    assert run.all_constraints_satisfied


def test_guidance_records_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="throttle_scale"):
        AscentGuidanceDecision(0.0, 1.1)
    with pytest.raises(ValueError, match="mass_kg"):
        _inputs(mass_kg=0.0)
