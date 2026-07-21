from pathlib import Path

import numpy as np

from aerognc.configuration import load_three_dof_configuration
from aerognc.simulation.simulator import simulate_three_dof

PROJECT_ROOT = Path(__file__).parents[2]


def test_nominal_ascent_completes_with_ordered_events_and_physical_bounds() -> None:
    configuration = load_three_dof_configuration(
        PROJECT_ROOT / "configs" / "three_dof_nominal.yaml"
    )
    result = simulate_three_dof(configuration)
    event_names = [event.name for event in result.events]
    assert event_names == ["burnout", "apogee", "ground_impact"]
    assert result.events[0].time_s < result.events[1].time_s < result.events[2].time_s
    assert result.events[-1].time_s < configuration.simulation.maximum_time_s
    assert result.columns["altitude_m"].max() > 1_000.0
    assert result.columns["altitude_m"].min() >= -1.0e-8
    assert result.columns["mass_kg"].min() >= configuration.vehicle.mass_properties.dry_mass_kg
    assert np.all(np.diff(result.columns["mass_kg"]) <= 1.0e-10)
    assert np.all(result.columns["dynamic_pressure_pa"] >= 0.0)


def test_nominal_simulation_meets_local_runtime_requirement() -> None:
    configuration = load_three_dof_configuration(
        PROJECT_ROOT / "configs" / "three_dof_nominal.yaml"
    )
    result = simulate_three_dof(configuration)
    assert result.execution_time_s < 5.0
