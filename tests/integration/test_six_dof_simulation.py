from pathlib import Path

import numpy as np

from aerognc.configuration import load_six_dof_configuration
from aerognc.simulation.six_dof_simulator import simulate_six_dof

PROJECT_ROOT = Path(__file__).parents[2]


def test_configured_closed_loop_six_dof_ascent_is_finite_and_bounded() -> None:
    configuration = load_six_dof_configuration(PROJECT_ROOT / "configs" / "six_dof_nominal.yaml")
    result = simulate_six_dof(configuration)
    assert result.time_s[-1] == configuration.duration_s
    assert np.all(np.isfinite(np.column_stack(tuple(result.columns.values()))))
    assert np.max(np.abs(result.columns["quaternion_norm"] - 1.0)) < 1.0e-10
    assert result.columns["altitude_m"][-1] > 500.0
    assert result.columns["attitude_error_deg"].max() < 10.0
    assert result.columns["mass_kg"].min() >= configuration.base.vehicle.mass_properties.dry_mass_kg
    assert [event.name for event in result.events] == ["burnout"]
