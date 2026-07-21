from pathlib import Path

import numpy as np
import pytest

from aerognc.configuration import load_rotating_six_dof_configuration
from aerognc.simulation.rotating_six_dof import simulate_rotating_six_dof


def test_rotating_planet_six_dof_scenario_is_bounded_and_frame_explicit() -> None:
    configuration = load_rotating_six_dof_configuration(Path("configs/rotating_six_dof.yaml"))
    result = simulate_rotating_six_dof(configuration)

    assert result.scenario_name == "rotating_planet_six_dof_ascent"
    assert result.time_s[-1] == pytest.approx(8.0)
    assert [event.name for event in result.events] == ["burnout"]
    assert result.columns["altitude_m"][0] == pytest.approx(5.0, abs=2.0e-6)
    assert result.columns["altitude_m"][-1] > 800.0
    assert np.max(np.abs(result.columns["quaternion_norm"] - 1.0)) < 1.0e-9
    assert result.columns["inertial_speed_mps"][0] > result.columns["total_velocity_mps"][0]
    dry_mass_kg = configuration.six_dof.base.vehicle.mass_properties.dry_mass_kg
    assert np.min(result.columns["mass_kg"]) >= dry_mass_kg
