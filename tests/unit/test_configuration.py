from pathlib import Path

import pytest
import yaml

from aerognc.configuration.loader import ConfigurationError, load_three_dof_configuration
from aerognc.vehicle.aero_database import TabulatedAerodynamicDatabase

PROJECT_ROOT = Path(__file__).parents[2]
NOMINAL_CONFIG = PROJECT_ROOT / "configs" / "three_dof_nominal.yaml"


def test_nominal_configuration_loads_and_is_explicitly_fictional() -> None:
    config = load_three_dof_configuration(NOMINAL_CONFIG)
    assert config.vehicle.fictional
    assert "fictional" in config.safety_scope.lower()
    assert config.vehicle.mass_properties.wet_mass_kg == pytest.approx(30.0)
    assert config.simulation.step_s == 0.02


def test_unknown_key_fails_with_context(tmp_path: Path) -> None:
    data = yaml.safe_load(NOMINAL_CONFIG.read_text(encoding="utf-8"))
    data["simulation"]["hidden_conversion"] = 3.0
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    # Keep the relative vehicle link valid for the temporary scenario.
    data["vehicle_file"] = str(PROJECT_ROOT / "configs" / "vehicle_asteria_sr1.yaml")
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"scenario\.simulation: unknown keys"):
        load_three_dof_configuration(config_path)


def test_invalid_step_and_missing_file_fail_clearly(tmp_path: Path) -> None:
    data = yaml.safe_load(NOMINAL_CONFIG.read_text(encoding="utf-8"))
    data["simulation"]["step_s"] = -0.1
    data["vehicle_file"] = str(PROJECT_ROOT / "configs" / "vehicle_asteria_sr1.yaml")
    config_path = tmp_path / "bad_step.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="step_s: must be positive"):
        load_three_dof_configuration(config_path)
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_three_dof_configuration(tmp_path / "missing.yaml")


def test_regular_grid_aerodynamic_database_configuration_loads() -> None:
    config = load_three_dof_configuration(PROJECT_ROOT / "configs" / "three_dof_aero_database.yaml")
    provider = config.vehicle.aerodynamics.coefficient_provider
    assert isinstance(provider, TabulatedAerodynamicDatabase)
    assert provider.axis_names == ("mach", "alpha_rad", "beta_rad")
    assert provider.source_path == (PROJECT_ROOT / "configs" / "aero_database_synthetic.csv")
    assert provider.source_sha256 is not None
