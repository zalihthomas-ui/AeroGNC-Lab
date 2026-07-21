from pathlib import Path

import pytest

from aerognc.configuration import ConfigurationError, load_aircraft_configuration

CONFIGURATION_PATH = Path("configs/aircraft_sandbox.yaml")


def test_aircraft_configuration_loads_fictional_coefficient_model() -> None:
    configuration = load_aircraft_configuration(CONFIGURATION_PATH)

    assert configuration.name == "Aquila-X1 Civilian Research Aircraft"
    assert configuration.aerodynamics.cl_alpha_per_rad > 0.0
    assert configuration.aerodynamics.cd_zero > 0.0
    assert configuration.mass.initial_mass_kg > configuration.mass.dry_mass_kg
    assert configuration.propulsion.rocket_assist_available
    assert all(
        word in configuration.safety_scope.casefold()
        for word in ("fictional", "civilian", "synthetic")
    )


def test_aircraft_configuration_rejects_unknown_field(tmp_path: Path) -> None:
    source = CONFIGURATION_PATH.read_text(encoding="utf-8")
    invalid = tmp_path / "invalid_aircraft.yaml"
    invalid.write_text(source.replace("planet:\n", "unexpected: true\nplanet:\n"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unexpected"):
        load_aircraft_configuration(invalid)


def test_aircraft_configuration_rejects_nonfictional_metadata(tmp_path: Path) -> None:
    source = CONFIGURATION_PATH.read_text(encoding="utf-8")
    invalid = tmp_path / "unsafe_aircraft.yaml"
    invalid.write_text(source.replace("fictional: true", "fictional: false"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="fictional"):
        load_aircraft_configuration(invalid)
