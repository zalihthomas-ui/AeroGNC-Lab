from pathlib import Path

import pytest

from aerognc.configuration import ConfigurationError
from aerognc.configuration.orbit_tour_loader import load_orbit_tour_configuration

CONFIGURATION_PATH = Path("configs/orbit_assisted_tour.yaml")


def test_orbit_tour_configuration_loads_fictional_si_case() -> None:
    configuration = load_orbit_tour_configuration(CONFIGURATION_PATH)

    assert configuration.departure_body == "Asteria"
    assert configuration.assist_body == "Neria"
    assert configuration.destination_body == "Caelus"
    assert configuration.assist_dwell_revolutions == 2
    assert configuration.assist_parking_altitude_m == pytest.approx(300_000.0)
    assert configuration.initial_mass_kg > configuration.dry_mass_kg


def test_orbit_tour_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    source = CONFIGURATION_PATH.read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("route:\n", "unexpected: true\nroute:\n"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_orbit_tour_configuration(path)
