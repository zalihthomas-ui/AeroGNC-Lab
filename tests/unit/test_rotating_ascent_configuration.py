from pathlib import Path

import pytest
import yaml

from aerognc.configuration import ConfigurationError, load_rotating_ascent_configuration


def test_rotating_ascent_configuration_is_fictional_and_unit_explicit() -> None:
    configuration = load_rotating_ascent_configuration("configs/rotating_planet_ascent.yaml")
    assert configuration.planet.name == "Orbis-A"
    assert configuration.planet.ellipsoid.semi_major_axis_m == pytest.approx(6_400_000.0)
    assert configuration.planet.rotation_rate_radps == pytest.approx(7.5e-5)
    assert configuration.launch_site.geodetic.altitude_m == pytest.approx(850.0)
    assert configuration.base_configuration.vehicle.fictional
    assert "Fictional civilian" in configuration.safety_scope


def test_rotating_ascent_configuration_rejects_unknown_and_nonfictional_fields(
    tmp_path: Path,
) -> None:
    source = Path("configs/rotating_planet_ascent.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["base_configuration"] = str(Path("configs/three_dof_nominal.yaml").resolve())
    payload["unexpected"] = 1
    invalid = tmp_path / "unknown.yaml"
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_rotating_ascent_configuration(invalid)

    payload.pop("unexpected")
    payload["planet"]["fictional"] = False
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be true"):
        load_rotating_ascent_configuration(invalid)
