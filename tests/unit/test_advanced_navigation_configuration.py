from pathlib import Path

import pytest

from aerognc.configuration import ConfigurationError
from aerognc.configuration.advanced_navigation_loader import (
    load_advanced_navigation_configuration,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_advanced_navigation_configuration_loads_strict_si_model() -> None:
    configuration = load_advanced_navigation_configuration(
        PROJECT_ROOT / "configs" / "advanced_navigation.yaml"
    )
    assert configuration.name == "rotating_delayed_navigation_verification"
    assert configuration.step_s == pytest.approx(0.02)
    assert configuration.rotating_ascent.planet.name == "Orbis-A"
    assert configuration.navigation_filter.fixed_lag_s > configuration.gnss.delay_s
    assert len(configuration.navigation_filter.initial_standard_deviation) == 15
    assert {fault.sensor_name for fault in configuration.faults} == {"gnss", "barometer"}


def test_advanced_navigation_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs" / "advanced_navigation.yaml").read_text(encoding="utf-8")
    invalid = source.replace("consistency:\n", "unexpected: true\nconsistency:\n")
    path = tmp_path / "invalid.yaml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_advanced_navigation_configuration(path)
