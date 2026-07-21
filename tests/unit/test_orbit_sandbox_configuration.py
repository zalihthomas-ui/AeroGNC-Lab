from pathlib import Path

import pytest

from aerognc.configuration import ConfigurationError, load_orbit_sandbox_configuration

CONFIGURATION_PATH = Path("configs/orbit_sandbox.yaml")


def test_orbit_sandbox_configuration_loads_public_safe_defaults() -> None:
    configuration = load_orbit_sandbox_configuration(CONFIGURATION_PATH)

    assert configuration.model == "perturbed_decay"
    assert configuration.primary.name == "Orbis-A"
    assert configuration.satellite.name == "Meridian-1"
    assert configuration.correction.enabled is False
    assert len(configuration.secondaries) == 2
    assert "fictional" in configuration.safety_scope.casefold()
    assert "civilian" in configuration.safety_scope.casefold()
    assert "synthetic" in configuration.safety_scope.casefold()


def test_orbit_sandbox_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    source = CONFIGURATION_PATH.read_text(encoding="utf-8")
    invalid = tmp_path / "invalid_orbit.yaml"
    invalid.write_text(source.replace("model:", "unexpected: true\nmodel:"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unexpected"):
        load_orbit_sandbox_configuration(invalid)


def test_orbit_sandbox_requires_explicit_public_safety_scope(tmp_path: Path) -> None:
    source = CONFIGURATION_PATH.read_text(encoding="utf-8")
    unsafe = tmp_path / "unsafe_orbit.yaml"
    unsafe.write_text(source.replace("fictional: true", "fictional: false"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="fictional"):
        load_orbit_sandbox_configuration(unsafe)
