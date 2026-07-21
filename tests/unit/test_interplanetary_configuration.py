from pathlib import Path

import pytest
import yaml

from aerognc.configuration import ConfigurationError, load_interplanetary_configuration

CONFIGURATION_PATH = Path("configs/interplanetary_gravity_assist.yaml")


def test_interplanetary_configuration_is_strict_fictional_and_role_complete() -> None:
    configuration = load_interplanetary_configuration(CONFIGURATION_PATH)
    assert "fictional" in configuration.safety_scope.casefold()
    assert "synthetic" in configuration.safety_scope.casefold()
    assert configuration.body_with_role("departure").name == "Asteria"
    assert configuration.body_with_role("assist").name == "Brontes"
    assert configuration.body_with_role("destination").name == "Caelus"
    assert configuration.spacecraft.reference_body == "Asteria"


def test_interplanetary_configuration_rejects_unsafe_scope_and_coarse_step(
    tmp_path: Path,
) -> None:
    data = yaml.safe_load(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    data["metadata"]["safety_scope"] = "generic scenario"
    unsafe_path = tmp_path / "unsafe.yaml"
    unsafe_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="fictional, civilian, and synthetic"):
        load_interplanetary_configuration(unsafe_path)

    data = yaml.safe_load(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    data["mission"]["step_s"] = 30_000.0
    coarse_path = tmp_path / "coarse.yaml"
    coarse_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot exceed six hours"):
        load_interplanetary_configuration(coarse_path)
