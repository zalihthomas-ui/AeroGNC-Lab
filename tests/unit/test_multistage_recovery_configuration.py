from pathlib import Path

import pytest
import yaml

from aerognc.configuration import load_multistage_recovery_configuration


def test_multistage_recovery_configuration_is_readable_and_fictional() -> None:
    configuration = load_multistage_recovery_configuration("configs/multistage_recovery.yaml")

    assert configuration.name == "fictional_two_stage_recovery_demo"
    assert "Fictional civilian" in configuration.safety_scope
    assert [stage.name for stage in configuration.vehicle.stages] == ["booster", "sustainer"]
    assert configuration.recovery.full_area_m2 == pytest.approx(1.0)
    assert configuration.step_s == pytest.approx(0.01)


def test_multistage_recovery_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/multistage_recovery.yaml").read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_multistage_recovery_configuration(path)
