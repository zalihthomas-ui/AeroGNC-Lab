from pathlib import Path

import pytest

from aerognc.configuration import ConfigurationError
from aerognc.configuration.launch_window_loader import load_launch_window_configuration

CONFIGURATION_PATH = Path("configs/launch_window_optimization.yaml")


def test_launch_window_configuration_loads_strict_case() -> None:
    configuration = load_launch_window_configuration(CONFIGURATION_PATH)
    assert configuration.departure_body == "Asteria"
    assert configuration.destination_body == "Neria"
    assert configuration.departure_grid_count == 9
    assert configuration.maximum_total_delta_v_mps == pytest.approx(7_500.0)


def test_launch_window_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    source = CONFIGURATION_PATH.read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("route:\n", "extra: true\nroute:\n"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_launch_window_configuration(path)
