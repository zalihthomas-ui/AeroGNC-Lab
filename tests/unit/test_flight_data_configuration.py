from pathlib import Path

import pytest

from aerognc.configuration import ConfigurationError
from aerognc.configuration.flight_data_loader import (
    load_flight_data_identification_configuration,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_flight_data_configuration_loads_si_case() -> None:
    configuration = load_flight_data_identification_configuration(
        PROJECT_ROOT / "configs" / "flight_data_identification.yaml"
    )

    assert configuration.name == "synthetic_pitch_flight_data_identification"
    assert configuration.plant.inertia_kgm2 == pytest.approx(12.8)
    assert configuration.command_sample_rate_hz == pytest.approx(50.0)
    assert configuration.sensor_sample_rate_hz == pytest.approx(47.0)
    assert len(configuration.outliers) == 4


def test_flight_data_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs" / "flight_data_identification.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("plant:\n", "unknown: true\nplant:\n"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_flight_data_identification_configuration(path)
