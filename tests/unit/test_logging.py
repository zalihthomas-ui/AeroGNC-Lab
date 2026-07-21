from pathlib import Path

import numpy as np
import pytest

from aerognc.simulation.logging import SimulationResult, write_result_csv, write_summary_json


def _result() -> SimulationResult:
    return SimulationResult(
        scenario_name="test",
        time_s=np.array([0.0, 0.1, 0.2]),
        columns={"altitude_m": np.array([0.0, 1.0, 2.0])},
        events=(),
        event_summary=(),
        maximum_summary={"altitude": {"value": 2.0, "unit": "m", "time_s": 0.2}},
        execution_time_s=0.01,
    )


def test_deterministic_writers(tmp_path: Path) -> None:
    result = _result()
    first_csv = write_result_csv(result, tmp_path / "first.csv")
    second_csv = write_result_csv(result, tmp_path / "second.csv")
    assert first_csv.read_bytes() == second_csv.read_bytes()
    first_json = write_summary_json(result, tmp_path / "first.json")
    second_json = write_summary_json(result, tmp_path / "second.json")
    assert first_json.read_bytes() == second_json.read_bytes()
    assert b"execution_time" not in first_json.read_bytes()


def test_result_rejects_nonmonotonic_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SimulationResult("bad", np.array([0.0, 0.0]), {}, (), (), {}, 0.0)
