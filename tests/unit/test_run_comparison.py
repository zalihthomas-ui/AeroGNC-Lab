import numpy as np
import pytest

from aerognc.project.comparison import compare_datasets, write_comparison_json
from aerognc.project.result_store import ResultDataset


def _case(name: str, time_s: np.ndarray, offset: float, unit: str = "m") -> ResultDataset:
    return ResultDataset(
        scenario_name=name,
        time_s=time_s,
        channels={"altitude_m": 2.0 * time_s + offset},
        units={"altitude_m": unit},
    )


def test_comparison_aligns_different_time_bases_and_reports_metrics(tmp_path) -> None:
    baseline = _case("baseline", np.linspace(0.0, 2.0, 5), 0.0)
    candidate = _case("candidate", np.linspace(0.5, 2.5, 9), 0.25)

    comparison = compare_datasets(baseline, candidate, sample_count=7)

    assert comparison.start_time_s == 0.5
    assert comparison.end_time_s == 2.0
    metric = comparison.channels[0]
    assert metric.bias == pytest.approx(0.25)
    assert metric.rms_difference == pytest.approx(0.25)
    assert metric.maximum_absolute_difference == pytest.approx(0.25)
    assert metric.final_difference == pytest.approx(0.25)
    assert metric.correlation == pytest.approx(1.0)
    path = write_comparison_json(comparison, tmp_path / "comparison.json")
    assert '"sample_count": 7' in path.read_text(encoding="utf-8")


def test_comparison_rejects_missing_channels_units_and_time_overlap() -> None:
    baseline = _case("baseline", np.array([0.0, 1.0]), 0.0)
    with pytest.raises(KeyError, match="missing"):
        compare_datasets(baseline, baseline, channels=["speed_mps"])
    with pytest.raises(ValueError, match="unit mismatch"):
        compare_datasets(baseline, _case("candidate", np.array([0.0, 1.0]), 0.0, "km"))
    with pytest.raises(ValueError, match="no common"):
        compare_datasets(baseline, _case("late", np.array([2.0, 3.0]), 0.0))
    with pytest.raises(ValueError, match="at least two"):
        compare_datasets(baseline, baseline, sample_count=1)
