import json

import numpy as np
import pytest

from aerognc.data.analysis import (
    compare_telemetry_channels,
    estimate_marker_time_alignment,
    find_telemetry_gaps,
    telemetry_residual_statistics,
    write_aligned_comparison_csv,
    write_telemetry_comparison_report,
)


def test_clock_alignment_gap_aware_comparison_and_report_are_deterministic(tmp_path) -> None:
    reference_markers = np.array([1.0, 6.0, 11.0, 16.0])
    scale = 1.0 + 25.0e-6
    offset_s = 0.31
    sensor_markers = scale * reference_markers + offset_s
    alignment = estimate_marker_time_alignment(reference_markers, sensor_markers)
    assert alignment.offset_s == pytest.approx(offset_s, abs=1.0e-12)
    assert alignment.drift_ppm == pytest.approx(25.0, abs=1.0e-8)

    reference_time = np.linspace(0.0, 20.0, 401)
    expected = np.sin(0.4 * reference_time)
    measurement_time = scale * reference_time + offset_s
    measured = expected + np.random.default_rng(77).normal(0.0, 0.01, reference_time.size)
    measured[150:158] = np.nan
    comparison = compare_telemetry_channels(
        reference_time,
        expected,
        measurement_time,
        measured,
        maximum_gap_s=0.15,
        unit="m",
        clock_alignment=alignment,
        maximum_lag=12,
    )

    assert comparison.statistics.bias == pytest.approx(0.0, abs=2.0e-3)
    assert comparison.statistics.rms < 0.012
    assert any(gap.kind == "missing" and gap.missing_sample_count == 8 for gap in comparison.gaps)
    assert np.count_nonzero(~comparison.finite_pair_mask) >= 8

    csv_path = write_aligned_comparison_csv(comparison, tmp_path / "comparison.csv")
    report_path = write_telemetry_comparison_report(comparison, tmp_path / "report.json")
    assert csv_path.read_text(encoding="utf-8").startswith("time_s,expected")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["clock_alignment"]["drift_ppm"] == pytest.approx(25.0)
    assert payload["coverage"]["finite_pair_count"] == comparison.statistics.sample_count


def test_gap_and_constant_residual_edge_cases() -> None:
    gaps = find_telemetry_gaps(
        [0.0, 0.1, 0.5, 0.6],
        [1.0, np.nan, np.nan, 2.0],
        maximum_gap_s=0.2,
    )
    assert [gap.kind for gap in gaps] == ["missing", "timestamp"]
    assert gaps[0].missing_sample_count == 2

    statistics = telemetry_residual_statistics(np.ones(30), maximum_lag=5)
    assert statistics.bias == pytest.approx(1.0)
    assert statistics.ljung_box_p_value == pytest.approx(1.0)
    assert statistics.white_at_95_percent


def test_alignment_and_residual_inputs_fail_clearly() -> None:
    with pytest.raises(ValueError, match="at least three"):
        estimate_marker_time_alignment([0.0, 1.0], [0.1, 1.1])
    with pytest.raises(ValueError, match="finite residual count"):
        telemetry_residual_statistics([1.0, np.nan, 2.0], maximum_lag=1)
