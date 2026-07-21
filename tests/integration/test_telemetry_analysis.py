"""Stored-file telemetry reconstruction and backward-smoothing integration case."""

import csv

import numpy as np

from aerognc.data.analysis import (
    compare_telemetry_channels,
    estimate_marker_time_alignment,
    write_aligned_comparison_csv,
    write_telemetry_comparison_report,
)
from aerognc.data.telemetry import (
    ChannelMapping,
    QualityMapping,
    TelemetryMapping,
    TimestampMapping,
    import_telemetry_csv,
    write_normalized_telemetry_csv,
    write_telemetry_provenance,
)
from aerognc.gnc.rts_smoother import rts_smooth


def test_csv_reload_clock_alignment_residual_report_and_stored_rts_smoothing(tmp_path) -> None:
    reference_time_s = np.linspace(0.0, 12.0, 121)
    truth_altitude_m = np.full(reference_time_s.shape, 100.0)
    scale = 1.0 + 18.0e-6
    offset_s = 0.27
    sensor_time_s = scale * reference_time_s + offset_s
    measured_altitude_m = truth_altitude_m + np.random.default_rng(902).normal(
        0.0, 1.2, reference_time_s.size
    )
    source = tmp_path / "raw_telemetry.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sensor_time_s", "quality", "altitude_m"))
        for index, values in enumerate(zip(sensor_time_s, measured_altitude_m, strict=True)):
            quality = "BAD" if 50 <= index <= 54 else "OK"
            writer.writerow((f"{values[0]:.12g}", quality, f"{values[1]:.12g}"))
    mapping = TelemetryMapping(
        "1.0",
        TimestampMapping("sensor_time_s", "s", 1.0),
        QualityMapping("quality", ("OK",), "keep_nan"),
        (ChannelMapping("altitude_m", "altitude_m", "m", "m", missing_policy="keep_nan"),),
    )

    record = import_telemetry_csv(source, mapping)
    write_normalized_telemetry_csv(record, tmp_path / "normalized_telemetry.csv")
    write_telemetry_provenance(record, tmp_path / "telemetry_provenance.json")
    marker_indices = np.array([0, 40, 80, 120])
    alignment = estimate_marker_time_alignment(
        reference_time_s[marker_indices], record.time_s[marker_indices]
    )
    comparison = compare_telemetry_channels(
        reference_time_s,
        truth_altitude_m,
        record.time_s,
        record.channels["altitude_m"],
        maximum_gap_s=0.25,
        unit="m",
        clock_alignment=alignment,
        maximum_lag=10,
    )
    write_aligned_comparison_csv(comparison, tmp_path / "aligned_comparison.csv")
    write_telemetry_comparison_report(comparison, tmp_path / "telemetry_report.json")
    assert comparison.statistics.rms < 1.5
    assert any(gap.kind == "missing" for gap in comparison.gaps)

    sample_count = reference_time_s.size
    filtered_state = np.empty((sample_count, 1))
    predicted_state = np.empty((sample_count, 1))
    filtered_covariance = np.empty((sample_count, 1, 1))
    predicted_covariance = np.empty((sample_count, 1, 1))
    estimate = 90.0
    variance = 25.0
    process_variance = 0.015
    measurement_variance = 1.2**2
    for index, measurement in enumerate(comparison.aligned_measured):
        if index:
            variance += process_variance
        predicted_state[index, 0] = estimate
        predicted_covariance[index, 0, 0] = variance
        if np.isfinite(measurement):
            gain = variance / (variance + measurement_variance)
            estimate += gain * (measurement - estimate)
            variance = (1.0 - gain) * variance
        filtered_state[index, 0] = estimate
        filtered_covariance[index, 0, 0] = variance
    transition = np.ones((sample_count - 1, 1, 1))
    history_path = tmp_path / "forward_filter_history.npz"
    np.savez(
        history_path,
        filtered_state=filtered_state,
        filtered_covariance=filtered_covariance,
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        transition=transition,
    )
    with np.load(history_path, allow_pickle=False) as stored:
        smoothed = rts_smooth(
            stored["filtered_state"],
            stored["filtered_covariance"],
            stored["predicted_state"],
            stored["predicted_covariance"],
            stored["transition"],
        )
    filtered_rms = np.sqrt(np.mean((filtered_state[:, 0] - truth_altitude_m) ** 2))
    smoothed_rms = np.sqrt(np.mean((smoothed.state[:, 0] - truth_altitude_m) ** 2))
    assert smoothed_rms <= filtered_rms
    assert np.min(np.linalg.eigvalsh(smoothed.covariance)) >= -1.0e-12
