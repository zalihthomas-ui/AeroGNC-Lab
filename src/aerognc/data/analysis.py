"""Asynchronous telemetry alignment, gap accounting, and residual evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2  # type: ignore[import-untyped]

from aerognc.mathematics.signal_processing import (
    AffineClockAlignment,
    resample_with_gap_policy,
)
from aerognc.mathematics.vectors import FloatArray

GapKind = Literal["timestamp", "missing"]


@dataclass(frozen=True, slots=True)
class TelemetryGap:
    """One source-time discontinuity or consecutive missing-value interval."""

    kind: GapKind
    start_time_s: float
    end_time_s: float
    duration_s: float
    missing_sample_count: int


@dataclass(frozen=True, slots=True)
class TelemetryResidualStatistics:
    """Finite-pair residual magnitude and whiteness diagnostics."""

    sample_count: int
    bias: float
    rms: float
    standard_deviation: float
    maximum_absolute: float
    autocorrelation: FloatArray
    maximum_absolute_autocorrelation: float
    ljung_box_q: float
    ljung_box_p_value: float
    white_at_95_percent: bool


@dataclass(frozen=True, slots=True)
class TelemetryComparison:
    """Expected and clock-aligned measured values on one reference timeline."""

    reference_time_s: FloatArray
    expected: FloatArray
    aligned_measured: FloatArray
    residual: FloatArray
    finite_pair_mask: npt.NDArray[np.bool_]
    gaps: tuple[TelemetryGap, ...]
    statistics: TelemetryResidualStatistics
    clock_alignment: AffineClockAlignment | None
    unit: str


def estimate_marker_time_alignment(
    reference_marker_times_s: npt.ArrayLike,
    sensor_marker_times_s: npt.ArrayLike,
) -> AffineClockAlignment:
    """Fit ``sensor = scale * reference + offset`` from 3+ paired marker epochs."""
    reference = np.asarray(reference_marker_times_s, dtype=np.float64)
    sensor = np.asarray(sensor_marker_times_s, dtype=np.float64)
    if reference.ndim != 1 or sensor.shape != reference.shape or reference.size < 3:
        raise ValueError("clock alignment requires matching vectors of at least three markers")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(sensor)):
        raise ValueError("clock alignment marker epochs must be finite")
    if np.any(np.diff(reference) <= 0.0) or np.any(np.diff(sensor) <= 0.0):
        raise ValueError("clock alignment marker epochs must be strictly increasing")
    design = np.column_stack((reference, np.ones(reference.size)))
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(design, sensor, rcond=None)
    scale = float(coefficients[0])
    if rank != 2 or scale <= 0.0:
        raise ValueError("clock marker correspondence does not define a valid affine map")
    residual = sensor - design @ coefficients
    return AffineClockAlignment(
        scale,
        float(coefficients[1]),
        (scale - 1.0) * 1.0e6,
        int(reference.size),
        float(np.sqrt(np.mean(residual**2))),
        reference.copy(),
        sensor.copy(),
    )


def find_telemetry_gaps(
    time_s: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    maximum_gap_s: float,
) -> tuple[TelemetryGap, ...]:
    """Return timestamp discontinuities and finite-length missing-value runs."""
    time = np.asarray(time_s, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    if time.ndim != 1 or signal.shape != time.shape or time.size < 2:
        raise ValueError("gap inputs must be matching vectors with at least two samples")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("gap timestamps must be finite and strictly increasing")
    if not np.isfinite(maximum_gap_s) or maximum_gap_s <= 0.0:
        raise ValueError("maximum_gap_s must be positive and finite")
    gaps: list[TelemetryGap] = []
    for index in np.flatnonzero(np.diff(time) > maximum_gap_s):
        start = float(time[index])
        stop = float(time[index + 1])
        gaps.append(TelemetryGap("timestamp", start, stop, stop - start, 0))
    missing = ~np.isfinite(signal)
    transitions = np.diff(np.concatenate(([False], missing, [False])).astype(np.int8))
    for start_index, stop_index in zip(
        np.flatnonzero(transitions == 1),
        np.flatnonzero(transitions == -1),
        strict=True,
    ):
        start = float(time[start_index])
        stop = float(time[stop_index - 1])
        gaps.append(
            TelemetryGap("missing", start, stop, stop - start, int(stop_index - start_index))
        )
    gaps.sort(key=lambda item: (item.start_time_s, item.kind, item.end_time_s))
    return tuple(gaps)


def telemetry_residual_statistics(
    residual: npt.ArrayLike,
    *,
    maximum_lag: int = 20,
) -> TelemetryResidualStatistics:
    """Calculate magnitude, ACF, and Ljung--Box statistics from finite residuals."""
    raw = np.asarray(residual, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("residual must be a one-dimensional sequence")
    values = raw[np.isfinite(raw)]
    if maximum_lag <= 0 or values.size <= maximum_lag + 2:
        raise ValueError("finite residual count must exceed maximum_lag + 2")
    centred = values - np.mean(values)
    variance_sum = float(centred @ centred)
    if variance_sum <= np.finfo(np.float64).eps:
        autocorrelation = np.concatenate(([1.0], np.zeros(maximum_lag)))
        ljung_box_q = 0.0
        ljung_box_p = 1.0
    else:
        autocorrelation = np.array(
            [1.0]
            + [
                float(centred[lag:] @ centred[:-lag] / variance_sum)
                for lag in range(1, maximum_lag + 1)
            ]
        )
        sample_count = values.size
        lags = np.arange(1, maximum_lag + 1)
        ljung_box_q = float(
            sample_count
            * (sample_count + 2)
            * np.sum(autocorrelation[1:] ** 2 / (sample_count - lags))
        )
        ljung_box_p = float(chi2.sf(ljung_box_q, maximum_lag))
    return TelemetryResidualStatistics(
        int(values.size),
        float(np.mean(values)),
        float(np.sqrt(np.mean(values**2))),
        float(np.std(values, ddof=1)),
        float(np.max(np.abs(values))),
        autocorrelation,
        float(np.max(np.abs(autocorrelation[1:]))),
        ljung_box_q,
        ljung_box_p,
        ljung_box_p >= 0.05,
    )


def compare_telemetry_channels(
    reference_time_s: npt.ArrayLike,
    expected: npt.ArrayLike,
    measurement_time_s: npt.ArrayLike,
    measured: npt.ArrayLike,
    *,
    maximum_gap_s: float,
    unit: str,
    clock_alignment: AffineClockAlignment | None = None,
    maximum_lag: int = 20,
) -> TelemetryComparison:
    """Clock-correct, gap-aware resample, and compare one stored channel."""
    reference_time = np.asarray(reference_time_s, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    measurement_time = np.asarray(measurement_time_s, dtype=np.float64)
    measured_values = np.asarray(measured, dtype=np.float64)
    if reference_time.ndim != 1 or expected_values.shape != reference_time.shape:
        raise ValueError("reference time and expected values must be matching vectors")
    if not np.all(np.isfinite(reference_time)) or not np.all(np.isfinite(expected_values)):
        raise ValueError("reference time and expected values must be finite")
    if np.any(np.diff(reference_time) <= 0.0):
        raise ValueError("reference timestamps must be strictly increasing")
    if measurement_time.ndim != 1 or measured_values.shape != measurement_time.shape:
        raise ValueError("measurement time and values must be matching vectors")
    if not unit.strip():
        raise ValueError("comparison unit cannot be empty")
    corrected_time = (
        measurement_time.copy()
        if clock_alignment is None
        else clock_alignment.sensor_to_reference(measurement_time)
    )
    gaps = find_telemetry_gaps(corrected_time, measured_values, maximum_gap_s=maximum_gap_s)
    aligned = resample_with_gap_policy(
        corrected_time,
        measured_values,
        reference_time,
        maximum_gap_s=maximum_gap_s,
    )
    residual = aligned - expected_values
    finite_pair = np.isfinite(residual)
    statistics = telemetry_residual_statistics(residual, maximum_lag=maximum_lag)
    return TelemetryComparison(
        reference_time.copy(),
        expected_values.copy(),
        aligned,
        residual,
        finite_pair,
        gaps,
        statistics,
        clock_alignment,
        unit,
    )


def write_aligned_comparison_csv(comparison: TelemetryComparison, path: str | Path) -> Path:
    """Write the deterministic sample-level comparison evidence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("time_s", "expected", "measured_aligned", "residual", "finite_pair"))
        for values in zip(
            comparison.reference_time_s,
            comparison.expected,
            comparison.aligned_measured,
            comparison.residual,
            comparison.finite_pair_mask,
            strict=True,
        ):
            writer.writerow(
                (
                    f"{float(values[0]):.12g}",
                    f"{float(values[1]):.12g}",
                    f"{float(values[2]):.12g}",
                    f"{float(values[3]):.12g}",
                    "1" if values[4] else "0",
                )
            )
    return destination


def write_telemetry_comparison_report(comparison: TelemetryComparison, path: str | Path) -> Path:
    """Write a deterministic, summary-only JSON analysis report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clock = comparison.clock_alignment
    payload = {
        "schema_version": "1.0",
        "unit": comparison.unit,
        "coverage": {
            "start_time_s": float(comparison.reference_time_s[0]),
            "end_time_s": float(comparison.reference_time_s[-1]),
            "sample_count": int(comparison.reference_time_s.size),
            "finite_pair_count": int(np.count_nonzero(comparison.finite_pair_mask)),
        },
        "clock_alignment": None
        if clock is None
        else {
            "scale": clock.scale,
            "offset_s": clock.offset_s,
            "drift_ppm": clock.drift_ppm,
            "marker_count": clock.marker_count,
            "marker_fit_rms_s": clock.marker_fit_rms_s,
        },
        "gaps": [asdict(gap) for gap in comparison.gaps],
        "residual_statistics": {
            key: value
            for key, value in asdict(comparison.statistics).items()
            if key != "autocorrelation"
        },
        "autocorrelation": comparison.statistics.autocorrelation.tolist(),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
