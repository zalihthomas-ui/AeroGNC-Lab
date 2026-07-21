import numpy as np
import pytest

from aerognc.mathematics.signal_processing import (
    estimate_affine_clock_alignment,
    hampel_filter,
    local_polynomial_smooth_derivative,
    resample_with_gap_policy,
)


def _markers(time_s: np.ndarray, centres_s: tuple[float, ...]) -> np.ndarray:
    values = np.zeros_like(time_s)
    for centre_s in centres_s:
        values += np.exp(-0.5 * ((time_s - centre_s) / 0.025) ** 2)
    return values


def test_affine_clock_alignment_recovers_offset_and_drift() -> None:
    reference_time_s = np.arange(0.0, 12.0, 0.002)
    scale = 1.0 + 72.0e-6
    offset_s = 0.315
    sensor_time_s = scale * reference_time_s + offset_s
    centres_s = (1.0, 3.5, 6.0, 8.5, 11.0)

    alignment = estimate_affine_clock_alignment(
        reference_time_s,
        _markers(reference_time_s, centres_s),
        sensor_time_s,
        _markers(reference_time_s, centres_s),
        threshold=0.2,
    )

    assert alignment.offset_s == pytest.approx(offset_s, abs=5.0e-5)
    assert alignment.drift_ppm == pytest.approx(72.0, abs=1.0)
    assert alignment.sensor_to_reference(sensor_time_s) == pytest.approx(reference_time_s)


def test_gap_aware_resampling_does_not_bridge_missing_interval() -> None:
    source_time_s = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    source_values = 2.0 * source_time_s
    destination_time_s = np.arange(0.0, 1.01, 0.1)

    result = resample_with_gap_policy(
        source_time_s,
        source_values,
        destination_time_s,
        maximum_gap_s=0.2,
    )

    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(2.0)
    assert np.all(np.isnan(result[3:8]))


def test_detrended_hampel_filter_replaces_isolated_spike() -> None:
    index = np.arange(81, dtype=np.float64)
    values = 0.004 * index**2 - 0.2 * index + 1.0
    values[40] += 9.0

    filtered, outliers = hampel_filter(values, half_window=7, threshold_sigma=5.0)

    assert np.flatnonzero(outliers).tolist() == [40]
    assert filtered[40] == pytest.approx(0.004 * 40.0**2 - 0.2 * 40.0 + 1.0)


def test_local_polynomial_returns_cubic_value_and_derivative() -> None:
    time_s = np.linspace(-1.0, 1.0, 101)
    values = 0.5 + 2.0 * time_s - 0.3 * time_s**2 + 0.2 * time_s**3
    expected_derivative = 2.0 - 0.6 * time_s + 0.6 * time_s**2

    smooth, derivative = local_polynomial_smooth_derivative(
        time_s,
        values,
        window=15,
        polynomial_order=3,
    )

    assert smooth == pytest.approx(values, abs=2.0e-12)
    assert derivative == pytest.approx(expected_derivative, abs=2.0e-11)
