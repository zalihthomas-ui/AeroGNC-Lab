"""Inspectable time-alignment, resampling, outlier, and local-polynomial utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class AffineClockAlignment:
    """Affine map ``sensor_time = scale * reference_time + offset``."""

    scale: float
    offset_s: float
    drift_ppm: float
    marker_count: int
    marker_fit_rms_s: float
    reference_marker_times_s: FloatArray
    sensor_marker_times_s: FloatArray

    def sensor_to_reference(self, sensor_time_s: npt.ArrayLike) -> FloatArray:
        """Map sensor-clock timestamps onto the reference clock."""
        values = np.asarray(sensor_time_s, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("sensor clock values must be finite")
        return np.asarray((values - self.offset_s) / self.scale, dtype=np.float64)


def marker_centres(
    time_s: npt.ArrayLike,
    marker: npt.ArrayLike,
    *,
    threshold: float,
) -> FloatArray:
    """Return amplitude-weighted centres of separated synchronization markers."""
    time = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(marker, dtype=np.float64)
    if time.ndim != 1 or values.shape != time.shape or time.size < 3:
        raise ValueError("marker time and value arrays must be matching vectors")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("marker timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("marker signal must be finite")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("marker threshold must be positive and finite")
    active = values >= threshold
    transitions = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    centres: list[float] = []
    for start, stop in zip(starts, stops, strict=True):
        weights = np.maximum(values[start:stop] - threshold, 0.0)
        if np.sum(weights) <= 0.0:
            continue
        centres.append(float(np.sum(time[start:stop] * weights) / np.sum(weights)))
    return np.asarray(centres, dtype=np.float64)


def estimate_affine_clock_alignment(
    reference_time_s: npt.ArrayLike,
    reference_marker: npt.ArrayLike,
    sensor_time_s: npt.ArrayLike,
    sensor_marker: npt.ArrayLike,
    *,
    threshold: float,
) -> AffineClockAlignment:
    """Fit sensor clock offset/drift from common marker centres."""
    reference_centres = marker_centres(reference_time_s, reference_marker, threshold=threshold)
    sensor_centres = marker_centres(sensor_time_s, sensor_marker, threshold=threshold)
    if reference_centres.size != sensor_centres.size or reference_centres.size < 3:
        raise ValueError("reference and sensor logs must contain the same 3+ sync markers")
    design = np.column_stack((reference_centres, np.ones(reference_centres.size)))
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        sensor_centres,
        rcond=None,
    )
    if rank != 2 or coefficients[0] <= 0.0:
        raise ValueError("synchronization markers do not define a valid affine clock")
    fitted = design @ coefficients
    residual = sensor_centres - fitted
    scale = float(coefficients[0])
    return AffineClockAlignment(
        scale=scale,
        offset_s=float(coefficients[1]),
        drift_ppm=(scale - 1.0) * 1.0e6,
        marker_count=int(reference_centres.size),
        marker_fit_rms_s=float(np.sqrt(np.mean(residual**2))),
        reference_marker_times_s=reference_centres,
        sensor_marker_times_s=sensor_centres,
    )


def resample_with_gap_policy(
    source_time_s: npt.ArrayLike,
    source_values: npt.ArrayLike,
    destination_time_s: npt.ArrayLike,
    *,
    maximum_gap_s: float,
) -> FloatArray:
    """Linearly resample finite values without bridging declared long gaps."""
    source_time = np.asarray(source_time_s, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    destination = np.asarray(destination_time_s, dtype=np.float64)
    if source_time.ndim != 1 or values.shape != source_time.shape or destination.ndim != 1:
        raise ValueError("resampling inputs must be one-dimensional with matching source shapes")
    if not np.all(np.isfinite(source_time)) or not np.all(np.diff(source_time) > 0.0):
        raise ValueError("source timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(destination)) or np.any(np.diff(destination) < 0.0):
        raise ValueError("destination timestamps must be finite and nondecreasing")
    if not np.isfinite(maximum_gap_s) or maximum_gap_s <= 0.0:
        raise ValueError("maximum_gap_s must be positive and finite")
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return np.full(destination.shape, np.nan)
    valid_time = source_time[finite]
    valid_values = values[finite]
    result = np.interp(destination, valid_time, valid_values, left=np.nan, right=np.nan)
    right = np.searchsorted(valid_time, destination, side="left")
    bracketed = (right > 0) & (right < valid_time.size)
    safe_right = np.clip(right, 1, valid_time.size - 1)
    gaps = valid_time[safe_right] - valid_time[safe_right - 1]
    result[~bracketed | (gaps > maximum_gap_s)] = np.nan
    exact_first = np.isclose(destination, valid_time[0], atol=1.0e-12)
    exact_last = np.isclose(destination, valid_time[-1], atol=1.0e-12)
    result[exact_first] = valid_values[0]
    result[exact_last] = valid_values[-1]
    return np.asarray(result, dtype=np.float64)


def hampel_filter(
    values: npt.ArrayLike,
    *,
    half_window: int,
    threshold_sigma: float,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Replace isolated detrended median-absolute-deviation outliers.

    A local quadratic is fitted without the candidate sample before the Hampel
    threshold is applied. This avoids mistaking a legitimate high signal slope for
    an outlier while preserving the robust median-absolute-deviation scale.
    """
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Hampel input must be one-dimensional")
    if half_window <= 0 or not np.isfinite(threshold_sigma) or threshold_sigma <= 0.0:
        raise ValueError("Hampel window and threshold must be positive")
    filtered = signal.copy()
    outlier = np.zeros(signal.shape, dtype=np.bool_)
    for index, value in enumerate(signal):
        if not np.isfinite(value):
            continue
        if index < half_window or index + half_window >= signal.size:
            continue
        start = max(0, index - half_window)
        stop = min(signal.size, index + half_window + 1)
        window = signal[start:stop]
        local_index = np.arange(start, stop, dtype=np.float64) - index
        finite = np.isfinite(window) & (local_index != 0.0)
        if np.count_nonzero(finite) != 2 * half_window:
            # Do not classify samples at a declared gap edge as isolated spikes.
            continue
        if np.count_nonzero(finite) < max(6, half_window):
            continue
        design = np.column_stack(
            (
                np.ones(np.count_nonzero(finite)),
                local_index[finite],
                local_index[finite] ** 2,
            )
        )
        coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
            design,
            window[finite],
            rcond=None,
        )
        if rank != 3:
            continue
        neighbour_residual = window[finite] - design @ coefficients
        median_residual = float(np.median(neighbour_residual))
        robust_sigma = 1.4826 * float(np.median(np.abs(neighbour_residual - median_residual)))
        predicted = float(coefficients[0])
        numerical_scale = max(
            abs(float(value)),
            abs(predicted),
            float(np.max(np.abs(window[finite]))),
        )
        # For exactly polynomial data, the MAD can collapse to solver noise.
        # Bound that noise using the fitted design's condition number.
        design_condition = float(singular_values[0] / singular_values[-1])
        least_squares_roundoff = np.finfo(np.float64).eps * design_condition * numerical_scale
        outlier_threshold = max(
            threshold_sigma * robust_sigma,
            least_squares_roundoff,
        )
        if abs(value - predicted) > outlier_threshold:
            filtered[index] = predicted
            outlier[index] = True
    return filtered, outlier


def local_polynomial_smooth_derivative(
    time_s: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    window: int,
    polynomial_order: int,
) -> tuple[FloatArray, FloatArray]:
    """Return local-polynomial value and first derivative at each finite sample."""
    time = np.asarray(time_s, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    if time.ndim != 1 or signal.shape != time.shape or time.size < window:
        raise ValueError(
            "local-polynomial inputs must be matching vectors at least one window long"
        )
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("local-polynomial timestamps must be finite and increasing")
    if window < 3 or window % 2 == 0 or not 1 <= polynomial_order < window:
        raise ValueError("local-polynomial window must be odd and exceed a positive order")
    smooth = np.full(signal.shape, np.nan)
    derivative = np.full(signal.shape, np.nan)
    half_window = window // 2
    for index, value in enumerate(signal):
        if not np.isfinite(value):
            continue
        start = max(0, index - half_window)
        stop = min(signal.size, index + half_window + 1)
        local_time = time[start:stop] - time[index]
        local_values = signal[start:stop]
        finite = np.isfinite(local_values)
        if np.count_nonzero(finite) < polynomial_order + 2:
            continue
        design = np.vander(local_time[finite], polynomial_order + 1, increasing=True)
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
            design,
            local_values[finite],
            rcond=None,
        )
        if rank == polynomial_order + 1:
            smooth[index] = coefficients[0]
            derivative[index] = coefficients[1]
    return smooth, derivative
