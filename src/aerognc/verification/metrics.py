"""Quantitative control-response metrics."""

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class StepResponseMetrics:
    """Controller comparison metrics in SI units and seconds."""

    rise_time_s: float
    settling_time_s: float
    overshoot_percent: float
    rms_tracking_error_rad: float
    maximum_tracking_error_rad: float
    control_effort_nm2s: float
    actuator_saturation_duration_s: float
    disturbance_recovery_s: float
    execution_time_s: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-friendly metric mapping."""
        return asdict(self)


def _first_crossing(time_s: np.ndarray, values: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(values >= threshold)
    return float("nan") if indices.size == 0 else float(time_s[indices[0]])


def step_response_metrics(
    time_s: npt.ArrayLike,
    reference_rad: npt.ArrayLike,
    response_rad: npt.ArrayLike,
    control_moment_nm: npt.ArrayLike,
    saturated: npt.ArrayLike,
    *,
    command_time_s: float,
    disturbance_start_time_s: float,
    disturbance_end_time_s: float,
    execution_time_s: float,
    settling_fraction: float = 0.02,
) -> StepResponseMetrics:
    """Calculate consistent step/tracking, effort, and recovery metrics."""
    time = np.asarray(time_s, dtype=np.float64)
    reference = np.asarray(reference_rad, dtype=np.float64)
    response = np.asarray(response_rad, dtype=np.float64)
    control = np.asarray(control_moment_nm, dtype=np.float64)
    saturation = np.asarray(saturated, dtype=bool)
    if any(array.shape != time.shape for array in (reference, response, control, saturation)):
        raise ValueError("metric arrays must have identical shapes")
    if time.ndim != 1 or time.size < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be a strictly increasing one-dimensional array")
    if not np.all(np.isfinite([time, reference, response, control])):
        raise ValueError("metric signals must be finite")
    command_mask = time >= command_time_s
    if not np.any(command_mask):
        raise ValueError("command_time_s lies outside the data")
    final_reference = float(reference[-1])
    initial_response = float(response[np.flatnonzero(command_mask)[0]])
    step_amplitude = final_reference - initial_response
    if abs(step_amplitude) <= 1.0e-12:
        raise ValueError("response does not contain a nonzero step")
    direction = np.sign(step_amplitude)
    normalised = direction * (response[command_mask] - initial_response)
    command_time = time[command_mask]
    amplitude = abs(step_amplitude)
    t10 = _first_crossing(command_time, normalised, 0.1 * amplitude)
    t90 = _first_crossing(command_time, normalised, 0.9 * amplitude)
    rise_time = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")

    error = reference - response
    tolerance = settling_fraction * max(abs(final_reference), amplitude)
    settling_time = float("nan")
    pre_disturbance = time < disturbance_start_time_s
    response_window = command_mask & pre_disturbance
    post_indices = np.flatnonzero(response_window)
    disturbance_index = int(np.searchsorted(time, disturbance_start_time_s, side="left"))
    for index in post_indices:
        if np.all(np.abs(error[index:disturbance_index]) <= tolerance):
            settling_time = float(time[index] - command_time_s)
            break
    peak_progress = float(np.max(direction * (response[response_window] - initial_response)))
    overshoot = max(0.0, (peak_progress - amplitude) / amplitude * 100.0)

    recovery_time = float("nan")
    recovery_indices = np.flatnonzero(time >= disturbance_end_time_s)
    for index in recovery_indices:
        if np.all(np.abs(error[index:]) <= tolerance):
            recovery_time = float(time[index] - disturbance_end_time_s)
            break
    dt = np.diff(time)
    control_effort = float(np.sum(0.5 * (control[:-1] ** 2 + control[1:] ** 2) * dt))
    saturation_duration = float(np.sum(dt * saturation[:-1]))
    return StepResponseMetrics(
        rise_time_s=rise_time,
        settling_time_s=settling_time,
        overshoot_percent=overshoot,
        rms_tracking_error_rad=float(np.sqrt(np.mean(error[command_mask] ** 2))),
        maximum_tracking_error_rad=float(np.max(np.abs(error[command_mask]))),
        control_effort_nm2s=control_effort,
        actuator_saturation_duration_s=saturation_duration,
        disturbance_recovery_s=recovery_time,
        execution_time_s=float(execution_time_s),
    )
