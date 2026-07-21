"""Batch Rauch--Tung--Striebel smoothing for stored linearised filter histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class RtsSmootherResult:
    """Smoothed state/covariance history and numerical PSD safeguards."""

    state: FloatArray
    covariance: FloatArray
    smoothing_gain: FloatArray
    covariance_projection_count: int
    minimum_eigenvalue_before_projection: float


def _validate_covariance_history(values: FloatArray, label: str) -> None:
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"{label} must have shape (sample_count, state_size, state_size)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite")
    for index, covariance in enumerate(values):
        if not np.allclose(covariance, covariance.T, atol=1.0e-11, rtol=1.0e-11):
            raise ValueError(f"{label}[{index}] must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-10:
            raise ValueError(f"{label}[{index}] must be positive semidefinite")


def _project_positive_semidefinite(covariance: FloatArray) -> tuple[FloatArray, float, bool]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum = float(np.min(eigenvalues))
    projected = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return np.asarray(0.5 * (projected + projected.T)), minimum, minimum < 0.0


def rts_smooth(
    filtered_state: npt.ArrayLike,
    filtered_covariance: npt.ArrayLike,
    predicted_state: npt.ArrayLike,
    predicted_covariance: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
) -> RtsSmootherResult:
    """Run the backward RTS recursion over a stored forward-filter history.

    ``predicted_state[k + 1]`` and ``predicted_covariance[k + 1]`` must be the
    one-step prediction made from filtered sample ``k``. ``transition_matrix[k]``
    maps perturbations from sample ``k`` to ``k + 1``. Index zero of each predicted
    history is retained for convenient log alignment but is not used by the recursion.
    """
    filtered_states = np.asarray(filtered_state, dtype=np.float64)
    predicted_states = np.asarray(predicted_state, dtype=np.float64)
    filtered_covariances = np.asarray(filtered_covariance, dtype=np.float64)
    predicted_covariances = np.asarray(predicted_covariance, dtype=np.float64)
    transitions = np.asarray(transition_matrix, dtype=np.float64)
    if filtered_states.ndim != 2 or filtered_states.shape[0] < 2:
        raise ValueError("RTS state history must be a matrix with at least two samples")
    sample_count, state_size = filtered_states.shape
    if predicted_states.shape != filtered_states.shape:
        raise ValueError("predicted_state must match filtered_state shape")
    expected_covariance_shape = (sample_count, state_size, state_size)
    if filtered_covariances.shape != expected_covariance_shape:
        raise ValueError("filtered_covariance shape does not match state history")
    if predicted_covariances.shape != expected_covariance_shape:
        raise ValueError("predicted_covariance shape does not match state history")
    if transitions.shape != (sample_count - 1, state_size, state_size):
        raise ValueError("transition_matrix must have shape (sample_count - 1, n, n)")
    if not np.all(np.isfinite(filtered_states)) or not np.all(np.isfinite(predicted_states)):
        raise ValueError("RTS state histories must be finite")
    if not np.all(np.isfinite(transitions)):
        raise ValueError("RTS transition matrices must be finite")
    _validate_covariance_history(filtered_covariances, "filtered_covariance")
    _validate_covariance_history(predicted_covariances, "predicted_covariance")

    smoothed_state = filtered_states.copy()
    smoothed_covariance = filtered_covariances.copy()
    gains = np.empty((sample_count - 1, state_size, state_size), dtype=np.float64)
    projection_count = 0
    minimum_before_projection = 0.0
    for index in range(sample_count - 2, -1, -1):
        prediction_covariance = predicted_covariances[index + 1]
        try:
            gain = np.linalg.solve(
                prediction_covariance,
                transitions[index] @ filtered_covariances[index],
            ).T
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"predicted_covariance[{index + 1}] must be nonsingular for RTS smoothing"
            ) from error
        gains[index] = gain
        smoothed_state[index] = filtered_states[index] + gain @ (
            smoothed_state[index + 1] - predicted_states[index + 1]
        )
        candidate_covariance = (
            filtered_covariances[index]
            + gain @ (smoothed_covariance[index + 1] - prediction_covariance) @ gain.T
        )
        projected, minimum, changed = _project_positive_semidefinite(candidate_covariance)
        smoothed_covariance[index] = projected
        minimum_before_projection = min(minimum_before_projection, minimum)
        projection_count += int(changed)
    return RtsSmootherResult(
        smoothed_state,
        smoothed_covariance,
        gains,
        projection_count,
        minimum_before_projection,
    )
