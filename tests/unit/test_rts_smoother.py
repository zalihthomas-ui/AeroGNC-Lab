import numpy as np
import pytest

from aerognc.gnc.rts_smoother import rts_smooth


def _forward_filter() -> tuple[np.ndarray, ...]:
    sample_count = 80
    step_s = 0.2
    transition = np.array([[1.0, step_s], [0.0, 1.0]])
    process_covariance = np.diag([2.0e-4, 2.0e-3])
    measurement_matrix = np.array([[1.0, 0.0]])
    measurement_variance = 0.4**2
    time_s = np.arange(sample_count) * step_s
    truth = np.column_stack((2.0 + 0.7 * time_s, np.full(sample_count, 0.7)))
    measurement = truth[:, 0] + np.random.default_rng(184).normal(0.0, 0.4, sample_count)

    filtered_state = np.empty_like(truth)
    predicted_state = np.empty_like(truth)
    filtered_covariance = np.empty((sample_count, 2, 2))
    predicted_covariance = np.empty((sample_count, 2, 2))
    current_state = np.array([0.0, 0.0])
    current_covariance = np.diag([4.0, 1.0])
    for index in range(sample_count):
        if index:
            current_state = transition @ current_state
            current_covariance = transition @ current_covariance @ transition.T + process_covariance
        predicted_state[index] = current_state
        predicted_covariance[index] = current_covariance
        innovation_variance = float(
            (measurement_matrix @ current_covariance @ measurement_matrix.T)[0, 0]
            + measurement_variance
        )
        gain = current_covariance @ measurement_matrix.T / innovation_variance
        current_state = current_state + gain[:, 0] * (
            measurement[index] - float((measurement_matrix @ current_state)[0])
        )
        residual = np.eye(2) - gain @ measurement_matrix
        current_covariance = (
            residual @ current_covariance @ residual.T + gain * measurement_variance @ gain.T
        )
        filtered_state[index] = current_state
        filtered_covariance[index] = current_covariance
    transitions = np.repeat(transition[None, :, :], sample_count - 1, axis=0)
    return (
        truth,
        filtered_state,
        filtered_covariance,
        predicted_state,
        predicted_covariance,
        transitions,
    )


def test_rts_smoother_preserves_psd_and_reduces_stored_trajectory_rms() -> None:
    (
        truth,
        filtered_state,
        filtered_covariance,
        predicted_state,
        predicted_covariance,
        transitions,
    ) = _forward_filter()
    result = rts_smooth(
        filtered_state,
        filtered_covariance,
        predicted_state,
        predicted_covariance,
        transitions,
    )

    filtered_rms = np.sqrt(np.mean((filtered_state[:, 0] - truth[:, 0]) ** 2))
    smoothed_rms = np.sqrt(np.mean((result.state[:, 0] - truth[:, 0]) ** 2))
    assert smoothed_rms <= filtered_rms
    assert np.min(np.linalg.eigvalsh(result.covariance)) >= -1.0e-12
    np.testing.assert_allclose(result.state[-1], filtered_state[-1])
    np.testing.assert_allclose(result.covariance[-1], filtered_covariance[-1])


def test_rts_smoother_rejects_incompatible_or_singular_histories() -> None:
    history = np.zeros((3, 1))
    covariance = np.ones((3, 1, 1))
    transitions = np.ones((2, 1, 1))
    with pytest.raises(ValueError, match="predicted_state"):
        rts_smooth(history, covariance, history[:2], covariance, transitions)

    singular = covariance.copy()
    singular[1] = 0.0
    with pytest.raises(ValueError, match="nonsingular"):
        rts_smooth(history, covariance, history, singular, transitions)
