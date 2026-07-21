import numpy as np
import pytest

from aerognc.gnc.error_state_ekf import (
    ErrorStateFilterTuning,
    ErrorStateNavigationEKF,
    NavigationNominalState,
)
from aerognc.mathematics.quaternion import quaternion_to_euler321


def _filter(position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> ErrorStateNavigationEKF:
    return ErrorStateNavigationEKF(
        NavigationNominalState(position, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        np.diag(
            [
                *([100.0] * 3),
                *([10.0] * 3),
                *([0.1] * 3),
                *([1.0e-4] * 3),
                *([1.0e-2] * 3),
            ]
        ),
        ErrorStateFilterTuning(1.0e-3, 0.02, 1.0e-5, 1.0e-4),
    )


def test_stationary_strapdown_prediction_preserves_state_and_covariance_psd() -> None:
    navigation_filter = _filter()
    for _index in range(100):
        navigation_filter.predict([0.0, 0.0, 0.0], [0.0, 0.0, -9.80665], 0.01)

    np.testing.assert_allclose(navigation_filter.state.position_ned_m, np.zeros(3), atol=1.0e-10)
    np.testing.assert_allclose(navigation_filter.state.velocity_ned_mps, np.zeros(3), atol=1.0e-10)
    assert np.linalg.norm(navigation_filter.state.quaternion_nb) == pytest.approx(1.0)
    assert np.min(np.linalg.eigvalsh(navigation_filter.covariance)) >= -1.0e-12


def test_gyro_propagation_and_gnss_barometer_updates_reduce_errors() -> None:
    navigation_filter = _filter((100.0, -50.0, 20.0))
    initial_position_norm = np.linalg.norm(navigation_filter.state.position_ned_m)
    initial_position_variance = float(np.trace(navigation_filter.covariance[:3, :3]))
    navigation_filter.predict([0.0, 0.0, 0.1], [0.0, 0.0, -9.80665], 1.0)
    _roll, _pitch, yaw = quaternion_to_euler321(navigation_filter.state.quaternion_nb)
    assert yaw == pytest.approx(0.1, abs=1.0e-12)

    navigation_filter.update_gnss(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),
    )
    navigation_filter.update_barometric_altitude(0.0, 0.25)

    assert np.linalg.norm(navigation_filter.state.position_ned_m) < initial_position_norm
    assert np.trace(navigation_filter.covariance[:3, :3]) < initial_position_variance
    assert navigation_filter.last_innovation is not None
    assert np.all(navigation_filter.standard_deviation() >= 0.0)
