import numpy as np

from aerognc.gnc.ekf import VerticalFilterTuning, VerticalNavigationEKF


def test_vertical_filter_reduces_altitude_error_and_keeps_covariance_symmetric() -> None:
    generator = np.random.default_rng(1204)
    step_s = 0.02
    time_s = np.arange(0.0, 40.0 + step_s, step_s)
    true_acceleration = 0.4 * np.sin(0.3 * time_s)
    true_velocity = np.zeros_like(time_s)
    true_altitude = np.zeros_like(time_s)
    for index in range(1, time_s.size):
        true_velocity[index] = true_velocity[index - 1] + true_acceleration[index - 1] * step_s
        true_altitude[index] = (
            true_altitude[index - 1]
            + true_velocity[index - 1] * step_s
            + 0.5 * true_acceleration[index - 1] * step_s**2
        )
    acceleration_measurement = true_acceleration + 0.12 + generator.normal(0.0, 0.08, time_s.size)
    raw_baro = true_altitude + generator.normal(0.0, 3.0, time_s.size)
    navigation = VerticalNavigationEKF(
        [0.0, 0.0, 0.0],
        np.diag([25.0, 4.0, 0.25]),
        VerticalFilterTuning(0.15, 0.005, 3.0, 1.5, 0.3),
    )
    estimated_altitude = np.empty_like(time_s)
    for index in range(time_s.size):
        if index > 0:
            navigation.predict(float(acceleration_measurement[index - 1]), step_s)
        if index % 10 == 0:
            navigation.update_barometer(float(raw_baro[index]))
        if index % 50 == 0:
            navigation.update_gnss(
                float(true_altitude[index] + generator.normal(0.0, 1.5)),
                float(true_velocity[index] + generator.normal(0.0, 0.3)),
            )
        estimated_altitude[index] = navigation.state[0]
        np.testing.assert_allclose(navigation.covariance, navigation.covariance.T, atol=1e-12)
        assert np.linalg.eigvalsh(navigation.covariance).min() >= -1e-12
    raw_rms = float(np.sqrt(np.mean((raw_baro - true_altitude) ** 2)))
    estimated_rms = float(np.sqrt(np.mean((estimated_altitude - true_altitude) ** 2)))
    assert estimated_rms < 0.5 * raw_rms
    assert abs(navigation.state[2] - 0.12) < 0.05
