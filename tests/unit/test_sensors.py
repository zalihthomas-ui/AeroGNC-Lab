import numpy as np
import pytest

from aerognc.vehicle.sensors import (
    BarometricAltimeter,
    SensorErrorParameters,
)


def _parameters(**overrides: object) -> SensorErrorParameters:
    values: dict[str, object] = {
        "sample_rate_hz": 10.0,
        "noise_std": [0.5],
        "constant_bias": [1.0],
        "bias_drift_std_per_sqrt_s": [0.02],
        "quantisation": [0.1],
        "delay_s": 0.2,
        "dropout_probability": 0.0,
    }
    values.update(overrides)
    return SensorErrorParameters(**values)  # type: ignore[arg-type]


def test_sensor_sequence_is_seeded_sampled_quantised_and_delayed() -> None:
    first = BarometricAltimeter(_parameters(), seed=42)
    second = BarometricAltimeter(_parameters(), seed=42)
    first_values = []
    second_values = []
    for time_s in np.arange(0.0, 1.01, 0.05):
        measurement_a = first.measure(float(time_s), [100.0])
        measurement_b = second.measure(float(time_s), [100.0])
        if measurement_a is not None:
            first_values.append(measurement_a)
        if measurement_b is not None:
            second_values.append(measurement_b)
    assert len(first_values) == len(second_values) > 0
    for left, right in zip(first_values, second_values, strict=True):
        np.testing.assert_array_equal(left.value, right.value)
        assert left.available_time_s - left.sample_time_s == pytest.approx(0.2)
        assert left.value[0] * 10.0 == pytest.approx(round(left.value[0] * 10.0))


def test_scheduled_dropout_and_reset() -> None:
    sensor = BarometricAltimeter(
        _parameters(delay_s=0.0, dropout_intervals_s=((0.2, 0.5),)), seed=7
    )
    times = np.arange(0.0, 0.71, 0.1)
    measurements = [sensor.measure(float(time_s), [0.0]) for time_s in times]
    sample_times = [item.sample_time_s for item in measurements if item is not None]
    assert not any(0.2 <= time_s < 0.5 for time_s in sample_times)
    sensor.reset()
    reset_first = sensor.measure(0.0, [0.0])
    comparison = BarometricAltimeter(
        _parameters(delay_s=0.0, dropout_intervals_s=((0.2, 0.5),)), seed=7
    ).measure(0.0, [0.0])
    assert reset_first is not None and comparison is not None
    np.testing.assert_array_equal(reset_first.value, comparison.value)
