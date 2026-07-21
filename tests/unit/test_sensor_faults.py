import numpy as np
import pytest

from aerognc.vehicle.sensor_faults import SensorFaultEvent, SensorFaultInjector


def test_bias_spike_stuck_and_dropout_fault_semantics() -> None:
    injector = SensorFaultInjector(
        (
            SensorFaultEvent("gnss", "bias_step", 1.0, 2.0, [2.0, -1.0]),
            SensorFaultEvent("gnss", "spike", 1.2, 1.8, [10.0, 20.0]),
            SensorFaultEvent("barometer", "stuck", 2.0, 3.0),
            SensorFaultEvent("barometer", "dropout", 4.0, 5.0),
        )
    )
    np.testing.assert_allclose(injector.apply("gnss", 1.25, [1.0, 1.0]), [13.0, 20.0])
    np.testing.assert_allclose(injector.apply("gnss", 1.5, [1.0, 1.0]), [3.0, 0.0])
    np.testing.assert_allclose(injector.apply("barometer", 2.1, [100.0]), [100.0])
    np.testing.assert_allclose(injector.apply("barometer", 2.5, [140.0]), [100.0])
    assert injector.apply("barometer", 4.5, [150.0]) is None
    np.testing.assert_allclose(injector.apply("barometer", 5.1, [160.0]), [160.0])


def test_fault_injector_reset_rearms_one_shot_spike() -> None:
    injector = SensorFaultInjector((SensorFaultEvent("gnss", "spike", 0.0, 1.0, [5.0]),))
    np.testing.assert_allclose(injector.apply("gnss", 0.1, [0.0]), [5.0])
    np.testing.assert_allclose(injector.apply("gnss", 0.2, [0.0]), [0.0])
    injector.reset()
    np.testing.assert_allclose(injector.apply("gnss", 0.3, [0.0]), [5.0])


def test_fault_dimension_mismatch_and_invalid_interval_fail_clearly() -> None:
    injector = SensorFaultInjector((SensorFaultEvent("gnss", "bias_step", 0.0, 1.0, [1.0, 2.0]),))
    with pytest.raises(ValueError, match="expected"):
        injector.apply("gnss", 0.5, [1.0])
    with pytest.raises(ValueError, match="increasing"):
        SensorFaultEvent("gnss", "dropout", 2.0, 1.0)
