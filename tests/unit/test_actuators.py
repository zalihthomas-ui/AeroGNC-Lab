import numpy as np
import pytest

from aerognc.vehicle.actuators import ActuatorAllocator, ActuatorLimits, FirstOrderActuator


def test_actuator_delay_rate_and_position_limits() -> None:
    actuator = FirstOrderActuator(ActuatorLimits(0.1, 0.2, 0.5, command_delay_s=0.2))
    first = actuator.update(1.0, 0.1)
    second = actuator.update(1.0, 0.1)
    assert first == 0.0
    assert second == 0.0
    positions = [actuator.update(1.0, 0.1) for _ in range(20)]
    assert np.max(np.diff([second, *positions])) <= 0.05 + 1.0e-12
    assert positions[-1] == pytest.approx(0.2)
    assert actuator.saturated


def test_allocator_preserves_sign_and_limits() -> None:
    allocator = ActuatorAllocator([10.0, -20.0, 5.0], [0.1, 0.2, 0.3])
    commands = allocator.allocate([2.0, 2.0, -4.0])
    np.testing.assert_allclose(commands, [0.1, -0.1, -0.3])
    np.testing.assert_allclose(allocator.achieved_moment_body_nm(commands), [1.0, 2.0, -1.5])
