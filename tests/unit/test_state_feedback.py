import numpy as np
import pytest

from aerognc.gnc.state_feedback import GainSchedule, StateFeedbackController, ackermann_gain


def test_ackermann_places_requested_poles() -> None:
    system = np.array([[0.0, 1.0], [0.0, 0.0]])
    input_matrix = np.array([[0.0], [0.5]])
    requested = np.array([-2.0, -3.5])
    gain = ackermann_gain(system, input_matrix, requested)
    achieved = np.linalg.eigvals(system - input_matrix @ gain[None, :])
    np.testing.assert_allclose(np.sort(achieved), np.sort(requested), atol=1.0e-12)


def test_uncontrollable_system_is_rejected() -> None:
    with pytest.raises(ValueError, match="not controllable"):
        ackermann_gain(np.eye(2), np.zeros((2, 1)), [-1.0, -2.0])


def test_state_feedback_limit_and_gain_schedule() -> None:
    controller = StateFeedbackController([2.0, 1.0], 0.5)
    assert controller.command([1.0, 0.0], [0.0, 0.0]) == -0.5
    schedule = GainSchedule([0.0, 10.0], [[1.0, 2.0], [3.0, 6.0]])
    np.testing.assert_allclose(schedule.gain_at(5.0), [2.0, 4.0])
