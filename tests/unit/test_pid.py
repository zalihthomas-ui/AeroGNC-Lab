import pytest

from aerognc.gnc.pid import PIDController, PIDGains


def test_pid_output_limit_and_anti_windup() -> None:
    controller = PIDController(
        PIDGains(2.0, 4.0, 0.0, output_min=-1.0, output_max=1.0, anti_windup_gain=1.0)
    )
    outputs = [controller.update(10.0, 0.01) for _ in range(1_000)]
    assert max(outputs) == 1.0
    assert abs(controller.integral_state) < 1.0e-12
    assert controller.saturated
    assert controller.update(-0.1, 0.01) < 0.0


def test_pid_derivative_filter_and_reset() -> None:
    controller = PIDController(PIDGains(0.0, 0.0, 1.0, derivative_filter_tau_s=0.1))
    assert controller.update(0.0, 0.01) == 0.0
    assert 0.0 < controller.update(1.0, 0.01) < 100.0
    controller.reset()
    assert controller.previous_error is None
    assert controller.integral_state == 0.0
    with pytest.raises(ValueError):
        controller.update(0.0, 0.0)


def test_pid_tracks_output_for_bumpless_transfer() -> None:
    controller = PIDController(
        PIDGains(
            proportional=2.0,
            integral=0.5,
            derivative=1.0,
            output_min=-10.0,
            output_max=10.0,
            integral_min=-20.0,
            integral_max=20.0,
        )
    )

    assert controller.track_output(error=3.0, output=4.0) == pytest.approx(4.0)
    assert controller.previous_error == pytest.approx(3.0)
    assert controller.filtered_derivative == 0.0
    assert controller.update(3.0, 0.1) == pytest.approx(4.15)

    assert controller.track_output(error=0.0, output=99.0) == pytest.approx(10.0)
    assert controller.saturated
    with pytest.raises(ValueError, match="finite"):
        controller.track_output(float("nan"), 0.0)
