import numpy as np
import pytest

from aerognc.mathematics.integrators import EventSpec, integrate_fixed_step, rk4_step


def test_rk4_constant_derivative() -> None:
    result = rk4_step(lambda _t, _x: np.array([2.0, -1.0]), 0.0, [3.0, 4.0], 0.5)
    np.testing.assert_allclose(result, [4.0, 3.5], atol=1.0e-15)


def test_directed_terminal_event_is_located_and_truncates() -> None:
    event = EventSpec(
        "threshold", lambda _t, state: float(state[0] - 0.35), direction=1, terminal=True
    )
    result = integrate_fixed_step(
        lambda _t, _x: np.array([1.0]), [0.0], (0.0, 2.0), 0.1, events=[event]
    )
    assert len(result.events) == 1
    assert result.events[0].time_s == pytest.approx(0.35, abs=1.0e-12)
    assert result.time_s[-1] == pytest.approx(0.35, abs=1.0e-12)
    assert result.state[-1, 0] == pytest.approx(0.35, abs=1.0e-12)


def test_direction_filter_ignores_wrong_direction() -> None:
    event = EventSpec("falling-only", lambda _t, state: float(state[0] - 0.2), direction=-1)
    result = integrate_fixed_step(
        lambda _t, _x: np.array([1.0]), [0.0], (0.0, 0.5), 0.1, events=[event]
    )
    assert result.events == ()


def test_integrator_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        rk4_step(lambda _t, _x: [1.0], 0.0, [0.0], 0.0)
    with pytest.raises(ValueError, match="unique"):
        integrate_fixed_step(
            lambda _t, _x: np.array([1.0]),
            [0.0],
            (0.0, 1.0),
            0.1,
            events=[EventSpec("same", lambda _t, _x: 1.0), EventSpec("same", lambda _t, _x: -1.0)],
        )
