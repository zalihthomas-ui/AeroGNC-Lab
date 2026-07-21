import numpy as np
import pytest
from scipy.integrate import solve_ivp

from aerognc.mathematics.adaptive_integrators import AdaptiveOptions, integrate_adaptive
from aerognc.mathematics.integrators import EventSpec


def test_dormand_prince_matches_exponential_and_scipy() -> None:
    options = AdaptiveOptions(
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
        initial_step_s=0.4,
        minimum_step_s=1.0e-12,
        maximum_step_s=0.5,
    )
    result = integrate_adaptive(lambda _t, y: y, [1.0], (0.0, 2.0), options=options)
    reference = solve_ivp(
        lambda _t, y: y,
        (0.0, 2.0),
        np.array([1.0]),
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    assert result.state[-1, 0] == pytest.approx(np.exp(2.0), rel=2.0e-10)
    assert result.state[-1, 0] == pytest.approx(reference.y[0, -1], rel=2.0e-10)
    assert result.statistics.accepted_steps > 1
    assert result.statistics.rejected_steps >= 1
    assert result.statistics.derivative_evaluations == 7 * (
        result.statistics.accepted_steps + result.statistics.rejected_steps
    )


def test_dense_event_bisection_honors_direction_and_terminal() -> None:
    event = EventSpec(
        "position-root",
        lambda _t, state: float(state[0] - 0.5),
        direction=1,
        terminal=True,
    )
    ignored = EventSpec(
        "wrong-direction",
        lambda _t, state: float(state[0] - 0.25),
        direction=-1,
    )
    result = integrate_adaptive(
        lambda _t, state: np.array([2.0 * state[1], 1.0]),
        [0.0, 0.0],
        (0.0, 2.0),
        options=AdaptiveOptions(
            initial_step_s=0.8,
            minimum_step_s=1.0e-12,
            maximum_step_s=1.0,
            event_time_tolerance_s=1.0e-11,
        ),
        events=(event, ignored),
    )
    expected_time = np.sqrt(0.5)

    assert [item.name for item in result.events] == ["position-root"]
    assert result.events[0].time_s == pytest.approx(expected_time, abs=1.0e-9)
    assert result.time_s[-1] == pytest.approx(expected_time, abs=1.0e-9)
    assert result.state[-1, 0] == pytest.approx(0.5, abs=1.0e-9)


def test_state_projection_preserves_quaternion_norm() -> None:
    def projection(state: np.ndarray) -> np.ndarray:
        projected = state.copy()
        projected /= np.linalg.norm(projected)
        return projected

    result = integrate_adaptive(
        lambda _t, q: 0.5 * np.array([-q[1], q[0], q[3], -q[2]]),
        [1.0, 0.0, 0.0, 0.0],
        (0.0, 20.0),
        options=AdaptiveOptions(initial_step_s=0.1, maximum_step_s=0.5),
        state_projection=projection,
    )
    np.testing.assert_allclose(np.linalg.norm(result.state, axis=1), 1.0, atol=2.0e-15)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("relative_tolerance", 0.0),
        ("absolute_tolerance", np.nan),
        ("initial_step_s", -1.0),
        ("event_time_tolerance_s", 0.0),
    ],
)
def test_adaptive_options_reject_invalid_values(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        AdaptiveOptions(**{keyword: value})


def test_adaptive_integrator_rejects_duplicate_events_and_bad_derivative() -> None:
    same = EventSpec("same", lambda _t, _state: 1.0)
    with pytest.raises(ValueError, match="unique"):
        integrate_adaptive(lambda _t, _state: [1.0], [0.0], (0.0, 1.0), events=(same, same))
    with pytest.raises(ValueError, match="shape"):
        integrate_adaptive(lambda _t, _state: [1.0, 2.0], [0.0], (0.0, 1.0))
