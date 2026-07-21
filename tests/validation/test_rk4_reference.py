import numpy as np
from scipy.integrate import solve_ivp

from aerognc.mathematics.integrators import integrate_fixed_step


def test_rk4_fourth_order_convergence() -> None:
    exact = np.e
    errors = []
    for step_s in (0.2, 0.1, 0.05):
        result = integrate_fixed_step(lambda _t, state: state, [1.0], (0.0, 1.0), step_s)
        errors.append(abs(result.state[-1, 0] - exact))
    assert errors[0] / errors[1] > 12.0
    assert errors[1] / errors[2] > 12.0


def test_rk4_against_independent_scipy_reference() -> None:
    def derivative(_time_s: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -0.4 * state[1] - np.sin(state[0])])

    custom = integrate_fixed_step(derivative, [0.7, -0.2], (0.0, 4.0), 0.002)
    reference = solve_ivp(
        derivative,
        (0.0, 4.0),
        [0.7, -0.2],
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
        t_eval=[4.0],
    )
    np.testing.assert_allclose(custom.state[-1], reference.y[:, -1], rtol=0.0, atol=1.0e-9)
