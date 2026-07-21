import numpy as np
from scipy.integrate import solve_ivp

from aerognc.mathematics.adaptive_integrators import AdaptiveOptions, integrate_adaptive


def test_adaptive_harmonic_oscillator_matches_independent_scipy_solution() -> None:
    def oscillator(_time_s: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -2.25 * state[0]])

    options = AdaptiveOptions(
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-11,
        initial_step_s=0.3,
        minimum_step_s=1.0e-12,
        maximum_step_s=0.4,
    )
    custom = integrate_adaptive(oscillator, [1.0, 0.2], (0.0, 12.0), options=options)
    reference = solve_ivp(
        oscillator,
        (0.0, 12.0),
        np.array([1.0, 0.2]),
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(custom.state[-1], reference.y[:, -1], rtol=2.0e-8, atol=2.0e-9)
