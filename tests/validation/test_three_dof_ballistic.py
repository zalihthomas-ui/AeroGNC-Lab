import numpy as np

from aerognc.dynamics.three_dof import point_mass_derivative
from aerognc.mathematics.integrators import integrate_fixed_step


def test_vacuum_ballistic_motion_matches_analytical_solution() -> None:
    gravity = np.array([0.0, 0.0, 9.80665])
    initial = np.array([0.0, 0.0, 0.0, 15.0, -2.0, -100.0])

    def derivative(_time_s: float, state: np.ndarray) -> np.ndarray:
        return point_mass_derivative(
            state,
            mass_kg=5.0,
            applied_force_ned_n=np.zeros(3),
            gravity_ned_mps2=gravity,
        )

    result = integrate_fixed_step(derivative, initial, (0.0, 7.0), 0.1)
    expected_position = initial[:3] + initial[3:] * 7.0 + 0.5 * gravity * 7.0**2
    expected_velocity = initial[3:] + gravity * 7.0
    np.testing.assert_allclose(result.state[-1, :3], expected_position, atol=1.0e-11)
    np.testing.assert_allclose(result.state[-1, 3:], expected_velocity, atol=1.0e-12)
