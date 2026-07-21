import numpy as np
import pytest
from scipy.linalg import expm

from aerognc.mathematics.adaptive_integrators import AdaptiveOptions
from aerognc.mathematics.variational import (
    central_difference_jacobian,
    dynamics_jacobians,
    propagate_variational,
)


def test_central_difference_jacobian_for_nonlinear_vector_function() -> None:
    point = np.array([0.4, -0.7])
    jacobian = central_difference_jacobian(
        lambda value: np.array([value[0] ** 2 + value[1], np.sin(value[0] * value[1])]),
        point,
    )
    expected = np.array(
        [
            [2.0 * point[0], 1.0],
            [point[1] * np.cos(point[0] * point[1]), point[0] * np.cos(point[0] * point[1])],
        ]
    )
    np.testing.assert_allclose(jacobian, expected, rtol=2.0e-9, atol=2.0e-10)


def test_variational_solution_matches_linear_matrix_exponential() -> None:
    system = np.array([[0.0, 1.0], [-4.0, -0.2]])
    input_matrix = np.array([[0.0], [1.0]])

    def derivative(_time_s: float, state: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return system @ state + input_matrix @ parameters

    duration_s = 1.7
    initial = np.array([0.2, -0.1])
    parameters = np.array([0.3])
    result = propagate_variational(
        derivative,
        initial,
        parameters,
        (0.0, duration_s),
        integration_options=AdaptiveOptions(
            relative_tolerance=2.0e-10,
            absolute_tolerance=1.0e-12,
            initial_step_s=0.05,
            minimum_step_s=1.0e-12,
            maximum_step_s=0.15,
        ),
    )
    transition = expm(system * duration_s)
    augmented = np.block([[system, input_matrix], [np.zeros((1, 2)), np.zeros((1, 1))]])
    sensitivity = expm(augmented * duration_s)[:2, 2:]
    expected_state = transition @ initial + sensitivity @ parameters

    np.testing.assert_allclose(result.state_transition[-1], transition, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(
        result.parameter_sensitivity[-1], sensitivity, rtol=2.0e-9, atol=2.0e-10
    )
    np.testing.assert_allclose(result.state[-1], expected_state, rtol=2.0e-9, atol=2.0e-10)


def test_dynamics_jacobians_support_no_selected_parameters() -> None:
    state_jacobian, parameter_jacobian = dynamics_jacobians(
        lambda _time_s, state, _parameters: np.array([state[1], -state[0]]),
        0.0,
        [1.0, 2.0],
        [],
    )
    np.testing.assert_allclose(state_jacobian, [[0.0, 1.0], [-1.0, 0.0]], atol=1.0e-10)
    assert parameter_jacobian.shape == (2, 0)


def test_variational_inputs_fail_clearly() -> None:
    with pytest.raises(ValueError, match="positive"):
        central_difference_jacobian(lambda value: value, [1.0], perturbations=[0.0])
    with pytest.raises(ValueError, match="output size"):
        dynamics_jacobians(
            lambda _time_s, _state, _parameters: np.array([1.0, 2.0]),
            0.0,
            [1.0],
            [],
        )
