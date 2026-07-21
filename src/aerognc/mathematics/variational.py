"""Finite-difference variational dynamics and sensitivity propagation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.adaptive_integrators import (
    AdaptiveIntegrationResult,
    AdaptiveOptions,
    AdaptiveStatistics,
    integrate_adaptive,
)
from aerognc.mathematics.vectors import FloatArray

VectorFunction = Callable[[FloatArray], npt.ArrayLike]
ParameterizedDerivative = Callable[[float, FloatArray, FloatArray], npt.ArrayLike]


def _vector(value: npt.ArrayLike, *, name: str, allow_empty: bool = False) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or (array.size == 0 and not allow_empty) or not np.all(np.isfinite(array)):
        qualifier = "finite one-dimensional" if allow_empty else "non-empty finite one-dimensional"
        raise ValueError(f"{name} must be a {qualifier} array")
    return array.copy()


def _perturbations(
    point: FloatArray,
    supplied: npt.ArrayLike | None,
    *,
    name: str,
    relative_step: float,
) -> FloatArray:
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be positive and finite")
    if supplied is None:
        return relative_step * np.maximum(1.0, np.abs(point))
    values = np.asarray(supplied, dtype=np.float64)
    if values.shape != point.shape or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must match its vector and contain positive finite values")
    return values.copy()


def central_difference_jacobian(
    function: VectorFunction,
    point: npt.ArrayLike,
    *,
    perturbations: npt.ArrayLike | None = None,
    relative_step: float = 1.0e-6,
) -> FloatArray:
    """Calculate an output-by-input Jacobian using central differences."""
    location = _vector(point, name="point")
    steps = _perturbations(
        location, perturbations, name="perturbations", relative_step=relative_step
    )
    baseline = np.asarray(function(location.copy()), dtype=np.float64)
    if baseline.ndim != 1 or baseline.size == 0 or not np.all(np.isfinite(baseline)):
        raise ValueError("function output must be a non-empty finite vector")
    jacobian = np.empty((baseline.size, location.size), dtype=np.float64)
    for column, step in enumerate(steps):
        positive = location.copy()
        negative = location.copy()
        positive[column] += step
        negative[column] -= step
        upper = np.asarray(function(positive), dtype=np.float64)
        lower = np.asarray(function(negative), dtype=np.float64)
        if upper.shape != baseline.shape or lower.shape != baseline.shape:
            raise ValueError("function output shape changed during finite differencing")
        if not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower)):
            raise FloatingPointError("function returned non-finite finite-difference values")
        jacobian[:, column] = (upper - lower) / (2.0 * step)
    return jacobian


def dynamics_jacobians(
    derivative: ParameterizedDerivative,
    time_s: float,
    state: npt.ArrayLike,
    parameters: npt.ArrayLike,
    *,
    state_perturbations: npt.ArrayLike | None = None,
    parameter_perturbations: npt.ArrayLike | None = None,
    relative_step: float = 1.0e-6,
) -> tuple[FloatArray, FloatArray]:
    """Return state and selected-parameter Jacobians of one dynamics model."""
    if not np.isfinite(time_s):
        raise ValueError("time_s must be finite")
    state_vector = _vector(state, name="state")
    parameter_vector = _vector(parameters, name="parameters", allow_empty=True)
    state_steps = _perturbations(
        state_vector,
        state_perturbations,
        name="state_perturbations",
        relative_step=relative_step,
    )
    state_jacobian = central_difference_jacobian(
        lambda candidate: derivative(time_s, candidate, parameter_vector),
        state_vector,
        perturbations=state_steps,
        relative_step=relative_step,
    )
    if state_jacobian.shape[0] != state_vector.size:
        raise ValueError("dynamics derivative output size must equal state size")
    if parameter_vector.size == 0:
        return state_jacobian, np.empty((state_vector.size, 0), dtype=np.float64)
    parameter_steps = _perturbations(
        parameter_vector,
        parameter_perturbations,
        name="parameter_perturbations",
        relative_step=relative_step,
    )
    parameter_jacobian = central_difference_jacobian(
        lambda candidate: derivative(time_s, state_vector, candidate),
        parameter_vector,
        perturbations=parameter_steps,
        relative_step=relative_step,
    )
    if parameter_jacobian.shape[0] != state_vector.size:
        raise ValueError("dynamics derivative output size must equal state size")
    return state_jacobian, parameter_jacobian


@dataclass(frozen=True, slots=True)
class VariationalResult:
    """Nominal state, state-transition matrix, and parameter sensitivities."""

    time_s: FloatArray
    state: FloatArray
    state_transition: FloatArray
    parameter_sensitivity: FloatArray
    statistics: AdaptiveStatistics


def propagate_variational(
    derivative: ParameterizedDerivative,
    initial_state: npt.ArrayLike,
    parameters: npt.ArrayLike,
    time_span_s: tuple[float, float],
    *,
    integration_options: AdaptiveOptions | None = None,
    state_perturbations: npt.ArrayLike | None = None,
    parameter_perturbations: npt.ArrayLike | None = None,
    relative_difference_step: float = 1.0e-6,
) -> VariationalResult:
    r"""Integrate nominal and first-order variational equations.

    With :math:`\dot{x}=f(t,x,p)`, this propagates
    :math:`\dot{\Phi}=A\Phi` and :math:`\dot{S}=AS+B`, where
    :math:`A=\partial f/\partial x`, :math:`B=\partial f/\partial p`,
    :math:`\Phi(t_0)=I`, and :math:`S(t_0)=0`.
    """
    state = _vector(initial_state, name="initial_state")
    parameter_vector = _vector(parameters, name="parameters", allow_empty=True)
    state_steps = _perturbations(
        state,
        state_perturbations,
        name="state_perturbations",
        relative_step=relative_difference_step,
    )
    parameter_steps = _perturbations(
        parameter_vector,
        parameter_perturbations,
        name="parameter_perturbations",
        relative_step=relative_difference_step,
    )
    state_size = state.size
    parameter_size = parameter_vector.size
    initial_transition = np.eye(state_size, dtype=np.float64)
    initial_sensitivity = np.zeros((state_size, parameter_size), dtype=np.float64)
    augmented_initial = np.concatenate(
        (state, initial_transition.ravel(), initial_sensitivity.ravel())
    )

    def augmented_derivative(time_s: float, augmented: FloatArray) -> FloatArray:
        nominal = augmented[:state_size]
        transition_start = state_size
        sensitivity_start = transition_start + state_size * state_size
        transition = augmented[transition_start:sensitivity_start].reshape(state_size, state_size)
        sensitivity = augmented[sensitivity_start:].reshape(state_size, parameter_size)
        nominal_derivative = np.asarray(
            derivative(time_s, nominal.copy(), parameter_vector.copy()), dtype=np.float64
        )
        if nominal_derivative.shape != nominal.shape or not np.all(np.isfinite(nominal_derivative)):
            raise ValueError("dynamics derivative must match the finite nominal-state shape")
        state_jacobian, parameter_jacobian = dynamics_jacobians(
            derivative,
            time_s,
            nominal,
            parameter_vector,
            state_perturbations=state_steps,
            parameter_perturbations=parameter_steps,
            relative_step=relative_difference_step,
        )
        transition_derivative = state_jacobian @ transition
        sensitivity_derivative = state_jacobian @ sensitivity + parameter_jacobian
        return np.concatenate(
            (
                nominal_derivative,
                transition_derivative.ravel(),
                sensitivity_derivative.ravel(),
            )
        )

    integrated: AdaptiveIntegrationResult = integrate_adaptive(
        augmented_derivative,
        augmented_initial,
        time_span_s,
        options=integration_options,
    )
    transition_start = state_size
    sensitivity_start = transition_start + state_size * state_size
    return VariationalResult(
        time_s=integrated.time_s,
        state=integrated.state[:, :state_size],
        state_transition=integrated.state[:, transition_start:sensitivity_start].reshape(
            -1, state_size, state_size
        ),
        parameter_sensitivity=integrated.state[:, sensitivity_start:].reshape(
            -1, state_size, parameter_size
        ),
        statistics=integrated.statistics,
    )
