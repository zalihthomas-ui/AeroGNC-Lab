"""Trim, linearisation, modal, LQR, margin, identification, and SIL utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

NonlinearDynamics = Callable[[FloatArray, FloatArray], FloatArray]


def finite_difference_jacobian(
    function: Callable[[FloatArray], FloatArray],
    point: npt.ArrayLike,
    *,
    relative_step: float = 1.0e-6,
) -> FloatArray:
    """Return a central finite-difference Jacobian with scaled perturbations."""
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.ndim != 1 or not np.all(np.isfinite(point_array)):
        raise ValueError("Jacobian point must be a finite one-dimensional array")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be positive and finite")
    baseline = np.asarray(function(point_array), dtype=np.float64)
    if baseline.ndim != 1 or not np.all(np.isfinite(baseline)):
        raise ValueError("Jacobian function must return a finite one-dimensional array")
    jacobian = np.empty((baseline.size, point_array.size), dtype=np.float64)
    for column in range(point_array.size):
        step = relative_step * max(1.0, abs(point_array[column]))
        positive = point_array.copy()
        negative = point_array.copy()
        positive[column] += step
        negative[column] -= step
        jacobian[:, column] = (
            np.asarray(function(positive), dtype=np.float64)
            - np.asarray(function(negative), dtype=np.float64)
        ) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("finite-difference Jacobian contains non-finite values")
    return jacobian


@dataclass(frozen=True, slots=True)
class TrimResult:
    """Nonlinear least-squares trim solution and convergence record."""

    decision: FloatArray
    residual: FloatArray
    iterations: int
    converged: bool


def solve_trim(
    residual_function: Callable[[FloatArray], FloatArray],
    initial_decision: npt.ArrayLike,
    *,
    lower_bounds: npt.ArrayLike | None = None,
    upper_bounds: npt.ArrayLike | None = None,
    tolerance: float = 1.0e-9,
    maximum_iterations: int = 40,
) -> TrimResult:
    """Solve measurable steady-flight residuals with damped Gauss-Newton steps."""
    decision = np.asarray(initial_decision, dtype=np.float64)
    if decision.ndim != 1 or not np.all(np.isfinite(decision)):
        raise ValueError("initial trim decision must be a finite vector")
    lower = (
        np.full(decision.shape, -np.inf)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=np.float64)
    )
    upper = (
        np.full(decision.shape, np.inf)
        if upper_bounds is None
        else np.asarray(upper_bounds, dtype=np.float64)
    )
    if lower.shape != decision.shape or upper.shape != decision.shape or np.any(lower > upper):
        raise ValueError("trim bounds must match the decision and satisfy lower <= upper")
    if tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("trim tolerance and maximum_iterations must be positive")
    decision = np.clip(decision, lower, upper)
    residual = np.asarray(residual_function(decision), dtype=np.float64)
    if residual.ndim != 1 or not np.all(np.isfinite(residual)):
        raise ValueError("trim residual function must return a finite vector")
    converged = False
    iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        if float(np.linalg.norm(residual, ord=np.inf)) <= tolerance:
            converged = True
            break
        jacobian = finite_difference_jacobian(residual_function, decision)
        damping = 1.0e-8 * np.eye(decision.size)
        step = -np.linalg.solve(
            jacobian.T @ jacobian + damping,
            jacobian.T @ residual,
        )
        accepted = False
        current_norm = float(np.linalg.norm(residual))
        scale = 1.0
        for _line_search in range(16):
            candidate = np.clip(decision + scale * step, lower, upper)
            candidate_residual = np.asarray(residual_function(candidate), dtype=np.float64)
            if (
                np.all(np.isfinite(candidate_residual))
                and np.linalg.norm(candidate_residual) < current_norm
            ):
                decision = candidate
                residual = candidate_residual
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    if float(np.linalg.norm(residual, ord=np.inf)) <= tolerance:
        converged = True
    return TrimResult(decision.copy(), residual.copy(), iterations, converged)


@dataclass(frozen=True, slots=True)
class LinearModel:
    """Continuous perturbation model about a state/input operating point."""

    system_matrix: FloatArray
    input_matrix: FloatArray
    trim_state: FloatArray
    trim_input: FloatArray
    derivative_at_trim: FloatArray


def linearize_dynamics(
    dynamics: NonlinearDynamics,
    trim_state: npt.ArrayLike,
    trim_input: npt.ArrayLike,
    *,
    relative_step: float = 1.0e-6,
) -> LinearModel:
    """Linearise ``x_dot=f(x,u)`` using independently perturbed central differences."""
    state = np.asarray(trim_state, dtype=np.float64)
    control = np.asarray(trim_input, dtype=np.float64)
    if state.ndim != 1 or control.ndim != 1:
        raise ValueError("trim state and input must be one-dimensional")
    derivative = np.asarray(dynamics(state, control), dtype=np.float64)
    if derivative.shape != state.shape or not np.all(np.isfinite(derivative)):
        raise ValueError("dynamics must return one finite derivative per state")
    system = finite_difference_jacobian(
        lambda candidate: dynamics(candidate, control), state, relative_step=relative_step
    )
    input_matrix = finite_difference_jacobian(
        lambda candidate: dynamics(state, candidate), control, relative_step=relative_step
    )
    return LinearModel(system, input_matrix, state.copy(), control.copy(), derivative)


def controllability_matrix(system_matrix: npt.ArrayLike, input_matrix: npt.ArrayLike) -> FloatArray:
    """Return ``[B, AB, ..., A^(n-1)B]``."""
    system = np.asarray(system_matrix, dtype=np.float64)
    inputs = np.asarray(input_matrix, dtype=np.float64)
    if system.ndim != 2 or system.shape[0] != system.shape[1]:
        raise ValueError("system matrix must be square")
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    if inputs.ndim != 2 or inputs.shape[0] != system.shape[0]:
        raise ValueError("input matrix row count must equal system order")
    return np.hstack(
        [np.linalg.matrix_power(system, exponent) @ inputs for exponent in range(system.shape[0])]
    )


def observability_matrix(system_matrix: npt.ArrayLike, output_matrix: npt.ArrayLike) -> FloatArray:
    """Return the stacked continuous observability matrix."""
    system = np.asarray(system_matrix, dtype=np.float64)
    outputs = np.asarray(output_matrix, dtype=np.float64)
    if system.ndim != 2 or system.shape[0] != system.shape[1]:
        raise ValueError("system matrix must be square")
    if outputs.ndim == 1:
        outputs = outputs[None, :]
    if outputs.ndim != 2 or outputs.shape[1] != system.shape[0]:
        raise ValueError("output matrix column count must equal system order")
    return np.vstack(
        [outputs @ np.linalg.matrix_power(system, exponent) for exponent in range(system.shape[0])]
    )


@dataclass(frozen=True, slots=True)
class DynamicMode:
    """One continuous eigenmode with conventional damping quantities."""

    eigenvalue: complex
    natural_frequency_radps: float
    damping_ratio: float
    time_constant_s: float
    stable: bool


def analyze_modes(system_matrix: npt.ArrayLike) -> tuple[DynamicMode, ...]:
    """Extract eigenvalues, natural frequencies, damping and stability flags."""
    system = np.asarray(system_matrix, dtype=np.float64)
    if system.ndim != 2 or system.shape[0] != system.shape[1]:
        raise ValueError("system matrix must be square")
    modes: list[DynamicMode] = []
    for eigenvalue in np.linalg.eigvals(system):
        frequency = float(abs(eigenvalue))
        damping = float(-eigenvalue.real / frequency) if frequency > 0.0 else np.inf
        time_constant = float(-1.0 / eigenvalue.real) if eigenvalue.real < 0.0 else np.inf
        modes.append(
            DynamicMode(
                complex(eigenvalue),
                frequency,
                damping,
                time_constant,
                bool(eigenvalue.real < 0.0),
            )
        )
    return tuple(modes)


@dataclass(frozen=True, slots=True)
class LQRDesign:
    """Continuous optimal feedback solution computed from a Hamiltonian eigenspace."""

    gain: FloatArray
    riccati_solution: FloatArray
    closed_loop_eigenvalues: npt.NDArray[np.complex128]
    riccati_residual_norm: float


def continuous_lqr(
    system_matrix: npt.ArrayLike,
    input_matrix: npt.ArrayLike,
    state_weight: npt.ArrayLike,
    input_weight: npt.ArrayLike,
) -> LQRDesign:
    """Solve the continuous algebraic Riccati equation without a control toolbox."""
    system = np.asarray(system_matrix, dtype=np.float64)
    inputs = np.asarray(input_matrix, dtype=np.float64)
    q_weight = np.asarray(state_weight, dtype=np.float64)
    r_weight = np.asarray(input_weight, dtype=np.float64)
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    order = system.shape[0] if system.ndim == 2 else 0
    if system.shape != (order, order) or q_weight.shape != (order, order):
        raise ValueError("A and Q must be square with the same order")
    if (
        inputs.ndim != 2
        or inputs.shape[0] != order
        or r_weight.shape
        != (
            inputs.shape[1],
            inputs.shape[1],
        )
    ):
        raise ValueError("B and R dimensions are inconsistent with A")
    if not all(np.all(np.isfinite(matrix)) for matrix in (system, inputs, q_weight, r_weight)):
        raise ValueError("LQR matrices must be finite")
    if not np.allclose(q_weight, q_weight.T) or not np.allclose(r_weight, r_weight.T):
        raise ValueError("LQR weights must be symmetric")
    if np.any(np.linalg.eigvalsh(q_weight) < -1.0e-12) or np.any(
        np.linalg.eigvalsh(r_weight) <= 0.0
    ):
        raise ValueError("Q must be positive semidefinite and R positive definite")
    inverse_r = np.linalg.inv(r_weight)
    hamiltonian = np.block([[system, -inputs @ inverse_r @ inputs.T], [-q_weight, -system.T]])
    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    stable_indices = np.flatnonzero(eigenvalues.real < 0.0)
    if stable_indices.size != order:
        raise ValueError("Hamiltonian does not have the required stable invariant subspace")
    stable_vectors = eigenvectors[:, stable_indices]
    upper = stable_vectors[:order, :]
    lower = stable_vectors[order:, :]
    riccati_complex = lower @ np.linalg.inv(upper)
    if np.max(np.abs(riccati_complex.imag)) > 1.0e-7:
        raise FloatingPointError("LQR Riccati solution retained a significant imaginary part")
    riccati = np.asarray(riccati_complex.real, dtype=np.float64)
    riccati = 0.5 * (riccati + riccati.T)
    gain = np.linalg.solve(r_weight, inputs.T @ riccati)
    residual = (
        system.T @ riccati
        + riccati @ system
        - riccati @ inputs @ inverse_r @ inputs.T @ riccati
        + q_weight
    )
    closed_loop_eigenvalues = np.asarray(
        np.linalg.eigvals(system - inputs @ gain), dtype=np.complex128
    )
    return LQRDesign(gain, riccati, closed_loop_eigenvalues, float(np.linalg.norm(residual)))


def build_lqr_gain_schedule(
    scheduling_points: Sequence[float],
    linear_models: Sequence[LinearModel],
    state_weight: npt.ArrayLike,
    input_weight: npt.ArrayLike,
) -> tuple[FloatArray, tuple[LQRDesign, ...]]:
    """Design one LQR row per operating point for later linear interpolation."""
    points = np.asarray(scheduling_points, dtype=np.float64)
    if points.ndim != 1 or points.size != len(linear_models) or np.any(np.diff(points) <= 0.0):
        raise ValueError("gain-schedule points must be increasing and match model count")
    designs = tuple(
        continuous_lqr(model.system_matrix, model.input_matrix, state_weight, input_weight)
        for model in linear_models
    )
    if any(design.gain.shape[0] != 1 for design in designs):
        raise ValueError("gain-schedule helper currently expects one control input")
    return np.vstack([design.gain[0] for design in designs]), designs


def frequency_response(
    system_matrix: npt.ArrayLike,
    input_matrix: npt.ArrayLike,
    output_matrix: npt.ArrayLike,
    feedthrough_matrix: npt.ArrayLike,
    angular_frequency_radps: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Evaluate ``C(jwI-A)^-1B+D`` for every requested frequency."""
    system = np.asarray(system_matrix, dtype=np.float64)
    inputs = np.asarray(input_matrix, dtype=np.float64)
    outputs = np.asarray(output_matrix, dtype=np.float64)
    feedthrough = np.asarray(feedthrough_matrix, dtype=np.float64)
    frequency = np.asarray(angular_frequency_radps, dtype=np.float64)
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    if outputs.ndim == 1:
        outputs = outputs[None, :]
    if frequency.ndim != 1 or np.any(frequency <= 0.0) or not np.all(np.isfinite(frequency)):
        raise ValueError("frequencies must be a positive finite vector")
    if system.ndim != 2 or system.shape[0] != system.shape[1]:
        raise ValueError("system matrix must be square")
    if inputs.shape[0] != system.shape[0] or outputs.shape[1] != system.shape[0]:
        raise ValueError("state-space matrix dimensions are inconsistent")
    if feedthrough.shape != (outputs.shape[0], inputs.shape[1]):
        raise ValueError("feedthrough shape must be outputs by inputs")
    identity = np.eye(system.shape[0], dtype=np.complex128)
    response = np.empty((frequency.size, outputs.shape[0], inputs.shape[1]), dtype=np.complex128)
    for index, omega in enumerate(frequency):
        response[index] = (
            outputs @ np.linalg.solve(1j * omega * identity - system, inputs) + feedthrough
        )
    return response


@dataclass(frozen=True, slots=True)
class StabilityMargins:
    """SISO gain/phase margins and their crossover frequencies."""

    gain_margin: float
    phase_margin_deg: float
    gain_crossover_radps: float
    phase_crossover_radps: float


def _first_crossing(x_values: FloatArray, y_values: FloatArray, level: float) -> float | None:
    shifted = y_values - level
    indices = np.flatnonzero(shifted[:-1] * shifted[1:] <= 0.0)
    if indices.size == 0:
        return None
    index = int(indices[0])
    if shifted[index + 1] == shifted[index]:
        return float(x_values[index])
    fraction = -shifted[index] / (shifted[index + 1] - shifted[index])
    return float(x_values[index] + fraction * (x_values[index + 1] - x_values[index]))


def stability_margins_siso(
    angular_frequency_radps: npt.ArrayLike, response: npt.ArrayLike
) -> StabilityMargins:
    """Interpolate classical margins from a densely sampled complex open loop."""
    frequency = np.asarray(angular_frequency_radps, dtype=np.float64)
    values = np.asarray(response, dtype=np.complex128)
    if frequency.ndim != 1 or values.shape != frequency.shape or frequency.size < 3:
        raise ValueError("margin inputs must be matching one-dimensional arrays")
    if np.any(frequency <= 0.0) or not np.all(np.diff(frequency) > 0.0):
        raise ValueError("margin frequencies must be positive and strictly increasing")
    log_frequency = np.log(frequency)
    magnitude = np.abs(values)
    phase_deg = np.unwrap(np.angle(values)) * 180.0 / np.pi
    gain_log = _first_crossing(log_frequency, magnitude, 1.0)
    phase_log = _first_crossing(log_frequency, phase_deg, -180.0)
    if gain_log is None:
        gain_crossover = np.inf
        phase_margin = np.inf
    else:
        gain_crossover = float(np.exp(gain_log))
        phase_at_gain = float(np.interp(gain_log, log_frequency, phase_deg))
        phase_margin = 180.0 + phase_at_gain
    if phase_log is None:
        phase_crossover = np.inf
        gain_margin = np.inf
    else:
        phase_crossover = float(np.exp(phase_log))
        magnitude_at_phase = float(np.interp(phase_log, log_frequency, magnitude))
        gain_margin = np.inf if magnitude_at_phase == 0.0 else 1.0 / magnitude_at_phase
    return StabilityMargins(gain_margin, phase_margin, gain_crossover, phase_crossover)


@dataclass(frozen=True, slots=True)
class SystemIdentificationResult:
    """Least-squares continuous state model and fit residual."""

    system_matrix: FloatArray
    input_matrix: FloatArray
    derivative_rms_error: FloatArray
    condition_number: float


def identify_linear_state_space(
    time_s: npt.ArrayLike,
    state_history: npt.ArrayLike,
    input_history: npt.ArrayLike,
) -> SystemIdentificationResult:
    """Fit ``x_dot=A x+B u`` from uniformly or nonuniformly sampled flight data."""
    time = np.asarray(time_s, dtype=np.float64)
    states = np.asarray(state_history, dtype=np.float64)
    controls = np.asarray(input_history, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    if time.ndim != 1 or states.ndim != 2 or controls.ndim != 2:
        raise ValueError("identification time/state/input arrays have invalid dimensions")
    if time.size < 5 or states.shape[0] != time.size or controls.shape[0] != time.size:
        raise ValueError("identification histories must share at least five samples")
    if np.any(np.diff(time) <= 0.0) or not all(
        np.all(np.isfinite(array)) for array in (time, states, controls)
    ):
        raise ValueError("identification data must be finite with increasing time")
    derivatives = np.gradient(states, time, axis=0, edge_order=2)
    regressors = np.hstack((states, controls))
    coefficients, _residuals, _rank, singular_values = np.linalg.lstsq(
        regressors, derivatives, rcond=None
    )
    prediction = regressors @ coefficients
    state_count = states.shape[1]
    system = coefficients[:state_count].T
    inputs = coefficients[state_count:].T
    rms = np.sqrt(np.mean((derivatives - prediction) ** 2, axis=0))
    condition = (
        np.inf if singular_values[-1] == 0.0 else float(singular_values[0] / singular_values[-1])
    )
    return SystemIdentificationResult(system, inputs, rms, condition)


@dataclass(frozen=True, slots=True)
class SILTimingResult:
    """Controller software-in-the-loop latency statistics."""

    sample_count: int
    mean_execution_s: float
    p95_execution_s: float
    maximum_execution_s: float
    missed_deadline_count: int
    output_checksum: float


def benchmark_controller_sil(
    controller: Callable[[FloatArray], npt.ArrayLike | float],
    input_samples: npt.ArrayLike,
    *,
    deadline_s: float,
    repeat_count: int = 1,
) -> SILTimingResult:
    """Measure deterministic controller-call timing without claiming hardware HIL."""
    samples = np.asarray(input_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("SIL inputs must be a nonempty finite two-dimensional array")
    if not np.isfinite(deadline_s) or deadline_s <= 0.0 or repeat_count <= 0:
        raise ValueError("SIL deadline and repeat_count must be positive")
    durations: list[float] = []
    checksum = 0.0
    for _repeat in range(repeat_count):
        for sample in samples:
            start_ns = perf_counter_ns()
            output = np.asarray(controller(sample), dtype=np.float64)
            durations.append((perf_counter_ns() - start_ns) * 1.0e-9)
            if not np.all(np.isfinite(output)):
                raise FloatingPointError("SIL controller produced a non-finite output")
            checksum += float(np.sum(output))
    duration_array = np.asarray(durations)
    return SILTimingResult(
        duration_array.size,
        float(np.mean(duration_array)),
        float(np.percentile(duration_array, 95.0)),
        float(np.max(duration_array)),
        int(np.count_nonzero(duration_array > deadline_s)),
        checksum,
    )
