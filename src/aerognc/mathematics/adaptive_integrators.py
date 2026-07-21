"""Direct Dormand--Prince 5(4) integration with dense scalar events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.integrators import (
    DerivativeFunction,
    EventOccurrence,
    EventSpec,
    StateProjection,
)
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class AdaptiveOptions:
    """Error-control and step-size settings, all expressed in SI seconds."""

    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-9
    initial_step_s: float = 1.0e-3
    minimum_step_s: float = 1.0e-10
    maximum_step_s: float = 1.0
    event_time_tolerance_s: float = 1.0e-10
    max_attempted_steps: int = 1_000_000

    def __post_init__(self) -> None:
        positive_values = {
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "initial_step_s": self.initial_step_s,
            "minimum_step_s": self.minimum_step_s,
            "maximum_step_s": self.maximum_step_s,
            "event_time_tolerance_s": self.event_time_tolerance_s,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.minimum_step_s > self.initial_step_s:
            raise ValueError("minimum_step_s cannot exceed initial_step_s")
        if self.initial_step_s > self.maximum_step_s:
            raise ValueError("initial_step_s cannot exceed maximum_step_s")
        if self.max_attempted_steps <= 0:
            raise ValueError("max_attempted_steps must be positive")


@dataclass(frozen=True, slots=True)
class AdaptiveStatistics:
    """Operational statistics for one propagation."""

    accepted_steps: int
    rejected_steps: int
    derivative_evaluations: int
    minimum_accepted_step_s: float
    maximum_accepted_step_s: float
    final_step_s: float
    recommended_next_step_s: float


@dataclass(frozen=True, slots=True)
class AdaptiveIntegrationResult:
    """Accepted states, located events, and solver statistics."""

    time_s: FloatArray
    state: FloatArray
    events: tuple[EventOccurrence, ...]
    statistics: AdaptiveStatistics


def _validate_state(value: npt.ArrayLike, *, name: str) -> FloatArray:
    state = np.asarray(value, dtype=np.float64)
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return state.copy()


def _derivative(
    function: DerivativeFunction,
    time_s: float,
    state: FloatArray,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    value = np.asarray(function(time_s, state.copy()), dtype=np.float64)
    if value.shape != expected_shape:
        raise ValueError(f"derivative shape {value.shape} does not match state {expected_shape}")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("derivative contains non-finite values")
    return value


def _dormand_prince_step(
    function: DerivativeFunction,
    time_s: float,
    state: FloatArray,
    step_s: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, int]:
    """Return fifth-order state, embedded error, and endpoint derivatives."""
    shape = state.shape
    k1 = _derivative(function, time_s, state, shape)
    k2 = _derivative(function, time_s + step_s / 5.0, state + step_s * k1 / 5.0, shape)
    k3 = _derivative(
        function,
        time_s + 3.0 * step_s / 10.0,
        state + step_s * (3.0 * k1 / 40.0 + 9.0 * k2 / 40.0),
        shape,
    )
    k4 = _derivative(
        function,
        time_s + 4.0 * step_s / 5.0,
        state + step_s * (44.0 * k1 / 45.0 - 56.0 * k2 / 15.0 + 32.0 * k3 / 9.0),
        shape,
    )
    k5 = _derivative(
        function,
        time_s + 8.0 * step_s / 9.0,
        state
        + step_s
        * (
            19372.0 * k1 / 6561.0
            - 25360.0 * k2 / 2187.0
            + 64448.0 * k3 / 6561.0
            - 212.0 * k4 / 729.0
        ),
        shape,
    )
    k6 = _derivative(
        function,
        time_s + step_s,
        state
        + step_s
        * (
            9017.0 * k1 / 3168.0
            - 355.0 * k2 / 33.0
            + 46732.0 * k3 / 5247.0
            + 49.0 * k4 / 176.0
            - 5103.0 * k5 / 18656.0
        ),
        shape,
    )
    fifth = state + step_s * (
        35.0 * k1 / 384.0
        + 500.0 * k3 / 1113.0
        + 125.0 * k4 / 192.0
        - 2187.0 * k5 / 6784.0
        + 11.0 * k6 / 84.0
    )
    k7 = _derivative(function, time_s + step_s, fifth, shape)
    fourth = state + step_s * (
        5179.0 * k1 / 57600.0
        + 7571.0 * k3 / 16695.0
        + 393.0 * k4 / 640.0
        - 92097.0 * k5 / 339200.0
        + 187.0 * k6 / 2100.0
        + k7 / 40.0
    )
    return fifth, fifth - fourth, k1, k7, 7


def _crossed(value_before: float, value_after: float, direction: int) -> bool:
    if not np.isfinite(value_before) or not np.isfinite(value_after):
        raise FloatingPointError("event function returned a non-finite value")
    increasing = value_before < 0.0 <= value_after
    decreasing = value_before > 0.0 >= value_after
    if value_before == 0.0 and value_after != 0.0:
        increasing = value_after > 0.0
        decreasing = value_after < 0.0
    return (direction >= 0 and increasing) or (direction <= 0 and decreasing)


def _dense_hermite(
    fraction: float,
    state_before: FloatArray,
    state_after: FloatArray,
    derivative_before: FloatArray,
    derivative_after: FloatArray,
    step_s: float,
) -> FloatArray:
    s = fraction
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * state_before
        + (s3 - 2.0 * s2 + s) * step_s * derivative_before
        + (-2.0 * s3 + 3.0 * s2) * state_after
        + (s3 - s2) * step_s * derivative_after
    )


def _project_state(state: FloatArray, projection: StateProjection | None) -> FloatArray:
    if projection is None:
        projected = state
    else:
        projected = np.asarray(projection(state.copy()), dtype=np.float64)
    if projected.shape != state.shape or not np.all(np.isfinite(projected)):
        raise FloatingPointError("state projection returned an invalid state")
    return projected.copy()


def _locate_event(
    event: EventSpec,
    time_before_s: float,
    state_before: FloatArray,
    value_before: float,
    time_after_s: float,
    state_after: FloatArray,
    value_after: float,
    derivative_before: FloatArray,
    derivative_after: FloatArray,
    time_tolerance_s: float,
    projection: StateProjection | None,
) -> EventOccurrence:
    if value_before == 0.0:
        return EventOccurrence(event.name, time_before_s, state_before.copy())
    if value_after == 0.0:
        return EventOccurrence(event.name, time_after_s, state_after.copy())

    lower = 0.0
    upper = 1.0
    lower_value = value_before
    step_s = time_after_s - time_before_s
    while (upper - lower) * step_s > time_tolerance_s:
        middle = 0.5 * (lower + upper)
        middle_time_s = time_before_s + middle * step_s
        middle_state = _project_state(
            _dense_hermite(
                middle,
                state_before,
                state_after,
                derivative_before,
                derivative_after,
                step_s,
            ),
            projection,
        )
        middle_value = float(event.function(middle_time_s, middle_state))
        if not np.isfinite(middle_value):
            raise FloatingPointError("event function returned a non-finite value")
        if middle_value == 0.0:
            lower = upper = middle
            break
        if np.signbit(middle_value) == np.signbit(lower_value):
            lower = middle
            lower_value = middle_value
        else:
            upper = middle

    fraction = 0.5 * (lower + upper)
    event_time_s = time_before_s + fraction * step_s
    event_state = _project_state(
        _dense_hermite(
            fraction,
            state_before,
            state_after,
            derivative_before,
            derivative_after,
            step_s,
        ),
        projection,
    )
    return EventOccurrence(event.name, float(event_time_s), event_state)


def integrate_adaptive(
    derivative: DerivativeFunction,
    initial_state: npt.ArrayLike,
    time_span_s: tuple[float, float],
    *,
    options: AdaptiveOptions | None = None,
    events: Sequence[EventSpec] = (),
    state_projection: StateProjection | None = None,
) -> AdaptiveIntegrationResult:
    """Integrate using an embedded Dormand--Prince 5(4) pair.

    Accepted-step endpoints are returned. Scalar roots are bracketed only across
    accepted steps and refined on a cubic Hermite interpolant by safeguarded
    bisection. Each named event is recorded once, matching the one-shot flight-event
    contract used by the fixed-step solver.
    """
    start_s, end_s = time_span_s
    if not np.isfinite([start_s, end_s]).all() or end_s <= start_s:
        raise ValueError("time_span_s must be finite and strictly increasing")
    if len({event.name for event in events}) != len(events):
        raise ValueError("event names must be unique")
    if options is None:
        options = AdaptiveOptions()

    state = _project_state(_validate_state(initial_state, name="initial_state"), state_projection)
    time_s = float(start_s)
    step_s = min(options.initial_step_s, options.maximum_step_s, end_s - start_s)
    time_values = [time_s]
    state_values = [state.copy()]
    occurrences: list[EventOccurrence] = []
    active = {event.name: event for event in events}
    previous_values = {name: float(spec.function(time_s, state)) for name, spec in active.items()}
    for value in previous_values.values():
        if not np.isfinite(value):
            raise FloatingPointError("event function returned a non-finite value")

    accepted = 0
    rejected = 0
    evaluations = 0
    accepted_steps: list[float] = []
    final_step_s = 0.0
    recommended_next_step_s = step_s
    terminal_reached = False

    while time_s < end_s and not terminal_reached:
        if accepted + rejected >= options.max_attempted_steps:
            raise RuntimeError("maximum attempted adaptive steps exceeded")
        remaining_s = end_s - time_s
        actual_step_s = min(step_s, remaining_s)
        fifth, error, derivative_before, derivative_after, calls = _dormand_prince_step(
            derivative, time_s, state, actual_step_s
        )
        evaluations += calls
        scale = options.absolute_tolerance + options.relative_tolerance * np.maximum(
            np.abs(state), np.abs(fifth)
        )
        error_norm = float(np.sqrt(np.mean(np.square(error / scale))))
        if not np.isfinite(error_norm):
            raise FloatingPointError("adaptive error norm is non-finite")

        factor = 5.0 if error_norm == 0.0 else float(np.clip(0.9 * error_norm ** (-0.2), 0.2, 5.0))
        candidate_next_step_s = float(
            np.clip(actual_step_s * factor, options.minimum_step_s, options.maximum_step_s)
        )

        if error_norm > 1.0:
            rejected += 1
            if actual_step_s <= options.minimum_step_s * (1.0 + 1.0e-12):
                raise RuntimeError("adaptive tolerance cannot be met at minimum_step_s")
            step_s = min(actual_step_s * min(factor, 0.9), options.maximum_step_s)
            step_s = max(step_s, options.minimum_step_s)
            continue

        next_time_s = time_s + actual_step_s
        if end_s - next_time_s <= 1.0e-13 * max(1.0, abs(end_s)):
            next_time_s = float(end_s)
        next_state = _project_state(fifth, state_projection)
        accepted += 1
        accepted_steps.append(actual_step_s)
        final_step_s = actual_step_s
        recommended_next_step_s = candidate_next_step_s

        crossings: list[tuple[EventSpec, EventOccurrence]] = []
        for name, event in tuple(active.items()):
            value_before = previous_values[name]
            value_after = float(event.function(next_time_s, next_state))
            if _crossed(value_before, value_after, event.direction):
                occurrence = _locate_event(
                    event,
                    time_s,
                    state,
                    value_before,
                    next_time_s,
                    next_state,
                    value_after,
                    derivative_before,
                    derivative_after,
                    options.event_time_tolerance_s,
                    state_projection,
                )
                crossings.append((event, occurrence))
                del active[name]
                del previous_values[name]
            else:
                previous_values[name] = value_after

        terminal_crossings = [item for item in crossings if item[0].terminal]
        if terminal_crossings:
            _terminal, terminal_occurrence = min(
                terminal_crossings, key=lambda item: item[1].time_s
            )
            occurrences.extend(
                occurrence
                for _event, occurrence in crossings
                if occurrence.time_s <= terminal_occurrence.time_s
            )
            time_values.append(terminal_occurrence.time_s)
            state_values.append(terminal_occurrence.state.copy())
            terminal_reached = True
        else:
            occurrences.extend(occurrence for _event, occurrence in crossings)
            time_s = next_time_s
            state = next_state
            time_values.append(time_s)
            state_values.append(state.copy())
            step_s = candidate_next_step_s

    occurrences.sort(key=lambda occurrence: occurrence.time_s)
    minimum = min(accepted_steps) if accepted_steps else 0.0
    maximum = max(accepted_steps) if accepted_steps else 0.0
    return AdaptiveIntegrationResult(
        time_s=np.asarray(time_values, dtype=np.float64),
        state=np.vstack(state_values),
        events=tuple(occurrences),
        statistics=AdaptiveStatistics(
            accepted_steps=accepted,
            rejected_steps=rejected,
            derivative_evaluations=evaluations,
            minimum_accepted_step_s=float(minimum),
            maximum_accepted_step_s=float(maximum),
            final_step_s=float(final_step_s),
            recommended_next_step_s=float(recommended_next_step_s),
        ),
    )
