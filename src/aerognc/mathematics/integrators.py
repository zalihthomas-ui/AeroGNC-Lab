"""Custom fixed-step fourth-order Runge-Kutta integration with scalar events."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

DerivativeFunction = Callable[[float, FloatArray], npt.ArrayLike]
EventFunction = Callable[[float, FloatArray], float]
StateProjection = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Scalar zero-crossing definition.

    ``direction`` is +1 for increasing crossings, -1 for decreasing crossings,
    or 0 for either direction. The integrator records the first crossing per event.
    """

    name: str
    function: EventFunction
    direction: int = 0
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("event name cannot be empty")
        if self.direction not in {-1, 0, 1}:
            raise ValueError("event direction must be -1, 0, or 1")


@dataclass(frozen=True, slots=True)
class EventOccurrence:
    """Located scalar-event crossing."""

    name: str
    time_s: float
    state: FloatArray


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """Fixed-step trajectory and event records."""

    time_s: FloatArray
    state: FloatArray
    events: tuple[EventOccurrence, ...]


def _validated_derivative(value: npt.ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    derivative = np.asarray(value, dtype=np.float64)
    if derivative.shape != shape:
        raise ValueError(f"derivative shape {derivative.shape} does not match state {shape}")
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError("derivative contains non-finite values")
    return derivative


def rk4_step(
    derivative: DerivativeFunction,
    time_s: float,
    state: npt.ArrayLike,
    step_s: float,
) -> FloatArray:
    """Advance one classical explicit RK4 step."""
    if not np.isfinite(time_s):
        raise ValueError("time_s must be finite")
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("step_s must be positive and finite")
    y = np.asarray(state, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("state must be a finite one-dimensional array")
    k1 = _validated_derivative(derivative(time_s, y.copy()), y.shape)
    k2 = _validated_derivative(derivative(time_s + 0.5 * step_s, y + 0.5 * step_s * k1), y.shape)
    k3 = _validated_derivative(derivative(time_s + 0.5 * step_s, y + 0.5 * step_s * k2), y.shape)
    k4 = _validated_derivative(derivative(time_s + step_s, y + step_s * k3), y.shape)
    return y + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _crossed(value_before: float, value_after: float, direction: int) -> bool:
    if not np.isfinite(value_before) or not np.isfinite(value_after):
        raise FloatingPointError("event function returned a non-finite value")
    increasing = value_before < 0.0 <= value_after
    decreasing = value_before > 0.0 >= value_after
    exact_departure = value_before == 0.0 and value_after != 0.0
    if exact_departure:
        increasing = value_after > 0.0
        decreasing = value_after < 0.0
    return (direction >= 0 and increasing) or (direction <= 0 and decreasing)


def _locate_linear(
    event: EventSpec,
    time_before_s: float,
    state_before: FloatArray,
    value_before: float,
    time_after_s: float,
    state_after: FloatArray,
    value_after: float,
) -> EventOccurrence:
    denominator = abs(value_before) + abs(value_after)
    fraction = 0.5 if denominator == 0.0 else abs(value_before) / denominator
    time_s = time_before_s + fraction * (time_after_s - time_before_s)
    state = state_before + fraction * (state_after - state_before)
    return EventOccurrence(event.name, float(time_s), state)


def integrate_fixed_step(
    derivative: DerivativeFunction,
    initial_state: npt.ArrayLike,
    time_span_s: tuple[float, float],
    step_s: float,
    *,
    events: Sequence[EventSpec] = (),
    state_projection: StateProjection | None = None,
) -> IntegrationResult:
    """Integrate across a time span with optional directed scalar events.

    The final step is shortened to end exactly at ``time_span_s[1]``. Event state
    and time are bracketed using linear interpolation. Each named event is recorded
    once, which is appropriate for the discrete flight events used by this project.
    """
    start_s, end_s = time_span_s
    if not np.isfinite([start_s, end_s]).all() or end_s <= start_s:
        raise ValueError("time_span_s must be finite and strictly increasing")
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("step_s must be positive and finite")
    state = np.asarray(initial_state, dtype=np.float64)
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be a finite one-dimensional array")
    state = state.copy()
    if state_projection is not None:
        state = np.asarray(state_projection(state), dtype=np.float64)
    if len({event.name for event in events}) != len(events):
        raise ValueError("event names must be unique")

    time_values = [float(start_s)]
    state_values = [state.copy()]
    occurrences: list[EventOccurrence] = []
    active = {event.name: event for event in events}
    previous_values = {name: spec.function(start_s, state) for name, spec in active.items()}
    time_s = float(start_s)

    while time_s < end_s:
        actual_step_s = min(step_s, end_s - time_s)
        next_state = rk4_step(derivative, time_s, state, actual_step_s)
        if state_projection is not None:
            next_state = np.asarray(state_projection(next_state), dtype=np.float64)
        if next_state.shape != state.shape or not np.all(np.isfinite(next_state)):
            raise FloatingPointError("state projection returned an invalid state")
        next_time_s = time_s + actual_step_s
        if end_s - next_time_s <= 1.0e-12 * max(1.0, abs(end_s)):
            next_time_s = float(end_s)

        crossings: list[tuple[EventSpec, EventOccurrence]] = []
        for name, event in tuple(active.items()):
            value_after = float(event.function(next_time_s, next_state))
            value_before = float(previous_values[name])
            if _crossed(value_before, value_after, event.direction):
                occurrence = _locate_linear(
                    event,
                    time_s,
                    state,
                    value_before,
                    next_time_s,
                    next_state,
                    value_after,
                )
                crossings.append((event, occurrence))
                del active[name]
                del previous_values[name]
            else:
                previous_values[name] = value_after

        terminal_crossings = [item for item in crossings if item[0].terminal]
        if terminal_crossings:
            _event, occurrence = min(terminal_crossings, key=lambda item: item[1].time_s)
            for _candidate, candidate_occurrence in crossings:
                if candidate_occurrence.time_s <= occurrence.time_s:
                    occurrences.append(candidate_occurrence)
            time_values.append(occurrence.time_s)
            state_values.append(occurrence.state.copy())
            break

        occurrences.extend(occurrence for _, occurrence in crossings)
        time_s = next_time_s
        state = next_state
        time_values.append(time_s)
        state_values.append(state.copy())

    occurrences.sort(key=lambda occurrence: occurrence.time_s)
    return IntegrationResult(
        time_s=np.asarray(time_values, dtype=np.float64),
        state=np.vstack(state_values),
        events=tuple(occurrences),
    )
