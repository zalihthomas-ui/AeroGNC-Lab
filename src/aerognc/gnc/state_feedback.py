"""Manual SISO pole placement and bounded state feedback."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


def ackermann_gain(
    system_matrix: npt.ArrayLike,
    input_matrix: npt.ArrayLike,
    desired_poles: Sequence[complex],
) -> FloatArray:
    """Compute SISO feedback gain using Ackermann's formula.

    This implementation deliberately avoids specialised control toolboxes. It is
    intended for small, well-conditioned educational models; production designs
    should prefer numerically robust Schur-based algorithms.
    """
    system = np.asarray(system_matrix, dtype=np.float64)
    input_vector = np.asarray(input_matrix, dtype=np.float64)
    if system.ndim != 2 or system.shape[0] != system.shape[1]:
        raise ValueError("system_matrix must be square")
    order = system.shape[0]
    if input_vector.shape == (order,):
        input_vector = input_vector[:, None]
    if input_vector.shape != (order, 1):
        raise ValueError("input_matrix must describe one input")
    if len(desired_poles) != order:
        raise ValueError("desired pole count must equal system order")
    if not np.all(np.isfinite(system)) or not np.all(np.isfinite(input_vector)):
        raise ValueError("state-space matrices must be finite")

    controllability = np.hstack(
        [np.linalg.matrix_power(system, exponent) @ input_vector for exponent in range(order)]
    )
    if np.linalg.matrix_rank(controllability) != order:
        raise ValueError("system is not controllable")
    coefficients = np.poly(np.asarray(desired_poles, dtype=np.complex128))
    phi = np.linalg.matrix_power(system.astype(np.complex128), order)
    for index in range(1, order + 1):
        phi += coefficients[index] * np.linalg.matrix_power(system, order - index)
    selector = np.zeros(order)
    selector[-1] = 1.0
    left_factor = np.linalg.solve(controllability.T, selector)
    gain_complex = left_factor @ phi
    if np.max(np.abs(gain_complex.imag)) > 1.0e-10:
        raise ValueError("desired poles do not produce a real feedback gain")
    return np.asarray(gain_complex.real, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class StateFeedbackController:
    """Bounded full-state regulator about a supplied reference state."""

    gain: FloatArray
    output_limit: float

    def __init__(self, gain: npt.ArrayLike, output_limit: float) -> None:
        gain_array = np.asarray(gain, dtype=np.float64)
        if gain_array.ndim != 1 or gain_array.size == 0 or not np.all(np.isfinite(gain_array)):
            raise ValueError("gain must be a finite one-dimensional array")
        if not np.isfinite(output_limit) or output_limit <= 0.0:
            raise ValueError("output_limit must be positive and finite")
        object.__setattr__(self, "gain", gain_array.copy())
        object.__setattr__(self, "output_limit", float(output_limit))

    def command(self, state: npt.ArrayLike, reference_state: npt.ArrayLike) -> float:
        """Return ``-K(x-x_ref)`` with symmetric output saturation."""
        state_array = np.asarray(state, dtype=np.float64)
        reference = np.asarray(reference_state, dtype=np.float64)
        if state_array.shape != self.gain.shape or reference.shape != self.gain.shape:
            raise ValueError("state and reference_state must match gain shape")
        command = -float(self.gain @ (state_array - reference))
        return float(np.clip(command, -self.output_limit, self.output_limit))


@dataclass(frozen=True, slots=True)
class GainSchedule:
    """Linearly interpolated state-feedback gains versus one scheduling variable."""

    scheduling_points: FloatArray
    gains: FloatArray

    def __init__(self, scheduling_points: npt.ArrayLike, gains: npt.ArrayLike) -> None:
        points = np.asarray(scheduling_points, dtype=np.float64)
        gain_table = np.asarray(gains, dtype=np.float64)
        if points.ndim != 1 or points.size < 2 or not np.all(np.diff(points) > 0.0):
            raise ValueError("scheduling_points must be strictly increasing")
        if gain_table.ndim != 2 or gain_table.shape[0] != points.size:
            raise ValueError("gains must have one row per scheduling point")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(gain_table)):
            raise ValueError("gain schedule must be finite")
        object.__setattr__(self, "scheduling_points", points.copy())
        object.__setattr__(self, "gains", gain_table.copy())

    def gain_at(self, scheduling_value: float) -> FloatArray:
        """Return clamped component-wise linear interpolation."""
        if not np.isfinite(scheduling_value):
            raise ValueError("scheduling_value must be finite")
        return np.array(
            [
                np.interp(
                    scheduling_value,
                    self.scheduling_points,
                    self.gains[:, component],
                )
                for component in range(self.gains.shape[1])
            ]
        )
