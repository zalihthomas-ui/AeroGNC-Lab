"""Direct PID implementation with filtering, limiting, and anti-windup."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PIDGains:
    """PID tuning and implementation limits."""

    proportional: float
    integral: float
    derivative: float
    output_min: float = -np.inf
    output_max: float = np.inf
    integral_min: float = -np.inf
    integral_max: float = np.inf
    derivative_filter_tau_s: float = 0.0
    anti_windup_gain: float = 1.0

    def __post_init__(self) -> None:
        finite_required = np.array(
            [
                self.proportional,
                self.integral,
                self.derivative,
                self.derivative_filter_tau_s,
                self.anti_windup_gain,
            ]
        )
        if not np.all(np.isfinite(finite_required)):
            raise ValueError("PID gains and time constants must be finite")
        if self.output_min >= self.output_max:
            raise ValueError("output_min must be below output_max")
        if self.integral_min >= self.integral_max:
            raise ValueError("integral_min must be below integral_max")
        if self.derivative_filter_tau_s < 0.0 or self.anti_windup_gain < 0.0:
            raise ValueError("filter time and anti-windup gain must be nonnegative")


class PIDController:
    """Sampled PID controller with conditional integration and back-calculation."""

    def __init__(self, gains: PIDGains) -> None:
        self.gains = gains
        self.integral_state = 0.0
        self.previous_error: float | None = None
        self.filtered_derivative = 0.0
        self.saturated = False

    def reset(self) -> None:
        """Clear integrator, derivative, and saturation state."""
        self.integral_state = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0
        self.saturated = False

    def update(self, error: float, step_s: float) -> float:
        """Advance one sample and return the bounded controller output."""
        if not np.isfinite(error):
            raise ValueError("error must be finite")
        if not np.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("step_s must be positive and finite")
        raw_derivative = (
            0.0 if self.previous_error is None else (error - self.previous_error) / step_s
        )
        tau_s = self.gains.derivative_filter_tau_s
        if tau_s == 0.0:
            self.filtered_derivative = raw_derivative
        else:
            alpha = tau_s / (tau_s + step_s)
            self.filtered_derivative = (
                alpha * self.filtered_derivative + (1.0 - alpha) * raw_derivative
            )

        candidate_integral = float(
            np.clip(
                self.integral_state + error * step_s,
                self.gains.integral_min,
                self.gains.integral_max,
            )
        )
        unconstrained = self._unconstrained(error, candidate_integral)
        output = float(np.clip(unconstrained, self.gains.output_min, self.gains.output_max))
        drives_further_high = unconstrained > self.gains.output_max and error > 0.0
        drives_further_low = unconstrained < self.gains.output_min and error < 0.0
        if drives_further_high or drives_further_low:
            candidate_integral = self.integral_state
            unconstrained = self._unconstrained(error, candidate_integral)
            output = float(np.clip(unconstrained, self.gains.output_min, self.gains.output_max))
        elif self.gains.integral != 0.0 and output != unconstrained:
            correction = (
                self.gains.anti_windup_gain
                * (output - unconstrained)
                / self.gains.integral
                * step_s
            )
            candidate_integral = float(
                np.clip(
                    candidate_integral + correction,
                    self.gains.integral_min,
                    self.gains.integral_max,
                )
            )
        self.integral_state = candidate_integral
        self.previous_error = error
        self.saturated = not np.isclose(output, unconstrained)
        return output

    def _unconstrained(self, error: float, integral_state: float) -> float:
        return (
            self.gains.proportional * error
            + self.gains.integral * integral_state
            + self.gains.derivative * self.filtered_derivative
        )
