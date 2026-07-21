"""Bounded first-order actuator dynamics and fictional torque allocation."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class ActuatorLimits:
    """Actuator dynamic and command bounds in SI/radian units."""

    time_constant_s: float
    position_limit_rad: float
    rate_limit_radps: float
    command_delay_s: float = 0.0

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.time_constant_s,
                self.position_limit_rad,
                self.rate_limit_radps,
                self.command_delay_s,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("actuator limits must be finite")
        if self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive")
        if self.position_limit_rad <= 0.0 or self.rate_limit_radps <= 0.0:
            raise ValueError("position and rate limits must be positive")
        if self.command_delay_s < 0.0:
            raise ValueError("command_delay_s must be nonnegative")


class FirstOrderActuator:
    """Delayed first-order actuator with rate and position saturation."""

    def __init__(self, limits: ActuatorLimits, initial_position_rad: float = 0.0) -> None:
        if not np.isfinite(initial_position_rad):
            raise ValueError("initial_position_rad must be finite")
        self.limits = limits
        self.position_rad = float(
            np.clip(initial_position_rad, -limits.position_limit_rad, limits.position_limit_rad)
        )
        self.rate_radps = 0.0
        self.elapsed_time_s = 0.0
        self.saturated = False
        self._history: deque[tuple[float, float]] = deque([(0.0, self.position_rad)])
        self._delayed_command_rad = self.position_rad

    def reset(self, position_rad: float = 0.0) -> None:
        """Reset state and delay history."""
        if not np.isfinite(position_rad):
            raise ValueError("position_rad must be finite")
        self.position_rad = float(
            np.clip(position_rad, -self.limits.position_limit_rad, self.limits.position_limit_rad)
        )
        self.rate_radps = 0.0
        self.elapsed_time_s = 0.0
        self.saturated = False
        self._history = deque([(0.0, self.position_rad)])
        self._delayed_command_rad = self.position_rad

    def update(self, command_rad: float, step_s: float) -> float:
        """Advance actuator state by one sample and return position [rad]."""
        if not np.isfinite(command_rad):
            raise ValueError("command_rad must be finite")
        if not np.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("step_s must be positive and finite")
        limited_command = float(
            np.clip(command_rad, -self.limits.position_limit_rad, self.limits.position_limit_rad)
        )
        self.saturated = not np.isclose(limited_command, command_rad)
        self._history.append((self.elapsed_time_s, limited_command))
        query_time_s = self.elapsed_time_s - self.limits.command_delay_s
        while len(self._history) >= 2 and self._history[1][0] <= query_time_s:
            self._history.popleft()
        if self._history[0][0] <= query_time_s:
            self._delayed_command_rad = self._history[0][1]

        requested_rate = (
            self._delayed_command_rad - self.position_rad
        ) / self.limits.time_constant_s
        self.rate_radps = float(
            np.clip(requested_rate, -self.limits.rate_limit_radps, self.limits.rate_limit_radps)
        )
        if not np.isclose(self.rate_radps, requested_rate):
            self.saturated = True
        unconstrained_position = self.position_rad + step_s * self.rate_radps
        self.position_rad = float(
            np.clip(
                unconstrained_position,
                -self.limits.position_limit_rad,
                self.limits.position_limit_rad,
            )
        )
        if not np.isclose(self.position_rad, unconstrained_position):
            self.saturated = True
            self.rate_radps = 0.0
        self.elapsed_time_s += step_s
        return self.position_rad


@dataclass(frozen=True, slots=True)
class ActuatorAllocator:
    """Diagonal fictional control-channel torque allocation."""

    moment_per_command_nm: FloatArray
    command_limit_rad: FloatArray

    def __init__(
        self, moment_per_command_nm: npt.ArrayLike, command_limit_rad: npt.ArrayLike
    ) -> None:
        effectiveness = as_vector(moment_per_command_nm, 3, name="moment_per_command_nm")
        limits = as_vector(command_limit_rad, 3, name="command_limit_rad")
        if np.any(np.abs(effectiveness) <= 1.0e-12):
            raise ValueError("each moment effectiveness must be nonzero")
        if np.any(limits <= 0.0):
            raise ValueError("command limits must be positive")
        object.__setattr__(self, "moment_per_command_nm", effectiveness)
        object.__setattr__(self, "command_limit_rad", limits)

    def allocate(self, requested_moment_body_nm: npt.ArrayLike) -> FloatArray:
        """Map requested body moment [N m] to bounded channel commands [rad]."""
        requested = as_vector(requested_moment_body_nm, 3, name="requested_moment_body_nm")
        command = requested / self.moment_per_command_nm
        return np.clip(command, -self.command_limit_rad, self.command_limit_rad)

    def achieved_moment_body_nm(self, command_rad: npt.ArrayLike) -> FloatArray:
        """Return moment generated by bounded commands [N m]."""
        command = as_vector(command_rad, 3, name="command_rad")
        command = np.clip(command, -self.command_limit_rad, self.command_limit_rad)
        return self.moment_per_command_nm * command
