"""Fixed-wing control-surface actuator bank with failure injection.

Maps the autopilot's normalized surface commands (``[-1, 1]``) and throttle
(``[0, 1]``) to physical deflections [rad] / throttle fraction, applying
first-order lag, rate and position limits, command delay, neutral and trim
offsets (reusing :class:`~aerognc.vehicle.actuators.FirstOrderActuator`), plus an
injectable failure mode per channel for fault-tolerance studies.

Sign convention matches :mod:`aerognc.gnc.fixedwing_autopilot`: positive aileron
=> roll right, positive elevator => pitch up, positive rudder => yaw right.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from aerognc.vehicle.actuators import ActuatorLimits, FirstOrderActuator


class SurfaceFailureMode(StrEnum):
    """Per-channel actuator failure modes."""

    NONE = "none"
    STUCK = "stuck"
    REDUCED_AUTHORITY = "reduced_authority"
    REVERSED = "reversed"
    OSCILLATING = "oscillating"
    LOSS = "loss"  # centres and holds neutral (no authority)


@dataclass(frozen=True, slots=True)
class ControlSurfaceConfig:
    """Physical limits and offsets for one aerodynamic surface."""

    max_deflection_rad: float
    time_constant_s: float = 0.1
    rate_limit_radps: float = float(np.deg2rad(120.0))
    command_delay_s: float = 0.0
    neutral_rad: float = 0.0
    trim_rad: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_deflection_rad) or self.max_deflection_rad <= 0.0:
            raise ValueError("max_deflection_rad must be positive and finite")


class ControlSurface:
    """A single deflection channel with actuator dynamics and failure injection."""

    def __init__(
        self,
        config: ControlSurfaceConfig,
        *,
        failure: SurfaceFailureMode = SurfaceFailureMode.NONE,
        reduced_authority_fraction: float = 0.3,
        oscillation_amplitude_rad: float = float(np.deg2rad(5.0)),
        oscillation_frequency_hz: float = 3.0,
    ) -> None:
        # Position limit spans neutral+trim plus full authority in either sense.
        position_limit = abs(config.neutral_rad) + abs(config.trim_rad) + config.max_deflection_rad
        self.config = config
        self.failure = failure
        self.reduced_authority_fraction = float(np.clip(reduced_authority_fraction, 0.0, 1.0))
        self.oscillation_amplitude_rad = oscillation_amplitude_rad
        self.oscillation_frequency_hz = oscillation_frequency_hz
        self._actuator = FirstOrderActuator(
            ActuatorLimits(
                time_constant_s=config.time_constant_s,
                position_limit_rad=position_limit,
                rate_limit_radps=config.rate_limit_radps,
                command_delay_s=config.command_delay_s,
            ),
            initial_position_rad=config.neutral_rad + config.trim_rad,
        )

    @property
    def deflection_rad(self) -> float:
        """Return the current physical deflection [rad]."""
        return self._actuator.position_rad

    @property
    def saturated(self) -> bool:
        """Return whether the actuator hit a rate/position limit last step."""
        return self._actuator.saturated

    def reset(self) -> None:
        """Reset to the neutral+trim position."""
        self._actuator.reset(self.config.neutral_rad + self.config.trim_rad)

    def update(self, normalized_command: float, step_s: float) -> float:
        """Advance one step and return the physical deflection [rad]."""
        if not np.isfinite(normalized_command):
            raise ValueError("normalized_command must be finite")
        command = float(np.clip(normalized_command, -1.0, 1.0))

        if self.failure is SurfaceFailureMode.STUCK:
            # Ignore the command: hold the last commanded position.
            return self._actuator.position_rad
        if self.failure is SurfaceFailureMode.REVERSED:
            command = -command
        elif self.failure is SurfaceFailureMode.REDUCED_AUTHORITY:
            command *= self.reduced_authority_fraction

        neutral = self.config.neutral_rad + self.config.trim_rad
        if self.failure is SurfaceFailureMode.LOSS:
            target = neutral
        else:
            target = neutral + command * self.config.max_deflection_rad

        deflection = self._actuator.update(target, step_s)
        if self.failure is SurfaceFailureMode.OSCILLATING:
            phase = 2.0 * np.pi * self.oscillation_frequency_hz * self._actuator.elapsed_time_s
            limit = abs(neutral) + self.config.max_deflection_rad
            deflection = float(
                np.clip(deflection + self.oscillation_amplitude_rad * np.sin(phase), -limit, limit)
            )
        return deflection


@dataclass(frozen=True, slots=True)
class SurfaceDeflections:
    """Physical actuator outputs for one step."""

    aileron_rad: float
    elevator_rad: float
    rudder_rad: float
    throttle: float
    saturated: bool


class ControlSurfaceSet:
    """Aileron/elevator/rudder deflection channels plus a first-order throttle."""

    def __init__(
        self,
        aileron: ControlSurface,
        elevator: ControlSurface,
        rudder: ControlSurface,
        *,
        throttle_time_constant_s: float = 0.6,
        throttle_stuck: bool = False,
        initial_throttle: float = 0.0,
    ) -> None:
        if throttle_time_constant_s <= 0.0:
            raise ValueError("throttle_time_constant_s must be positive")
        if not np.isfinite(initial_throttle) or not 0.0 <= initial_throttle <= 1.0:
            raise ValueError("initial_throttle must lie in [0, 1]")
        self.aileron = aileron
        self.elevator = elevator
        self.rudder = rudder
        self._throttle_tc_s = throttle_time_constant_s
        self._initial_throttle = float(initial_throttle)
        self._throttle = self._initial_throttle
        self._throttle_stuck = throttle_stuck

    @classmethod
    def from_limits(
        cls,
        *,
        aileron_limit_rad: float,
        elevator_limit_rad: float,
        rudder_limit_rad: float,
        time_constant_s: float = 0.12,
        rate_limit_radps: float = float(np.deg2rad(120.0)),
        throttle_time_constant_s: float = 0.6,
        initial_throttle: float = 0.0,
    ) -> "ControlSurfaceSet":
        """Build a nominal (fault-free) surface set from deflection limits."""

        def make(limit_rad: float) -> ControlSurface:
            return ControlSurface(
                ControlSurfaceConfig(
                    max_deflection_rad=limit_rad,
                    time_constant_s=time_constant_s,
                    rate_limit_radps=rate_limit_radps,
                )
            )

        return cls(
            make(aileron_limit_rad),
            make(elevator_limit_rad),
            make(rudder_limit_rad),
            throttle_time_constant_s=throttle_time_constant_s,
            initial_throttle=initial_throttle,
        )

    def reset(self) -> None:
        """Reset all channels."""
        self.aileron.reset()
        self.elevator.reset()
        self.rudder.reset()
        self._throttle = self._initial_throttle

    def update(
        self,
        aileron_cmd: float,
        elevator_cmd: float,
        rudder_cmd: float,
        throttle_cmd: float,
        step_s: float,
    ) -> SurfaceDeflections:
        """Advance all channels one step and return physical deflections."""
        if not np.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("step_s must be positive and finite")
        aileron_rad = self.aileron.update(aileron_cmd, step_s)
        elevator_rad = self.elevator.update(elevator_cmd, step_s)
        rudder_rad = self.rudder.update(rudder_cmd, step_s)
        if not self._throttle_stuck:
            target = float(np.clip(throttle_cmd, 0.0, 1.0))
            self._throttle += (target - self._throttle) * min(step_s / self._throttle_tc_s, 1.0)
        return SurfaceDeflections(
            aileron_rad=aileron_rad,
            elevator_rad=elevator_rad,
            rudder_rad=rudder_rad,
            throttle=float(np.clip(self._throttle, 0.0, 1.0)),
            saturated=self.aileron.saturated or self.elevator.saturated or self.rudder.saturated,
        )
