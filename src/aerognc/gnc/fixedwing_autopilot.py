"""Cascaded fixed-wing autopilot (successive loop closure).

Converts a :class:`~aerognc.gnc.waypoint_guidance.GuidanceCommand` and the current
:class:`~aerognc.navigation.state.NavigationState` into an outer
:class:`ControlCommand` (roll / pitch / throttle) and an inner
:class:`ActuatorCommand` (aileron / elevator / rudder / throttle), following the
successive-loop-closure structure in Beard & McLain, *Small Unmanned Aircraft*:

* lateral: course -> roll (PI, bank-limited, + guidance roll feedforward) ->
  aileron (roll error with roll-rate damping); a yaw damper drives the rudder;
* longitudinal: altitude -> pitch (PI, pitch-limited) -> elevator (pitch error
  with pitch-rate damping); airspeed -> throttle (PI about a trim throttle).

The three integrating outer loops reuse the project's :class:`PIDController`
(anti-windup, derivative filtering, output/integral limits). The inner attitude
loops are proportional-plus-rate-damping using the measured body rates.

**Sign convention (internal, normalized).** Positive aileron -> roll right;
positive elevator -> pitch up; positive rudder -> yaw right; surface commands are
normalized to ``[-1, 1]`` and throttle to ``[0, 1]``. The internal simulation
backend uses the same convention. Mapping these to a specific airframe's surfaces
and directions is a calibration step required before any hardware use.
"""

from dataclasses import dataclass

import numpy as np

from aerognc.gnc.pid import PIDController, PIDGains
from aerognc.gnc.waypoint_guidance import GuidanceCommand
from aerognc.mathematics.local_frame import wrap_to_pi
from aerognc.navigation.state import NavigationState


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """Outer-loop attitude/throttle command."""

    roll_command_rad: float
    pitch_command_rad: float
    throttle_command: float


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """Inner-loop normalized surface commands and throttle."""

    aileron: float  # [-1, 1]
    elevator: float  # [-1, 1]
    rudder: float  # [-1, 1]
    throttle: float  # [0, 1]


@dataclass(frozen=True, slots=True)
class AutopilotOutput:
    """Full autopilot result plus tracking-error diagnostics."""

    control: ControlCommand
    actuator: ActuatorCommand
    course_error_rad: float
    altitude_error_m: float
    airspeed_error_mps: float


@dataclass(frozen=True, slots=True)
class AutopilotGains:
    """Autopilot tuning, limits, and trim (SI units, normalized surfaces)."""

    # Outer loops.
    course_kp: float = 1.2
    course_ki: float = 0.1
    altitude_kp_rad_per_m: float = 0.03
    altitude_ki: float = 0.002
    airspeed_kp: float = 0.06
    airspeed_ki: float = 0.02
    # Inner loops.
    roll_kp: float = 1.6
    roll_rate_kd: float = 0.35
    pitch_kp: float = 2.2
    pitch_rate_kd: float = 0.4
    yaw_damper_kd: float = 0.5
    # Limits and trim.
    bank_limit_rad: float = float(np.deg2rad(35.0))
    pitch_limit_rad: float = float(np.deg2rad(20.0))
    integral_bank_limit_rad: float = float(np.deg2rad(15.0))
    integral_pitch_limit_rad: float = float(np.deg2rad(10.0))
    throttle_trim: float = 0.5
    throttle_delta_limit: float = 0.5
    elevator_trim: float = 0.0

    def __post_init__(self) -> None:
        limits = [
            self.bank_limit_rad,
            self.pitch_limit_rad,
            self.integral_bank_limit_rad,
            self.integral_pitch_limit_rad,
            self.throttle_delta_limit,
        ]
        if not np.all(np.isfinite(limits)) or np.any(np.asarray(limits) <= 0.0):
            raise ValueError("autopilot limits must be positive and finite")
        if not 0.0 <= self.throttle_trim <= 1.0:
            raise ValueError("throttle_trim must be in [0, 1]")


class FixedWingAutopilot:
    """Cascaded lateral/longitudinal fixed-wing autopilot."""

    def __init__(self, gains: AutopilotGains | None = None) -> None:
        self.gains = gains or AutopilotGains()
        self._course_pid = PIDController(
            PIDGains(
                proportional=self.gains.course_kp,
                integral=self.gains.course_ki,
                derivative=0.0,
                output_min=-self.gains.bank_limit_rad,
                output_max=self.gains.bank_limit_rad,
                integral_min=-self.gains.integral_bank_limit_rad,
                integral_max=self.gains.integral_bank_limit_rad,
            )
        )
        self._altitude_pid = PIDController(
            PIDGains(
                proportional=self.gains.altitude_kp_rad_per_m,
                integral=self.gains.altitude_ki,
                derivative=0.0,
                output_min=-self.gains.pitch_limit_rad,
                output_max=self.gains.pitch_limit_rad,
                integral_min=-self.gains.integral_pitch_limit_rad,
                integral_max=self.gains.integral_pitch_limit_rad,
            )
        )
        self._airspeed_pid = PIDController(
            PIDGains(
                proportional=self.gains.airspeed_kp,
                integral=self.gains.airspeed_ki,
                derivative=0.0,
                output_min=-self.gains.throttle_delta_limit,
                output_max=self.gains.throttle_delta_limit,
                integral_min=-self.gains.throttle_delta_limit,
                integral_max=self.gains.throttle_delta_limit,
            )
        )

    def reset(self) -> None:
        """Reset all outer-loop integrators (bumpless re-engage)."""
        self._course_pid.reset()
        self._altitude_pid.reset()
        self._airspeed_pid.reset()

    def update(
        self, guidance: GuidanceCommand, state: NavigationState, dt_s: float
    ) -> AutopilotOutput:
        """Return the outer control command and inner actuator command."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        roll_rate = float(state.angular_rate_body_radps[0])
        pitch_rate = float(state.angular_rate_body_radps[1])
        yaw_rate = float(state.angular_rate_body_radps[2])

        # --- lateral: course -> roll -> aileron ---
        course_error = wrap_to_pi(guidance.course_command_rad - state.course_rad)
        roll_from_course = self._course_pid.update(course_error, dt_s)
        roll_command = float(
            np.clip(
                roll_from_course + guidance.roll_feedforward_rad,
                -self.gains.bank_limit_rad,
                self.gains.bank_limit_rad,
            )
        )
        aileron = float(
            np.clip(
                self.gains.roll_kp * (roll_command - state.roll_rad)
                - self.gains.roll_rate_kd * roll_rate,
                -1.0,
                1.0,
            )
        )
        rudder = float(np.clip(-self.gains.yaw_damper_kd * yaw_rate, -1.0, 1.0))

        # --- longitudinal: altitude -> pitch -> elevator ---
        altitude_error = guidance.altitude_command_m - state.altitude_m
        pitch_command = self._altitude_pid.update(altitude_error, dt_s)
        elevator = float(
            np.clip(
                self.gains.pitch_kp * (pitch_command - state.pitch_rad)
                - self.gains.pitch_rate_kd * pitch_rate
                + self.gains.elevator_trim,
                -1.0,
                1.0,
            )
        )

        # --- airspeed -> throttle ---
        airspeed_error = guidance.airspeed_command_mps - state.airspeed_mps
        throttle_delta = self._airspeed_pid.update(airspeed_error, dt_s)
        throttle = float(np.clip(self.gains.throttle_trim + throttle_delta, 0.0, 1.0))

        return AutopilotOutput(
            control=ControlCommand(roll_command, pitch_command, throttle),
            actuator=ActuatorCommand(aileron, elevator, rudder, throttle),
            course_error_rad=course_error,
            altitude_error_m=altitude_error,
            airspeed_error_mps=airspeed_error,
        )
