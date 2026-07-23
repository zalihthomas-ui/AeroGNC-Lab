"""Cascaded fixed-wing autopilot (successive loop closure).

Converts a :class:`~aerognc.gnc.waypoint_guidance.GuidanceCommand` and the current
:class:`~aerognc.navigation.state.NavigationState` into an outer
:class:`ControlCommand` (roll / pitch / throttle) and an inner
:class:`ActuatorCommand` (aileron / elevator / rudder / throttle), following the
successive-loop-closure structure in Beard & McLain, *Small Unmanned Aircraft*:

* lateral: course -> roll (PI, bank-limited, + guidance roll feedforward) ->
  aileron (roll error with roll-rate damping); a yaw damper drives the rudder;
* longitudinal: selectable altitude/airspeed PI loops or a TECS-style specific-
  energy sum/balance controller -> trim-aware pitch/throttle commands -> elevator
  with pitch-rate damping.

The three integrating outer loops reuse the project's :class:`PIDController`
(anti-windup, derivative filtering, output/integral limits). The inner attitude
loops are proportional-plus-rate-damping using the measured body rates.

**Sign convention (internal, normalized).** Positive aileron -> roll right;
positive elevator -> pitch up; positive rudder -> yaw right; surface commands are
normalized to ``[-1, 1]`` and throttle to ``[0, 1]``. The internal simulation
backend uses the same convention. Mapping these to a specific airframe's surfaces
and directions is a calibration step required before any hardware use.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from aerognc.gnc.pid import PIDController, PIDGains
from aerognc.gnc.total_energy_control import (
    LongitudinalControlMode,
    TotalEnergyControlGains,
    TotalEnergyController,
)
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
class AutopilotTrim:
    """Straight-flight feedforward commands resolved before loop activation."""

    pitch_rad: float = 0.0
    elevator_command: float = 0.0
    throttle: float = 0.5

    def __post_init__(self) -> None:
        values = np.asarray([self.pitch_rad, self.elevator_command, self.throttle])
        if not np.all(np.isfinite(values)):
            raise ValueError("autopilot trim values must be finite")
        if abs(self.pitch_rad) >= 0.5 * np.pi:
            raise ValueError("autopilot pitch trim must lie within (-pi/2, pi/2)")
        if not -1.0 <= self.elevator_command <= 1.0:
            raise ValueError("autopilot elevator trim command must lie in [-1, 1]")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("autopilot throttle trim must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class LongitudinalControlDiagnostics:
    """Reference, energy-error, and saturation evidence for one control step."""

    mode: str
    altitude_reference_m: float
    airspeed_reference_mps: float
    potential_energy_error_m2ps2: float
    kinetic_energy_error_m2ps2: float
    total_energy_error_m2ps2: float
    energy_balance_error_m2ps2: float
    pitch_saturated: bool
    throttle_saturated: bool


@dataclass(frozen=True, slots=True)
class AutopilotOutput:
    """Full autopilot result plus tracking-error diagnostics."""

    control: ControlCommand
    actuator: ActuatorCommand
    course_error_rad: float
    altitude_error_m: float
    airspeed_error_mps: float
    longitudinal: LongitudinalControlDiagnostics


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
        gains = np.asarray(
            [
                self.course_kp,
                self.course_ki,
                self.altitude_kp_rad_per_m,
                self.altitude_ki,
                self.airspeed_kp,
                self.airspeed_ki,
                self.roll_kp,
                self.roll_rate_kd,
                self.pitch_kp,
                self.pitch_rate_kd,
                self.yaw_damper_kd,
            ]
        )
        limits = [
            self.bank_limit_rad,
            self.pitch_limit_rad,
            self.integral_bank_limit_rad,
            self.integral_pitch_limit_rad,
            self.throttle_delta_limit,
        ]
        if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
            raise ValueError("autopilot feedback gains must be nonnegative and finite")
        if not np.all(np.isfinite(limits)) or np.any(np.asarray(limits) <= 0.0):
            raise ValueError("autopilot limits must be positive and finite")
        if not 0.0 <= self.throttle_trim <= 1.0:
            raise ValueError("throttle_trim must be in [0, 1]")
        if not np.isfinite(self.elevator_trim) or not -1.0 <= self.elevator_trim <= 1.0:
            raise ValueError("elevator_trim must be finite and lie in [-1, 1]")


class FixedWingAutopilot:
    """Cascaded lateral/longitudinal fixed-wing autopilot."""

    def __init__(
        self,
        gains: AutopilotGains | None = None,
        *,
        longitudinal_mode: LongitudinalControlMode = LongitudinalControlMode.ALTITUDE_AIRSPEED,
        total_energy_gains: TotalEnergyControlGains | None = None,
        trim: AutopilotTrim | None = None,
    ) -> None:
        self.gains = gains or AutopilotGains()
        if not isinstance(longitudinal_mode, LongitudinalControlMode):
            raise ValueError("longitudinal_mode must be a LongitudinalControlMode")
        self.longitudinal_mode = longitudinal_mode
        self.total_energy_gains = total_energy_gains or TotalEnergyControlGains()
        self.trim = trim or AutopilotTrim(
            pitch_rad=0.0,
            elevator_command=self.gains.elevator_trim,
            throttle=self.gains.throttle_trim,
        )
        if abs(self.trim.pitch_rad) > self.gains.pitch_limit_rad:
            raise ValueError("autopilot pitch trim exceeds the configured pitch limit")
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
        self._total_energy_controller = TotalEnergyController(
            self.total_energy_gains,
            pitch_trim_rad=self.trim.pitch_rad,
            throttle_trim=self.trim.throttle,
            pitch_limit_rad=self.gains.pitch_limit_rad,
            throttle_delta_limit=self.gains.throttle_delta_limit,
        )
        self._last_control: ControlCommand | None = None
        self._pending_bumpless_control: ControlCommand | None = None

    def reset(self) -> None:
        """Reset all outer-loop integrators (bumpless re-engage)."""
        self._course_pid.reset()
        self._altitude_pid.reset()
        self._airspeed_pid.reset()
        self._total_energy_controller.reset()
        self._last_control = None
        self._pending_bumpless_control = None

    def set_longitudinal_mode(self, mode: LongitudinalControlMode) -> None:
        """Switch outer-loop architecture while retaining the prior commands."""
        if not isinstance(mode, LongitudinalControlMode):
            raise ValueError("mode must be a LongitudinalControlMode")
        if mode is self.longitudinal_mode:
            return
        self._pending_bumpless_control = self._last_control
        self.longitudinal_mode = mode
        if mode is LongitudinalControlMode.TOTAL_ENERGY:
            self._total_energy_controller.reset()
        else:
            self._altitude_pid.reset()
            self._airspeed_pid.reset()

    def provenance(self) -> Mapping[str, object]:
        """Return mode, gains, and resolved trim for deterministic run metadata."""
        return {
            "implementation": type(self).__name__,
            "longitudinal_mode": self.longitudinal_mode.value,
            "trim": asdict(self.trim),
            "total_energy_gains": asdict(self.total_energy_gains),
        }

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

        # --- longitudinal outer loop: classic PI or total-energy coordination ---
        altitude_error = guidance.altitude_command_m - state.altitude_m
        airspeed_error = guidance.airspeed_command_mps - state.airspeed_mps
        pitch_command, throttle, longitudinal = self._longitudinal_update(
            guidance,
            state,
            altitude_error,
            airspeed_error,
            dt_s,
        )
        elevator = float(
            np.clip(
                self.gains.pitch_kp * (pitch_command - state.pitch_rad)
                - self.gains.pitch_rate_kd * pitch_rate
                + self.trim.elevator_command,
                -1.0,
                1.0,
            )
        )
        output = AutopilotOutput(
            control=ControlCommand(roll_command, pitch_command, throttle),
            actuator=ActuatorCommand(aileron, elevator, rudder, throttle),
            course_error_rad=course_error,
            altitude_error_m=altitude_error,
            airspeed_error_mps=airspeed_error,
            longitudinal=longitudinal,
        )
        self._last_control = output.control
        return output

    def _longitudinal_update(
        self,
        guidance: GuidanceCommand,
        state: NavigationState,
        altitude_error: float,
        airspeed_error: float,
        dt_s: float,
    ) -> tuple[float, float, LongitudinalControlDiagnostics]:
        pending = self._pending_bumpless_control
        self._pending_bumpless_control = None
        if self.longitudinal_mode is LongitudinalControlMode.TOTAL_ENERGY:
            if pending is None:
                energy = self._total_energy_controller.update(guidance, state, dt_s)
            else:
                energy = self._total_energy_controller.activate(
                    guidance,
                    state,
                    pitch_command_rad=pending.pitch_command_rad,
                    throttle_command=pending.throttle_command,
                )
            diagnostics = LongitudinalControlDiagnostics(
                mode=self.longitudinal_mode.value,
                altitude_reference_m=energy.altitude_reference_m,
                airspeed_reference_mps=energy.airspeed_reference_mps,
                potential_energy_error_m2ps2=energy.potential_energy_error_m2ps2,
                kinetic_energy_error_m2ps2=energy.kinetic_energy_error_m2ps2,
                total_energy_error_m2ps2=energy.total_energy_error_m2ps2,
                energy_balance_error_m2ps2=energy.energy_balance_error_m2ps2,
                pitch_saturated=energy.pitch_saturated,
                throttle_saturated=energy.throttle_saturated,
            )
            return energy.pitch_command_rad, energy.throttle_command, diagnostics

        if pending is not None:
            pitch_delta = self._altitude_pid.track_output(
                altitude_error,
                pending.pitch_command_rad - self.trim.pitch_rad,
            )
            throttle_delta = self._airspeed_pid.track_output(
                airspeed_error,
                pending.throttle_command - self.trim.throttle,
            )
        else:
            pitch_delta = self._altitude_pid.update(altitude_error, dt_s)
            throttle_delta = self._airspeed_pid.update(airspeed_error, dt_s)
        raw_pitch = self.trim.pitch_rad + pitch_delta
        raw_throttle = self.trim.throttle + throttle_delta
        pitch = float(np.clip(raw_pitch, -self.gains.pitch_limit_rad, self.gains.pitch_limit_rad))
        throttle = float(np.clip(raw_throttle, 0.0, 1.0))
        if not np.isclose(pitch, raw_pitch):
            self._altitude_pid.track_output(altitude_error, pitch - self.trim.pitch_rad)
        if not np.isclose(throttle, raw_throttle):
            self._airspeed_pid.track_output(airspeed_error, throttle - self.trim.throttle)
        gravity = self.total_energy_gains.gravity_mps2
        potential_error = gravity * altitude_error
        kinetic_error = 0.5 * (guidance.airspeed_command_mps**2 - state.airspeed_mps**2)
        diagnostics = LongitudinalControlDiagnostics(
            mode=self.longitudinal_mode.value,
            altitude_reference_m=guidance.altitude_command_m,
            airspeed_reference_mps=guidance.airspeed_command_mps,
            potential_energy_error_m2ps2=float(potential_error),
            kinetic_energy_error_m2ps2=float(kinetic_error),
            total_energy_error_m2ps2=float(potential_error + kinetic_error),
            energy_balance_error_m2ps2=float(potential_error - kinetic_error),
            pitch_saturated=self._altitude_pid.saturated or not np.isclose(pitch, raw_pitch),
            throttle_saturated=self._airspeed_pid.saturated
            or not np.isclose(throttle, raw_throttle),
        )
        return pitch, throttle, diagnostics


__all__ = [
    "ActuatorCommand",
    "AutopilotGains",
    "AutopilotOutput",
    "AutopilotTrim",
    "ControlCommand",
    "FixedWingAutopilot",
    "LongitudinalControlDiagnostics",
    "LongitudinalControlMode",
    "TotalEnergyControlGains",
]
