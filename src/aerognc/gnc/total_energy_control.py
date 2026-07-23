"""Bumpless total-energy altitude/airspeed control for fixed-wing simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from aerognc.gnc.pid import PIDController, PIDGains
from aerognc.gnc.waypoint_guidance import GuidanceCommand
from aerognc.navigation.state import NavigationState


class LongitudinalControlMode(StrEnum):
    """Selectable fixed-wing longitudinal outer-loop architecture."""

    ALTITUDE_AIRSPEED = "altitude_airspeed"
    TOTAL_ENERGY = "total_energy"


@dataclass(frozen=True, slots=True)
class TotalEnergyControlGains:
    """TECS-style specific-energy gains, reference limits, and feedforward."""

    total_energy_kp: float = 0.0015
    total_energy_ki: float = 0.00008
    energy_balance_kp: float = 0.0020
    energy_balance_ki: float = 0.00010
    altitude_reference_rate_limit_mps: float = 6.0
    airspeed_reference_rate_limit_mps2: float = 2.0
    climb_rate_throttle_feedforward_per_mps: float = 0.025
    flight_path_angle_feedforward_gain: float = 0.65
    gravity_mps2: float = 9.80665

    def __post_init__(self) -> None:
        gains = np.asarray(
            [
                self.total_energy_kp,
                self.total_energy_ki,
                self.energy_balance_kp,
                self.energy_balance_ki,
                self.climb_rate_throttle_feedforward_per_mps,
                self.flight_path_angle_feedforward_gain,
            ],
            dtype=np.float64,
        )
        limits = np.asarray(
            [
                self.altitude_reference_rate_limit_mps,
                self.airspeed_reference_rate_limit_mps2,
                self.gravity_mps2,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
            raise ValueError("total-energy gains and feedforward terms must be nonnegative")
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("total-energy reference limits and gravity must be positive")


@dataclass(frozen=True, slots=True)
class TotalEnergyControlOutput:
    """TECS commands plus auditable reference, energy, and saturation channels."""

    pitch_command_rad: float
    throttle_command: float
    altitude_reference_m: float
    airspeed_reference_mps: float
    potential_energy_error_m2ps2: float
    kinetic_energy_error_m2ps2: float
    total_energy_error_m2ps2: float
    energy_balance_error_m2ps2: float
    pitch_saturated: bool
    throttle_saturated: bool


class TotalEnergyController:
    """Coordinate pitch and throttle through specific-energy sum and balance."""

    def __init__(
        self,
        gains: TotalEnergyControlGains | None = None,
        *,
        pitch_trim_rad: float = 0.0,
        throttle_trim: float = 0.5,
        pitch_limit_rad: float = float(np.deg2rad(20.0)),
        throttle_delta_limit: float = 0.5,
    ) -> None:
        self.gains = gains or TotalEnergyControlGains()
        values = np.asarray(
            [pitch_trim_rad, throttle_trim, pitch_limit_rad, throttle_delta_limit],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("total-energy trim and limits must be finite")
        if pitch_limit_rad <= 0.0 or abs(pitch_trim_rad) > pitch_limit_rad:
            raise ValueError("pitch trim must lie within the positive pitch limit")
        if not 0.0 <= throttle_trim <= 1.0 or throttle_delta_limit <= 0.0:
            raise ValueError("throttle trim/limit are outside their domains")
        self.pitch_trim_rad = float(pitch_trim_rad)
        self.throttle_trim = float(throttle_trim)
        self.pitch_limit_rad = float(pitch_limit_rad)
        self.throttle_delta_limit = float(throttle_delta_limit)

        throttle_min = max(-self.throttle_trim, -self.throttle_delta_limit)
        throttle_max = min(1.0 - self.throttle_trim, self.throttle_delta_limit)
        pitch_min = -self.pitch_limit_rad - self.pitch_trim_rad
        pitch_max = self.pitch_limit_rad - self.pitch_trim_rad
        self._total_energy_pid = PIDController(
            PIDGains(
                proportional=self.gains.total_energy_kp,
                integral=self.gains.total_energy_ki,
                derivative=0.0,
                output_min=throttle_min,
                output_max=throttle_max,
                integral_min=self._integral_bound(throttle_min, self.gains.total_energy_ki),
                integral_max=self._integral_bound(throttle_max, self.gains.total_energy_ki),
            )
        )
        self._energy_balance_pid = PIDController(
            PIDGains(
                proportional=self.gains.energy_balance_kp,
                integral=self.gains.energy_balance_ki,
                derivative=0.0,
                output_min=pitch_min,
                output_max=pitch_max,
                integral_min=self._integral_bound(pitch_min, self.gains.energy_balance_ki),
                integral_max=self._integral_bound(pitch_max, self.gains.energy_balance_ki),
            )
        )
        self.reset()

    def reset(self) -> None:
        """Clear loop state so the next update activates bumplessly at trim."""
        self._total_energy_pid.reset()
        self._energy_balance_pid.reset()
        self._altitude_reference_m: float | None = None
        self._airspeed_reference_mps: float | None = None

    def activate(
        self,
        guidance: GuidanceCommand,
        state: NavigationState,
        *,
        pitch_command_rad: float | None = None,
        throttle_command: float | None = None,
    ) -> TotalEnergyControlOutput:
        """Initialize references and integrators at supplied commands without a bump."""
        requested_pitch = self.pitch_trim_rad if pitch_command_rad is None else pitch_command_rad
        requested_throttle = self.throttle_trim if throttle_command is None else throttle_command
        if not np.all(np.isfinite([requested_pitch, requested_throttle])):
            raise ValueError("activation commands must be finite")
        pitch = float(np.clip(requested_pitch, -self.pitch_limit_rad, self.pitch_limit_rad))
        throttle = float(np.clip(requested_throttle, 0.0, 1.0))
        self._altitude_reference_m = state.altitude_m
        self._airspeed_reference_mps = state.airspeed_mps
        throttle_feedforward, pitch_feedforward = self._feedforward(guidance, state.airspeed_mps)
        self._total_energy_pid.track_output(
            0.0,
            throttle - self.throttle_trim - throttle_feedforward,
        )
        self._energy_balance_pid.track_output(
            0.0,
            pitch - self.pitch_trim_rad - pitch_feedforward,
        )
        return TotalEnergyControlOutput(
            pitch_command_rad=pitch,
            throttle_command=throttle,
            altitude_reference_m=state.altitude_m,
            airspeed_reference_mps=state.airspeed_mps,
            potential_energy_error_m2ps2=0.0,
            kinetic_energy_error_m2ps2=0.0,
            total_energy_error_m2ps2=0.0,
            energy_balance_error_m2ps2=0.0,
            pitch_saturated=not np.isclose(pitch, requested_pitch),
            throttle_saturated=not np.isclose(throttle, requested_throttle),
        )

    def update(
        self,
        guidance: GuidanceCommand,
        state: NavigationState,
        dt_s: float,
    ) -> TotalEnergyControlOutput:
        """Advance reference governors and the total/balance energy loops."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("total-energy dt_s must be positive and finite")
        if self._altitude_reference_m is None or self._airspeed_reference_mps is None:
            return self.activate(guidance, state)

        self._altitude_reference_m = self._slew(
            self._altitude_reference_m,
            guidance.altitude_command_m,
            self.gains.altitude_reference_rate_limit_mps,
            dt_s,
        )
        self._airspeed_reference_mps = self._slew(
            self._airspeed_reference_mps,
            guidance.airspeed_command_mps,
            self.gains.airspeed_reference_rate_limit_mps2,
            dt_s,
        )
        potential_error = self.gains.gravity_mps2 * (self._altitude_reference_m - state.altitude_m)
        kinetic_error = 0.5 * (self._airspeed_reference_mps**2 - state.airspeed_mps**2)
        total_error = potential_error + kinetic_error
        balance_error = potential_error - kinetic_error
        throttle_feedforward, pitch_feedforward = self._feedforward(
            guidance,
            self._airspeed_reference_mps,
        )
        throttle_delta = self._total_energy_pid.update(total_error, dt_s)
        pitch_delta = self._energy_balance_pid.update(balance_error, dt_s)
        raw_throttle = self.throttle_trim + throttle_feedforward + throttle_delta
        raw_pitch = self.pitch_trim_rad + pitch_feedforward + pitch_delta
        throttle = float(np.clip(raw_throttle, 0.0, 1.0))
        pitch = float(np.clip(raw_pitch, -self.pitch_limit_rad, self.pitch_limit_rad))
        throttle_saturated = self._total_energy_pid.saturated or not np.isclose(
            throttle, raw_throttle
        )
        pitch_saturated = self._energy_balance_pid.saturated or not np.isclose(pitch, raw_pitch)
        if not np.isclose(throttle, raw_throttle):
            self._total_energy_pid.track_output(
                total_error,
                throttle - self.throttle_trim - throttle_feedforward,
            )
        if not np.isclose(pitch, raw_pitch):
            self._energy_balance_pid.track_output(
                balance_error,
                pitch - self.pitch_trim_rad - pitch_feedforward,
            )
        return TotalEnergyControlOutput(
            pitch_command_rad=pitch,
            throttle_command=throttle,
            altitude_reference_m=self._altitude_reference_m,
            airspeed_reference_mps=self._airspeed_reference_mps,
            potential_energy_error_m2ps2=float(potential_error),
            kinetic_energy_error_m2ps2=float(kinetic_error),
            total_energy_error_m2ps2=float(total_error),
            energy_balance_error_m2ps2=float(balance_error),
            pitch_saturated=pitch_saturated,
            throttle_saturated=throttle_saturated,
        )

    @staticmethod
    def _integral_bound(output_bound: float, integral_gain: float) -> float:
        if integral_gain == 0.0:
            return -1.0e12 if output_bound < 0.0 else 1.0e12
        return output_bound / integral_gain

    @staticmethod
    def _slew(current: float, target: float, rate_limit: float, dt_s: float) -> float:
        return float(current + np.clip(target - current, -rate_limit * dt_s, rate_limit * dt_s))

    def _feedforward(
        self,
        guidance: GuidanceCommand,
        airspeed_reference_mps: float,
    ) -> tuple[float, float]:
        climb_rate = float(
            np.clip(
                guidance.climb_rate_command_mps,
                -self.gains.altitude_reference_rate_limit_mps,
                self.gains.altitude_reference_rate_limit_mps,
            )
        )
        throttle = self.gains.climb_rate_throttle_feedforward_per_mps * climb_rate
        ratio = float(np.clip(climb_rate / max(airspeed_reference_mps, 1.0), -0.95, 0.95))
        pitch = self.gains.flight_path_angle_feedforward_gain * float(np.arcsin(ratio))
        return throttle, pitch


__all__ = [
    "LongitudinalControlMode",
    "TotalEnergyControlGains",
    "TotalEnergyControlOutput",
    "TotalEnergyController",
]
