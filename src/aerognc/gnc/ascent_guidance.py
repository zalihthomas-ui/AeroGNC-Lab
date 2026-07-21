"""Public-safe reference shaping and constraint governance for research ascent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.configuration.ascent_guidance_loader import AscentGuidanceConfiguration
from aerognc.mathematics.vectors import FloatArray

STANDARD_GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True, slots=True)
class AscentGuidanceDecision:
    """Two transparent offline reference-shaping parameters."""

    terminal_elevation_offset_rad: float
    throttle_scale: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.terminal_elevation_offset_rad, self.throttle_scale])):
            raise ValueError("ascent-guidance decision must be finite")
        if not 0.0 <= self.throttle_scale <= 1.0:
            raise ValueError("throttle_scale must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class AscentGuidanceInputs:
    """Measured/estimated quantities available to the online governor."""

    time_s: float
    dynamic_pressure_pa: float
    air_flight_path_angle_rad: float
    mass_kg: float
    nominal_thrust_n: float
    aerodynamic_force_magnitude_n: float
    predicted_ballistic_apogee_m: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.time_s,
                self.dynamic_pressure_pa,
                self.air_flight_path_angle_rad,
                self.mass_kg,
                self.nominal_thrust_n,
                self.aerodynamic_force_magnitude_n,
                self.predicted_ballistic_apogee_m,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("ascent-guidance inputs must be finite")
        if (
            self.time_s < 0.0
            or min(
                self.dynamic_pressure_pa,
                self.mass_kg,
                self.nominal_thrust_n,
                self.aerodynamic_force_magnitude_n,
            )
            < 0.0
        ):
            raise ValueError(
                "time, pressure, mass, thrust, and aerodynamic force cannot be negative"
            )
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")


@dataclass(frozen=True, slots=True)
class AscentGuidanceCommand:
    """Commanded pitch-plane elevation and throttle plus limiter diagnostics."""

    elevation_rad: float
    throttle: float
    reference_elevation_rad: float
    reference_throttle: float
    angle_of_attack_limited: bool
    dynamic_pressure_limited: bool
    proper_load_limited: bool
    apogee_limited: bool


class ConstraintAwareAscentGuidance:
    """Time schedule followed by explicit, independently testable constraint limiters.

    The algorithm has no destination/vehicle state input. It only shapes a fictional
    civilian research-rocket ascent against local structural/aerodynamic limits and a
    scalar apogee performance requirement.
    """

    def __init__(self, configuration: AscentGuidanceConfiguration) -> None:
        self.configuration = configuration
        self._time = np.asarray(configuration.reference_time_s, dtype=np.float64)
        self._elevation = np.asarray(configuration.reference_elevation_rad, dtype=np.float64)
        self._throttle = np.asarray(configuration.reference_throttle, dtype=np.float64)

    def reference(self, time_s: float, decision: AscentGuidanceDecision) -> tuple[float, float]:
        """Return the shaped open-loop reference before constraint governance."""
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        elevation = float(np.interp(time_s, self._time, self._elevation))
        throttle = float(np.interp(time_s, self._time, self._throttle))
        shaping_fraction = float(np.clip(time_s / self._time[-1], 0.0, 1.0))
        elevation += shaping_fraction * decision.terminal_elevation_offset_rad
        throttle *= decision.throttle_scale
        return elevation, float(np.clip(throttle, 0.0, 1.0))

    def command(
        self,
        inputs: AscentGuidanceInputs,
        decision: AscentGuidanceDecision,
    ) -> AscentGuidanceCommand:
        """Apply angle, max-Q, load, and predicted-apogee limiters in sequence."""
        configuration = self.configuration
        reference_elevation, reference_throttle = self.reference(inputs.time_s, decision)
        lower_elevation = (
            inputs.air_flight_path_angle_rad - configuration.maximum_angle_of_attack_rad
        )
        upper_elevation = (
            inputs.air_flight_path_angle_rad + configuration.maximum_angle_of_attack_rad
        )
        elevation = float(np.clip(reference_elevation, lower_elevation, upper_elevation))
        alpha_limited = not np.isclose(elevation, reference_elevation)

        q_soft = (
            configuration.dynamic_pressure_soft_fraction * configuration.maximum_dynamic_pressure_pa
        )
        q_span = configuration.maximum_dynamic_pressure_pa - q_soft
        q_scale = float(
            np.clip(
                (configuration.maximum_dynamic_pressure_pa - inputs.dynamic_pressure_pa) / q_span,
                0.0,
                1.0,
            )
        )

        if inputs.nominal_thrust_n > 0.0:
            proper_force_budget_n = max(
                0.0,
                configuration.maximum_proper_load_factor * STANDARD_GRAVITY_MPS2 * inputs.mass_kg
                - inputs.aerodynamic_force_magnitude_n,
            )
            load_scale = float(np.clip(proper_force_budget_n / inputs.nominal_thrust_n, 0.0, 1.0))
        else:
            load_scale = 0.0

        tolerance = configuration.apogee_tolerance_m
        effective_ballistic_target_m = (
            configuration.desired_apogee_m + configuration.ballistic_apogee_reserve_m
        )
        apogee_scale = float(
            np.clip(
                (effective_ballistic_target_m + tolerance - inputs.predicted_ballistic_apogee_m)
                / (2.0 * tolerance),
                0.0,
                1.0,
            )
        )
        if inputs.predicted_ballistic_apogee_m < effective_ballistic_target_m - tolerance:
            apogee_scale = 1.0

        throttle = min(reference_throttle, q_scale, load_scale, apogee_scale)
        motor_active = inputs.nominal_thrust_n > 0.0
        return AscentGuidanceCommand(
            elevation_rad=elevation,
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            reference_elevation_rad=reference_elevation,
            reference_throttle=reference_throttle,
            angle_of_attack_limited=alpha_limited,
            dynamic_pressure_limited=(motor_active and q_scale < reference_throttle - 1.0e-12),
            proper_load_limited=(motor_active and load_scale < reference_throttle - 1.0e-12),
            apogee_limited=(motor_active and apogee_scale < reference_throttle - 1.0e-12),
        )


def decision_vector(decision: AscentGuidanceDecision) -> FloatArray:
    """Return a stable numeric representation for optimization/reporting."""
    return np.array(
        [decision.terminal_elevation_offset_rad, decision.throttle_scale],
        dtype=np.float64,
    )


def decision_from_vector(value: npt.ArrayLike) -> AscentGuidanceDecision:
    """Construct a validated decision from a two-element array."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("ascent-guidance decision vector must contain two finite values")
    return AscentGuidanceDecision(float(vector[0]), float(vector[1]))
