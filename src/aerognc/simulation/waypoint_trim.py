"""Validated straight-flight trim resolution for waypoint simulation backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from aerognc.configuration.aircraft_loader import (
    AircraftInitialCondition,
    AircraftSandboxConfiguration,
)
from aerognc.gnc.fixedwing_autopilot import AutopilotTrim
from aerognc.gnc.flight_analysis import TrimResult, solve_trim
from aerognc.simulation.waypoint_backends import ReducedFixedWingParams
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    FixedWingFlightModel,
    aircraft_initial_state,
    local_ned_dcm_inertial,
    longitudinal_trim_elevator_rad,
)


class TrimFailurePolicy(StrEnum):
    """Runtime action when a requested nonlinear trim does not converge."""

    REJECT = "reject"
    FALLBACK_CONFIGURED = "fallback_configured"


class TrimConvergenceError(RuntimeError):
    """Raised when strict mission initialization cannot establish trim."""


@dataclass(frozen=True, slots=True)
class WaypointTrimOptions:
    """Numerical trim policy and bounded coefficient-aircraft search domain."""

    enabled: bool = False
    failure_policy: TrimFailurePolicy = TrimFailurePolicy.REJECT
    tolerance: float = 1.0e-8
    maximum_iterations: int = 60
    minimum_angle_of_attack_rad: float = float(np.deg2rad(-8.0))
    maximum_angle_of_attack_rad: float = float(np.deg2rad(12.0))

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("trim enabled flag must be boolean")
        values = np.asarray(
            [
                self.tolerance,
                self.minimum_angle_of_attack_rad,
                self.maximum_angle_of_attack_rad,
            ]
        )
        if not np.all(np.isfinite(values)) or self.tolerance <= 0.0:
            raise ValueError("trim tolerance and angle-of-attack bounds must be finite")
        if self.minimum_angle_of_attack_rad >= self.maximum_angle_of_attack_rad:
            raise ValueError("trim angle-of-attack bounds must be increasing")
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, int)
            or self.maximum_iterations <= 0
        ):
            raise ValueError("trim maximum_iterations must be a positive integer")
        if not isinstance(self.failure_policy, TrimFailurePolicy):
            raise ValueError("trim failure_policy must be a TrimFailurePolicy")


@dataclass(frozen=True, slots=True)
class WaypointTrimResult:
    """Resolved initialization/feedforward commands and numerical evidence."""

    backend: str
    source: str
    converged: bool
    used_fallback: bool
    angle_of_attack_rad: float
    pitch_rad: float
    elevator_deflection_rad: float
    elevator_command: float
    throttle: float
    residual: tuple[float, float, float]
    residual_infinity_norm: float
    iterations: int

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.angle_of_attack_rad,
                self.pitch_rad,
                self.elevator_deflection_rad,
                self.elevator_command,
                self.throttle,
                self.residual_infinity_norm,
                *self.residual,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("waypoint trim result must be finite")
        if not -1.0 <= self.elevator_command <= 1.0 or not 0.0 <= self.throttle <= 1.0:
            raise ValueError("waypoint trim commands lie outside normalized bounds")
        if self.iterations < 0:
            raise ValueError("waypoint trim iteration count must be nonnegative")

    @property
    def autopilot_trim(self) -> AutopilotTrim:
        """Return the feedforward contract consumed by the autopilot."""
        return AutopilotTrim(self.pitch_rad, self.elevator_command, self.throttle)

    def summary(self) -> dict[str, object]:
        """Return JSON-compatible trim provenance and convergence evidence."""
        return {
            "backend": self.backend,
            "source": self.source,
            "converged": self.converged,
            "used_fallback": self.used_fallback,
            "angle_of_attack_rad": self.angle_of_attack_rad,
            "pitch_rad": self.pitch_rad,
            "elevator_deflection_rad": self.elevator_deflection_rad,
            "elevator_command": self.elevator_command,
            "throttle": self.throttle,
            "residual": list(self.residual),
            "residual_infinity_norm": self.residual_infinity_norm,
            "iterations": self.iterations,
        }


def solve_reduced_waypoint_trim(
    parameters: ReducedFixedWingParams,
    *,
    airspeed_mps: float,
) -> WaypointTrimResult:
    """Resolve the reduced plant's analytic level-flight throttle equilibrium."""
    if not np.isfinite(airspeed_mps) or airspeed_mps <= 0.0:
        raise ValueError("trim airspeed_mps must be positive and finite")
    speed_range = (
        parameters.airspeed_at_full_throttle_mps - parameters.airspeed_at_zero_throttle_mps
    )
    unconstrained = (airspeed_mps - parameters.airspeed_at_zero_throttle_mps) / speed_range
    throttle = float(np.clip(unconstrained, 0.0, 1.0))
    achieved = parameters.airspeed_at_zero_throttle_mps + throttle * speed_range
    residual = (achieved - airspeed_mps, 0.0, 0.0)
    norm = float(np.max(np.abs(residual)))
    return WaypointTrimResult(
        backend="internal_reduced",
        source="analytic_reduced_equilibrium",
        converged=norm <= 1.0e-12,
        used_fallback=False,
        angle_of_attack_rad=0.0,
        pitch_rad=0.0,
        elevator_deflection_rad=0.0,
        elevator_command=0.0,
        throttle=throttle,
        residual=residual,
        residual_infinity_norm=norm,
        iterations=0,
    )


def solve_coefficient_waypoint_trim(
    configuration: AircraftSandboxConfiguration,
    *,
    altitude_m: float,
    airspeed_mps: float,
    heading_rad: float,
    steady_wind_ned_mps: tuple[float, float, float],
    elevator_command_limit_rad: float,
    options: WaypointTrimOptions,
) -> WaypointTrimResult:
    """Solve alpha/elevator/throttle for straight level coefficient-plant flight."""
    scalars = np.asarray(
        [altitude_m, airspeed_mps, heading_rad, elevator_command_limit_rad],
        dtype=np.float64,
    )
    wind = np.asarray(steady_wind_ned_mps, dtype=np.float64)
    if (
        not np.all(np.isfinite(scalars))
        or altitude_m < 0.0
        or airspeed_mps <= 0.0
        or elevator_command_limit_rad <= 0.0
    ):
        raise ValueError("coefficient trim condition is outside its finite positive domain")
    if wind.shape != (3,) or not np.all(np.isfinite(wind)):
        raise ValueError("coefficient trim wind must contain three finite NED values")
    physical_elevator_limit = min(
        elevator_command_limit_rad,
        configuration.geometry.elevator_limit_rad,
    )
    base_initial = replace(
        configuration.initial,
        altitude_m=float(altitude_m),
        true_airspeed_mps=float(airspeed_mps),
        heading_rad=float(heading_rad),
        flight_path_angle_rad=0.0,
    )

    def residual(decision: np.ndarray) -> np.ndarray:
        alpha_rad, elevator_up_rad, throttle = decision
        runtime = replace(
            configuration,
            initial=replace(base_initial, angle_of_attack_rad=float(alpha_rad)),
            initial_throttle=float(throttle),
            wind_north_mps=float(wind[0]),
            wind_east_mps=float(wind[1]),
        )
        model = FixedWingFlightModel(
            runtime,
            wind_horizon_s=1.0,
            steady_wind_ned_mps=wind,
        )
        state = aircraft_initial_state(runtime)
        if wind[2] != 0.0:
            state[3:6] += local_ned_dcm_inertial(state[:3]) @ np.array(
                [0.0, 0.0, wind[2]], dtype=np.float64
            )
        state[15] = -elevator_up_rad
        state[17] = throttle
        command = AircraftControlCommand(
            pitch=float(elevator_up_rad / configuration.geometry.elevator_limit_rad),
            throttle=float(throttle),
        )
        derivative = model.derivative(0.0, state, command)
        acceleration_ned = local_ned_dcm_inertial(state[:3]).T @ derivative[3:6]
        return np.asarray(
            [acceleration_ned[0], acceleration_ned[2], derivative[11]],
            dtype=np.float64,
        )

    configured = _configured_coefficient_result(
        configuration,
        base_initial=base_initial,
        elevator_command_limit_rad=elevator_command_limit_rad,
        residual_function=residual,
    )
    if not options.enabled:
        return configured
    initial_decision = np.asarray(
        [
            configured.angle_of_attack_rad,
            configured.elevator_deflection_rad,
            configured.throttle,
        ],
        dtype=np.float64,
    )
    try:
        numerical = solve_trim(
            residual,
            initial_decision,
            lower_bounds=np.asarray(
                [options.minimum_angle_of_attack_rad, -physical_elevator_limit, 0.0]
            ),
            upper_bounds=np.asarray(
                [options.maximum_angle_of_attack_rad, physical_elevator_limit, 1.0]
            ),
            tolerance=options.tolerance,
            maximum_iterations=options.maximum_iterations,
        )
        resolved = _numerical_coefficient_result(
            numerical,
            base_initial=base_initial,
            elevator_command_limit_rad=elevator_command_limit_rad,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        return _handle_trim_failure(configured, options, reason=str(error))
    if not resolved.converged:
        return _handle_trim_failure(
            configured,
            options,
            reason=(
                f"residual infinity norm {resolved.residual_infinity_norm:.6g} after "
                f"{resolved.iterations} iterations"
            ),
        )
    return resolved


def configuration_with_resolved_trim(
    configuration: AircraftSandboxConfiguration,
    result: WaypointTrimResult,
) -> AircraftSandboxConfiguration:
    """Return a coefficient configuration initialized at the resolved trim state."""
    return replace(
        configuration,
        initial=replace(
            configuration.initial,
            angle_of_attack_rad=result.angle_of_attack_rad,
            flight_path_angle_rad=result.pitch_rad - result.angle_of_attack_rad,
        ),
        initial_throttle=result.throttle,
    )


def _configured_coefficient_result(
    configuration: AircraftSandboxConfiguration,
    *,
    base_initial: AircraftInitialCondition,
    elevator_command_limit_rad: float,
    residual_function: Callable[[np.ndarray], np.ndarray],
) -> WaypointTrimResult:
    alpha = base_initial.angle_of_attack_rad
    flight_path = base_initial.flight_path_angle_rad
    elevator_up = -longitudinal_trim_elevator_rad(alpha, configuration)
    throttle = configuration.initial_throttle
    residual = np.asarray(
        residual_function(np.asarray([alpha, elevator_up, throttle], dtype=np.float64)),
        dtype=np.float64,
    )
    residual_tuple = _residual_tuple(residual)
    return WaypointTrimResult(
        backend="internal_coefficient",
        source="configured_static_pitch_trim",
        converged=False,
        used_fallback=False,
        angle_of_attack_rad=alpha,
        pitch_rad=alpha + flight_path,
        elevator_deflection_rad=elevator_up,
        elevator_command=float(elevator_up / elevator_command_limit_rad),
        throttle=throttle,
        residual=residual_tuple,
        residual_infinity_norm=float(np.max(np.abs(residual))),
        iterations=0,
    )


def _numerical_coefficient_result(
    numerical: TrimResult,
    *,
    base_initial: AircraftInitialCondition,
    elevator_command_limit_rad: float,
) -> WaypointTrimResult:
    alpha, elevator_up, throttle = (float(item) for item in numerical.decision)
    residual = _residual_tuple(numerical.residual)
    return WaypointTrimResult(
        backend="internal_coefficient",
        source="nonlinear_force_moment_equilibrium",
        converged=numerical.converged,
        used_fallback=False,
        angle_of_attack_rad=alpha,
        pitch_rad=alpha + base_initial.flight_path_angle_rad,
        elevator_deflection_rad=elevator_up,
        elevator_command=float(elevator_up / elevator_command_limit_rad),
        throttle=throttle,
        residual=residual,
        residual_infinity_norm=float(np.max(np.abs(numerical.residual))),
        iterations=numerical.iterations,
    )


def _handle_trim_failure(
    configured: WaypointTrimResult,
    options: WaypointTrimOptions,
    *,
    reason: str,
) -> WaypointTrimResult:
    if options.failure_policy is TrimFailurePolicy.REJECT:
        raise TrimConvergenceError(f"coefficient waypoint trim failed: {reason}")
    return replace(configured, used_fallback=True, source="configured_fallback_after_failure")


def _residual_tuple(values: np.ndarray) -> tuple[float, float, float]:
    if values.shape != (3,):  # pragma: no cover - solver contract invariant
        raise ValueError("waypoint trim residual must contain three values")
    return float(values[0]), float(values[1]), float(values[2])


__all__ = [
    "TrimConvergenceError",
    "TrimFailurePolicy",
    "WaypointTrimOptions",
    "WaypointTrimResult",
    "configuration_with_resolved_trim",
    "solve_coefficient_waypoint_trim",
    "solve_reduced_waypoint_trim",
]
