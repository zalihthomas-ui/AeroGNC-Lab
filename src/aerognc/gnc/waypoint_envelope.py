"""Estimated-state envelope margins for waypoint-control telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.environment.atmosphere import StandardAtmosphere1976
from aerognc.navigation.state import NavigationState
from aerognc.vehicle.control_surfaces import SurfaceDeflections
from aerognc.vehicle.fixed_wing import aircraft_stall_speed_mps


@dataclass(frozen=True, slots=True)
class WaypointEnvelopeReference:
    """Declared safety/actuator bounds and optional coefficient stall model."""

    minimum_altitude_m: float
    maximum_altitude_m: float
    minimum_airspeed_mps: float
    maximum_airspeed_mps: float
    maximum_bank_rad: float
    maximum_pitch_rad: float
    aileron_limit_rad: float
    elevator_limit_rad: float
    rudder_limit_rad: float
    gravity_mps2: float = 9.80665
    coefficient_configuration: AircraftSandboxConfiguration | None = None

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.minimum_altitude_m,
                self.maximum_altitude_m,
                self.minimum_airspeed_mps,
                self.maximum_airspeed_mps,
                self.maximum_bank_rad,
                self.maximum_pitch_rad,
                self.aileron_limit_rad,
                self.elevator_limit_rad,
                self.rudder_limit_rad,
                self.gravity_mps2,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("waypoint envelope reference values must be finite")
        if self.minimum_altitude_m >= self.maximum_altitude_m:
            raise ValueError("waypoint altitude envelope must be increasing")
        if self.minimum_airspeed_mps >= self.maximum_airspeed_mps:
            raise ValueError("waypoint airspeed envelope must be increasing")
        if np.any(values[2:] <= 0.0):
            raise ValueError(
                "waypoint speed, attitude, actuator, and gravity limits must be positive"
            )


@dataclass(frozen=True, slots=True)
class WaypointEnvelopeMargins:
    """Per-step stall, loading, attitude, actuator, and energy margins."""

    stall_reference_source: str
    stall_speed_reference_mps: float
    stall_margin_mps: float
    load_factor: float
    bank_margin_rad: float
    pitch_margin_rad: float
    aileron_margin_fraction: float
    elevator_margin_fraction: float
    rudder_margin_fraction: float
    minimum_surface_margin_fraction: float
    throttle_margin_fraction: float
    lower_specific_energy_margin_m2ps2: float
    upper_specific_energy_margin_m2ps2: float


def evaluate_waypoint_envelope(
    state: NavigationState,
    deflections: SurfaceDeflections,
    reference: WaypointEnvelopeReference,
) -> WaypointEnvelopeMargins:
    """Evaluate margins using only controller-facing state and declared references."""
    load_factor = float(1.0 / max(np.cos(abs(state.roll_rad)), 0.1))
    if reference.coefficient_configuration is None:
        stall_speed = reference.minimum_airspeed_mps
        stall_source = "declared_reduced_model_minimum_airspeed"
    else:
        atmosphere = StandardAtmosphere1976().properties(
            float(np.clip(state.altitude_m, -500.0, 47_000.0))
        )
        stall_speed = aircraft_stall_speed_mps(
            reference.coefficient_configuration.mass.initial_mass_kg,
            atmosphere.density_kgpm3,
            reference.coefficient_configuration,
            load_factor=load_factor,
        )
        stall_source = "coefficient_cl_max_at_estimated_altitude_and_load"

    surface_margins = (
        _surface_margin(deflections.aileron_rad, reference.aileron_limit_rad),
        _surface_margin(deflections.elevator_rad, reference.elevator_limit_rad),
        _surface_margin(deflections.rudder_rad, reference.rudder_limit_rad),
    )
    lower_energy = reference.gravity_mps2 * (
        state.altitude_m - reference.minimum_altitude_m
    ) + 0.5 * (state.airspeed_mps**2 - stall_speed**2)
    upper_energy = reference.gravity_mps2 * (
        reference.maximum_altitude_m - state.altitude_m
    ) + 0.5 * (reference.maximum_airspeed_mps**2 - state.airspeed_mps**2)
    return WaypointEnvelopeMargins(
        stall_reference_source=stall_source,
        stall_speed_reference_mps=float(stall_speed),
        stall_margin_mps=float(state.airspeed_mps - stall_speed),
        load_factor=load_factor,
        bank_margin_rad=float(reference.maximum_bank_rad - abs(state.roll_rad)),
        pitch_margin_rad=float(reference.maximum_pitch_rad - abs(state.pitch_rad)),
        aileron_margin_fraction=surface_margins[0],
        elevator_margin_fraction=surface_margins[1],
        rudder_margin_fraction=surface_margins[2],
        minimum_surface_margin_fraction=float(min(surface_margins)),
        throttle_margin_fraction=float(min(deflections.throttle, 1.0 - deflections.throttle)),
        lower_specific_energy_margin_m2ps2=float(lower_energy),
        upper_specific_energy_margin_m2ps2=float(upper_energy),
    )


def _surface_margin(deflection_rad: float, limit_rad: float) -> float:
    return float(1.0 - abs(deflection_rad) / limit_rad)


__all__ = [
    "WaypointEnvelopeMargins",
    "WaypointEnvelopeReference",
    "evaluate_waypoint_envelope",
]
