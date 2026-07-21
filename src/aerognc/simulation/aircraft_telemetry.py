"""Shared SI telemetry calculations for batch, live, recorder, and replay tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import as_vector
from aerognc.vehicle.fixed_wing import (
    STANDARD_GRAVITY_MPS2,
    AircraftControlCommand,
    AircraftState,
    FixedWingFlightModel,
    aircraft_stall_speed_mps,
    initial_tangent_displacement_ned_m,
    local_ned_dcm_inertial,
)


@dataclass(frozen=True, slots=True)
class AircraftTelemetry:
    """One named, unit-explicit view of the nonlinear aircraft state and loads."""

    time_s: float
    north_m: float
    east_m: float
    down_m: float
    altitude_m: float
    ground_speed_mps: float
    vertical_speed_mps: float
    true_airspeed_mps: float
    mach: float
    dynamic_pressure_pa: float
    angle_of_attack_deg: float
    sideslip_angle_deg: float
    stall_fraction: float
    stall_speed_1g_mps: float
    stall_margin_mps: float
    lift_coefficient: float
    drag_coefficient: float
    side_force_coefficient: float
    roll_moment_coefficient: float
    pitch_moment_coefficient: float
    yaw_moment_coefficient: float
    roll_deg: float
    pitch_deg: float
    heading_deg: float
    flight_path_angle_deg: float
    roll_rate_degps: float
    pitch_rate_degps: float
    yaw_rate_degps: float
    normal_load_g: float
    specific_force_g: float
    lift_over_weight: float
    air_breathing_thrust_n: float
    rocket_thrust_n: float
    mass_kg: float
    fuel_fraction: float
    throttle: float

    @property
    def stalled(self) -> bool:
        """Return whether the synthetic post-stall model is active."""
        return self.stall_fraction > 0.0


def aircraft_telemetry(
    model: FixedWingFlightModel,
    time_s: float,
    values: npt.ArrayLike,
    command: AircraftControlCommand,
    *,
    initial_position_inertial_m: npt.ArrayLike | None = None,
) -> AircraftTelemetry:
    """Calculate one consistent telemetry sample from a state and held command."""
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError("aircraft telemetry time must be finite and nonnegative")
    state_values = as_vector(values, 18, name="aircraft_state")
    state = AircraftState.from_array(state_values, normalize=True)
    configuration = model.configuration
    loads = model.loads(time_s, state_values, command)
    initial_position = (
        state.position_inertial_m
        if initial_position_inertial_m is None
        else as_vector(initial_position_inertial_m, 3, name="initial_position_inertial_m")
    )
    displacement_ned = initial_tangent_displacement_ned_m(
        state.position_inertial_m,
        time_s,
        initial_position,
        configuration.planet.rotation_rate_radps,
    )
    planet_rate = np.array([0.0, 0.0, configuration.planet.rotation_rate_radps])
    ground_velocity_inertial = state.velocity_inertial_mps - np.cross(
        planet_rate, state.position_inertial_m
    )
    local_dcm = local_ned_dcm_inertial(state.position_inertial_m)
    ground_velocity_ned = local_dcm.T @ ground_velocity_inertial
    air_velocity_ned = local_dcm.T @ loads.air_velocity_inertial_mps
    horizontal_airspeed = float(np.linalg.norm(air_velocity_ned[:2]))
    roll, pitch, heading = model.local_attitude_rad(state_values)
    stall_speed = aircraft_stall_speed_mps(
        state.mass_kg,
        max(loads.density_kgpm3, 1.0e-30),
        configuration,
    )
    aerodynamic = loads.aerodynamic
    lift_n = (
        aerodynamic.dynamic_pressure_pa
        * configuration.geometry.wing_area_m2
        * aerodynamic.lift_coefficient
    )
    specific_force_body_mps2 = loads.total_force_body_n / state.mass_kg
    available_fuel_kg = configuration.mass.initial_mass_kg - configuration.mass.dry_mass_kg
    fuel_fraction = (state.mass_kg - configuration.mass.dry_mass_kg) / available_fuel_kg
    return AircraftTelemetry(
        time_s=float(time_s),
        north_m=float(displacement_ned[0]),
        east_m=float(displacement_ned[1]),
        down_m=float(displacement_ned[2]),
        altitude_m=float(loads.altitude_m),
        ground_speed_mps=float(np.linalg.norm(ground_velocity_inertial)),
        vertical_speed_mps=float(-ground_velocity_ned[2]),
        true_airspeed_mps=aerodynamic.true_airspeed_mps,
        mach=aerodynamic.mach,
        dynamic_pressure_pa=aerodynamic.dynamic_pressure_pa,
        angle_of_attack_deg=float(np.rad2deg(aerodynamic.angle_of_attack_rad)),
        sideslip_angle_deg=float(np.rad2deg(aerodynamic.sideslip_angle_rad)),
        stall_fraction=aerodynamic.stall_fraction,
        stall_speed_1g_mps=stall_speed,
        stall_margin_mps=aerodynamic.true_airspeed_mps - stall_speed,
        lift_coefficient=aerodynamic.lift_coefficient,
        drag_coefficient=aerodynamic.drag_coefficient,
        side_force_coefficient=aerodynamic.side_force_coefficient,
        roll_moment_coefficient=aerodynamic.roll_moment_coefficient,
        pitch_moment_coefficient=aerodynamic.pitch_moment_coefficient,
        yaw_moment_coefficient=aerodynamic.yaw_moment_coefficient,
        roll_deg=float(np.rad2deg(roll)),
        pitch_deg=float(np.rad2deg(pitch)),
        heading_deg=float(np.mod(np.rad2deg(heading), 360.0)),
        flight_path_angle_deg=float(
            np.rad2deg(np.arctan2(-air_velocity_ned[2], max(horizontal_airspeed, 1.0e-12)))
        ),
        roll_rate_degps=float(np.rad2deg(state.angular_rate_body_radps[0])),
        pitch_rate_degps=float(np.rad2deg(state.angular_rate_body_radps[1])),
        yaw_rate_degps=float(np.rad2deg(state.angular_rate_body_radps[2])),
        normal_load_g=float(-specific_force_body_mps2[2] / STANDARD_GRAVITY_MPS2),
        specific_force_g=float(np.linalg.norm(specific_force_body_mps2) / STANDARD_GRAVITY_MPS2),
        lift_over_weight=float(lift_n / (state.mass_kg * STANDARD_GRAVITY_MPS2)),
        air_breathing_thrust_n=loads.thrust_air_breathing_n,
        rocket_thrust_n=loads.thrust_rocket_n,
        mass_kg=state.mass_kg,
        fuel_fraction=float(np.clip(fuel_fraction, 0.0, 1.0)),
        throttle=state.throttle,
    )
