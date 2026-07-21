"""Coefficient-driven nonlinear model for a fictional civilian research aircraft.

The model uses planet-centred inertial position/velocity, Hamilton scalar-first
quaternions mapping forward-right-down body components into the inertial frame, and
SI units throughout. Its synthetic coefficients are plausible educational inputs,
not type-certified flight data for any real aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.astrodynamics.perturbations import j2_acceleration_mps2
from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.environment.orbital_atmosphere import ReferenceOrbitalAtmosphere
from aerognc.environment.wind import OneCosineGust, WindModel, WindProfile
from aerognc.mathematics.geodesy import dcm_inertial_to_ecef
from aerognc.mathematics.quaternion import (
    dcm_to_quaternion,
    euler321_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler321,
)
from aerognc.mathematics.vectors import FloatArray, as_vector
from aerognc.vehicle.aero_database import AerodynamicCondition, TabulatedAerodynamicDatabase

STANDARD_GRAVITY_MPS2 = 9.80665
SEA_LEVEL_DENSITY_KGPM3 = 1.225
AIRCRAFT_STATE_SIZE = 18


@dataclass(frozen=True, slots=True)
class AircraftControlCommand:
    """Pilot command: normalized axes, throttle fraction, and rocket-assist switch."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.5
    rocket_assist: bool = False

    def __post_init__(self) -> None:
        values = np.asarray([self.roll, self.pitch, self.yaw, self.throttle])
        if not np.all(np.isfinite(values)):
            raise ValueError("aircraft control commands must be finite")
        if np.any(np.abs(values[:3]) > 1.0) or not 0.0 <= self.throttle <= 1.0:
            raise ValueError("aircraft axes must lie in [-1, 1] and throttle in [0, 1]")


@dataclass(frozen=True, slots=True)
class AircraftState:
    """Typed view of the 18-state fixed-wing plant."""

    position_inertial_m: FloatArray
    velocity_inertial_mps: FloatArray
    quaternion_ib: FloatArray
    angular_rate_body_radps: FloatArray
    mass_kg: float
    control_surface_rad: FloatArray
    throttle: float

    @classmethod
    def from_array(cls, values: npt.ArrayLike, *, normalize: bool = False) -> AircraftState:
        """Parse ``[r_i(3),v_i(3),q_ib(4),omega_b(3),m,da,de,dr,throttle]``."""
        state = as_vector(values, AIRCRAFT_STATE_SIZE, name="aircraft_state")
        quaternion = normalize_quaternion(state[6:10]) if normalize else state[6:10]
        return cls(
            state[:3],
            state[3:6],
            quaternion,
            state[10:13],
            float(state[13]),
            state[14:17],
            float(state[17]),
        )

    def as_array(self) -> FloatArray:
        """Return a new state vector in documented ordering."""
        return np.concatenate(
            (
                self.position_inertial_m,
                self.velocity_inertial_mps,
                self.quaternion_ib,
                self.angular_rate_body_radps,
                [self.mass_kg],
                self.control_surface_rad,
                [self.throttle],
            )
        )


@dataclass(frozen=True, slots=True)
class AircraftAerodynamicState:
    """Aerodynamic angles, coefficients, forces, and moments at one instant."""

    true_airspeed_mps: float
    mach: float
    dynamic_pressure_pa: float
    angle_of_attack_rad: float
    sideslip_angle_rad: float
    stall_fraction: float
    lift_coefficient: float
    drag_coefficient: float
    side_force_coefficient: float
    roll_moment_coefficient: float
    pitch_moment_coefficient: float
    yaw_moment_coefficient: float
    force_body_n: FloatArray
    moment_body_nm: FloatArray

    @property
    def stalled(self) -> bool:
        """Return whether angle of attack exceeds the configured onset boundary."""
        return self.stall_fraction > 0.0


@dataclass(frozen=True, slots=True)
class AircraftLoads:
    """Composed forces, moments, environment values, and mass flow."""

    altitude_m: float
    latitude_rad: float
    longitude_ground_rad: float
    air_velocity_inertial_mps: FloatArray
    air_velocity_body_mps: FloatArray
    aerodynamic: AircraftAerodynamicState
    thrust_air_breathing_n: float
    thrust_rocket_n: float
    total_force_body_n: FloatArray
    total_moment_body_nm: FloatArray
    gravity_inertial_mps2: FloatArray
    mass_flow_kgps: float
    density_kgpm3: float
    speed_of_sound_mps: float


def local_ned_dcm_inertial(position_inertial_m: npt.ArrayLike) -> FloatArray:
    """Return ``C_in`` whose columns are local north, east, and down in inertial axes."""
    position = as_vector(position_inertial_m, 3, name="position_inertial_m")
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("position radius must be positive")
    up = position / radius
    spin_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(spin_axis, up)
    east_norm = float(np.linalg.norm(east))
    if east_norm < 1.0e-10:
        east = np.array([0.0, 1.0, 0.0])
    else:
        east /= east_norm
    down = -up
    north = np.cross(east, down)
    north /= np.linalg.norm(north)
    return np.column_stack((north, east, down))


def initial_tangent_displacement_ned_m(
    position_inertial_m: npt.ArrayLike,
    time_s: float,
    initial_position_inertial_m: npt.ArrayLike,
    rotation_rate_radps: float,
) -> FloatArray:
    """Return current displacement in the initial planet-fixed local NED frame.

    Both Cartesian positions are inertial. The current position is first rotated into
    the planet-fixed frame at ``time_s`` so a point stationary on the rotating surface
    does not acquire a false eastward ground track.
    """
    if not np.all(np.isfinite([time_s, rotation_rate_radps])) or time_s < 0.0:
        raise ValueError("local-displacement time/rate must be finite and time nonnegative")
    current_inertial = as_vector(position_inertial_m, 3, name="position_inertial_m")
    initial_inertial = as_vector(
        initial_position_inertial_m, 3, name="initial_position_inertial_m"
    )
    current_fixed = dcm_inertial_to_ecef(rotation_rate_radps * time_s) @ current_inertial
    initial_fixed = initial_inertial
    return local_ned_dcm_inertial(initial_fixed).T @ (current_fixed - initial_fixed)


def spherical_position_m(radius_m: float, latitude_rad: float, longitude_rad: float) -> FloatArray:
    """Convert spherical latitude/longitude/radius to planet-centred components."""
    cosine_latitude = np.cos(latitude_rad)
    return radius_m * np.array(
        [
            cosine_latitude * np.cos(longitude_rad),
            cosine_latitude * np.sin(longitude_rad),
            np.sin(latitude_rad),
        ]
    )


def aircraft_initial_state(configuration: AircraftSandboxConfiguration) -> FloatArray:
    """Construct the inertial state from user-readable local flight conditions."""
    initial = configuration.initial
    planet = configuration.planet
    position = spherical_position_m(
        planet.radius_m + initial.altitude_m,
        initial.latitude_rad,
        initial.longitude_rad,
    )
    dcm_in = local_ned_dcm_inertial(position)
    pitch_rad = initial.flight_path_angle_rad + initial.angle_of_attack_rad
    dcm_nb = quaternion_to_dcm(
        euler321_to_quaternion(initial.bank_angle_rad, pitch_rad, initial.heading_rad)
    )
    dcm_ib = dcm_in @ dcm_nb
    quaternion_ib = dcm_to_quaternion(dcm_ib)
    horizontal_speed = initial.true_airspeed_mps * np.cos(initial.flight_path_angle_rad)
    air_velocity_ned = np.array(
        [
            horizontal_speed * np.cos(initial.heading_rad),
            horizontal_speed * np.sin(initial.heading_rad),
            -initial.true_airspeed_mps * np.sin(initial.flight_path_angle_rad),
        ]
    )
    wind_ned = np.array([configuration.wind_north_mps, configuration.wind_east_mps, 0.0])
    angular_velocity = np.array([0.0, 0.0, planet.rotation_rate_radps])
    velocity = np.cross(angular_velocity, position) + dcm_in @ (wind_ned + air_velocity_ned)
    trim_elevator = longitudinal_trim_elevator_rad(initial.angle_of_attack_rad, configuration)
    return AircraftState(
        position,
        velocity,
        quaternion_ib,
        np.zeros(3),
        configuration.mass.initial_mass_kg,
        np.array([0.0, trim_elevator, 0.0]),
        configuration.initial_throttle,
    ).as_array()


def longitudinal_trim_elevator_rad(
    angle_of_attack_rad: float, configuration: AircraftSandboxConfiguration
) -> float:
    """Return the bounded elevator deflection giving zero static pitching moment."""
    if not np.isfinite(angle_of_attack_rad):
        raise ValueError("angle_of_attack_rad must be finite")
    aerodynamics = configuration.aerodynamics
    if abs(aerodynamics.pitch_elevator_per_rad) <= 1.0e-12:
        raise ValueError("pitch_elevator_per_rad is too small to calculate trim")
    elevator = (
        -(aerodynamics.pitch_zero + aerodynamics.pitch_alpha_per_rad * angle_of_attack_rad)
        / aerodynamics.pitch_elevator_per_rad
    )
    return float(
        np.clip(
            elevator,
            -configuration.geometry.elevator_limit_rad,
            configuration.geometry.elevator_limit_rad,
        )
    )


def longitudinal_trim_command(
    configuration: AircraftSandboxConfiguration,
) -> AircraftControlCommand:
    """Return a held command matching the configured initial static pitch trim."""
    elevator = longitudinal_trim_elevator_rad(
        configuration.initial.angle_of_attack_rad, configuration
    )
    return AircraftControlCommand(
        pitch=-elevator / configuration.geometry.elevator_limit_rad,
        throttle=configuration.initial_throttle,
    )


def project_aircraft_state(
    values: npt.ArrayLike, configuration: AircraftSandboxConfiguration
) -> FloatArray:
    """Normalize attitude and enforce hard mass/actuator state bounds."""
    state = as_vector(values, AIRCRAFT_STATE_SIZE, name="aircraft_state")
    state[6:10] = normalize_quaternion(state[6:10])
    state[13] = max(configuration.mass.dry_mass_kg, state[13])
    limits = np.array(
        [
            configuration.geometry.aileron_limit_rad,
            configuration.geometry.elevator_limit_rad,
            configuration.geometry.rudder_limit_rad,
        ]
    )
    state[14:17] = np.clip(state[14:17], -limits, limits)
    state[17] = np.clip(state[17], 0.0, 1.0)
    return state


def _post_stall_lift(
    angle_of_attack_rad: float,
    linear_lift: float,
    configuration: AircraftSandboxConfiguration,
) -> tuple[float, float]:
    aerodynamics = configuration.aerodynamics
    absolute_alpha = abs(angle_of_attack_rad)
    if absolute_alpha <= aerodynamics.stall_angle_rad:
        return (
            float(np.clip(linear_lift, -aerodynamics.cl_maximum, aerodynamics.cl_maximum)),
            0.0,
        )
    stall_fraction = float(
        np.clip(
            (absolute_alpha - aerodynamics.stall_angle_rad) / np.deg2rad(10.0),
            0.0,
            1.0,
        )
    )
    decay_fraction = float(
        np.clip(
            (absolute_alpha - aerodynamics.stall_angle_rad)
            / (np.pi / 2.0 - aerodynamics.stall_angle_rad),
            0.0,
            1.0,
        )
    )
    sign = 1.0 if angle_of_attack_rad >= 0.0 else -1.0
    boundary_linear_lift = (
        aerodynamics.cl_zero
        + aerodynamics.cl_alpha_per_rad * sign * aerodynamics.stall_angle_rad
    )
    boundary_lift = float(
        np.clip(
            boundary_linear_lift,
            -aerodynamics.cl_maximum,
            aerodynamics.cl_maximum,
        )
    )
    magnitude = abs(boundary_lift) * (
        1.0 - (1.0 - aerodynamics.post_stall_lift_fraction) * decay_fraction
    )
    return sign * magnitude, stall_fraction


def aerodynamic_state(
    air_velocity_body_mps: npt.ArrayLike,
    angular_rate_body_radps: npt.ArrayLike,
    control_surface_rad: npt.ArrayLike,
    density_kgpm3: float,
    speed_of_sound_mps: float,
    configuration: AircraftSandboxConfiguration,
    coefficient_database: TabulatedAerodynamicDatabase | None = None,
) -> AircraftAerodynamicState:
    """Calculate nonlinear coefficient-driven forces/moments in FRD body axes."""
    velocity = as_vector(air_velocity_body_mps, 3, name="air_velocity_body_mps")
    rates = as_vector(angular_rate_body_radps, 3, name="angular_rate_body_radps")
    controls = as_vector(control_surface_rad, 3, name="control_surface_rad")
    if not np.isfinite(density_kgpm3) or density_kgpm3 < 0.0:
        raise ValueError("density_kgpm3 must be finite and nonnegative")
    if not np.isfinite(speed_of_sound_mps) or speed_of_sound_mps <= 0.0:
        raise ValueError("speed_of_sound_mps must be positive and finite")
    airspeed = float(np.linalg.norm(velocity))
    if airspeed <= 1.0e-8:
        alpha = 0.0
        beta = 0.0
    else:
        alpha = float(np.arctan2(velocity[2], velocity[0]))
        beta = float(np.arcsin(np.clip(velocity[1] / airspeed, -1.0, 1.0)))
    dynamic_pressure = 0.5 * density_kgpm3 * airspeed**2
    mach = airspeed / speed_of_sound_mps
    aero = configuration.aerodynamics
    geometry = configuration.geometry
    rate_scale_longitudinal = geometry.mean_chord_m / max(2.0 * airspeed, 1.0e-8)
    rate_scale_lateral = geometry.wingspan_m / max(2.0 * airspeed, 1.0e-8)
    aileron, elevator, rudder = controls
    if coefficient_database is None:
        static_lift = aero.cl_zero + aero.cl_alpha_per_rad * alpha
        lift_coefficient, stall_fraction = _post_stall_lift(alpha, static_lift, configuration)
        drag_coefficient = 0.0
        side_force_coefficient = aero.side_force_beta_per_rad * beta
        roll_coefficient = aero.roll_beta_per_rad * beta
        pitch_coefficient = aero.pitch_zero + aero.pitch_alpha_per_rad * alpha
        yaw_coefficient = aero.yaw_beta_per_rad * beta
    else:
        base = coefficient_database.evaluate(
            AerodynamicCondition(mach=mach, alpha_rad=alpha, beta_rad=beta)
        )
        lift_coefficient = -base.normal
        drag_coefficient = max(0.0, base.drag)
        side_force_coefficient = base.side
        roll_coefficient = base.roll
        pitch_coefficient = base.pitch
        yaw_coefficient = base.yaw
        stall_fraction = float(
            np.clip(
                (abs(alpha) - aero.stall_angle_rad) / np.deg2rad(10.0),
                0.0,
                1.0,
            )
        )
    lift_coefficient += (
        aero.cl_elevator_per_rad * elevator
        + aero.cl_pitch_rate * rates[1] * rate_scale_longitudinal
    )
    lift_coefficient = float(
        np.clip(lift_coefficient, -1.3 * aero.cl_maximum, 1.3 * aero.cl_maximum)
    )
    if coefficient_database is None:
        drag_coefficient = (
            aero.cd_zero
            + aero.induced_drag_factor * lift_coefficient**2
            + aero.stall_drag_increment * stall_fraction**2
        )
    drag_coefficient += 0.02 * (abs(aileron) + abs(elevator) + abs(rudder))
    side_force_coefficient += (
        aero.side_force_aileron_per_rad * aileron
        + aero.side_force_rudder_per_rad * rudder
    )
    roll_coefficient += (
        aero.roll_rate * rates[0] * rate_scale_lateral
        + aero.roll_yaw_rate * rates[2] * rate_scale_lateral
        + aero.roll_aileron_per_rad * aileron
        + aero.roll_rudder_per_rad * rudder
    )
    pitch_coefficient += (
        aero.pitch_rate * rates[1] * rate_scale_longitudinal
        + aero.pitch_elevator_per_rad * elevator
    )
    yaw_coefficient += (
        aero.yaw_roll_rate * rates[0] * rate_scale_lateral
        + aero.yaw_rate * rates[2] * rate_scale_lateral
        + aero.yaw_aileron_per_rad * aileron
        + aero.yaw_rudder_per_rad * rudder
    )
    cosine_alpha, sine_alpha = np.cos(alpha), np.sin(alpha)
    cosine_beta, sine_beta = np.cos(beta), np.sin(beta)
    dcm_body_wind = np.array(
        [
            [cosine_alpha * cosine_beta, -cosine_alpha * sine_beta, -sine_alpha],
            [sine_beta, cosine_beta, 0.0],
            [sine_alpha * cosine_beta, -sine_alpha * sine_beta, cosine_alpha],
        ]
    )
    force_wind = (
        dynamic_pressure
        * geometry.wing_area_m2
        * np.array([-drag_coefficient, side_force_coefficient, -lift_coefficient])
    )
    force_body = dcm_body_wind @ force_wind
    moment_body = (
        dynamic_pressure
        * geometry.wing_area_m2
        * np.array(
            [
                geometry.wingspan_m * roll_coefficient,
                geometry.mean_chord_m * pitch_coefficient,
                geometry.wingspan_m * yaw_coefficient,
            ]
        )
    )
    return AircraftAerodynamicState(
        airspeed,
        float(mach),
        float(dynamic_pressure),
        alpha,
        beta,
        stall_fraction,
        lift_coefficient,
        float(drag_coefficient),
        float(side_force_coefficient),
        float(roll_coefficient),
        float(pitch_coefficient),
        float(yaw_coefficient),
        force_body,
        moment_body,
    )


class FixedWingFlightModel:
    """Compose environment, coefficient aerodynamics, propulsion, and rigid-body EOM."""

    def __init__(
        self,
        configuration: AircraftSandboxConfiguration,
        *,
        wind_horizon_s: float | None = None,
        aerodynamic_database: TabulatedAerodynamicDatabase | None = None,
    ) -> None:
        if wind_horizon_s is not None and (
            not np.isfinite(wind_horizon_s) or wind_horizon_s <= 0.0
        ):
            raise ValueError("fixed-wing wind horizon must be positive and finite")
        self.configuration = configuration
        if aerodynamic_database is None and configuration.aerodynamic_backend == "table":
            if configuration.aerodynamic_table_path is None:  # pragma: no cover - config invariant
                raise ValueError("table aerodynamic backend has no configured table")
            aerodynamic_database = TabulatedAerodynamicDatabase.from_csv(
                configuration.aerodynamic_table_path,
                out_of_range="clamp",
            )
        if aerodynamic_database is not None and set(aerodynamic_database.axis_names) != {
            "mach",
            "alpha_rad",
            "beta_rad",
        }:
            raise ValueError(
                "fixed-wing static table axes must be mach, alpha_rad, and beta_rad"
            )
        self.aerodynamic_database = aerodynamic_database
        self.atmosphere = ReferenceOrbitalAtmosphere(configuration.planet.atmosphere_density_scale)
        self.wind_model = WindModel(
            WindProfile.constant(
                [configuration.wind_north_mps, configuration.wind_east_mps, 0.0]
            ),
            gust_std_ned_mps=configuration.turbulence_std_ned_mps,
            correlation_time_s=configuration.turbulence_correlation_time_s,
            sample_step_s=0.1,
            horizon_s=max(configuration.duration_s + 1.0, wind_horizon_s or 0.0),
            seed=configuration.wind_random_seed,
        )
        self.discrete_gust = OneCosineGust(
            configuration.gust_start_time_s,
            configuration.gust_duration_s,
            configuration.gust_amplitude_ned_mps,
        )

    def loads(
        self,
        time_s: float,
        values: npt.ArrayLike,
        command: AircraftControlCommand,
    ) -> AircraftLoads:
        """Evaluate all aircraft loads and supporting diagnostics."""
        state = AircraftState.from_array(values, normalize=True)
        configuration = self.configuration
        planet = configuration.planet
        radius = float(np.linalg.norm(state.position_inertial_m))
        if radius <= 0.0:
            raise FloatingPointError("aircraft reached zero planet-centred radius")
        altitude = radius - planet.radius_m
        latitude = float(np.arcsin(np.clip(state.position_inertial_m[2] / radius, -1.0, 1.0)))
        longitude_ground = float(
            np.arctan2(state.position_inertial_m[1], state.position_inertial_m[0])
            - planet.rotation_rate_radps * time_s
        )
        atmosphere = self.atmosphere.properties(max(-500.0, altitude))
        dcm_in = local_ned_dcm_inertial(state.position_inertial_m)
        wind_ned_mps = self.wind_model.velocity_ned_mps(
            time_s, max(-500.0, altitude)
        ) + self.discrete_gust.velocity_ned_mps(time_s)
        wind_inertial = dcm_in @ wind_ned_mps
        planet_rate = np.array([0.0, 0.0, planet.rotation_rate_radps])
        atmosphere_velocity = np.cross(planet_rate, state.position_inertial_m) + wind_inertial
        air_velocity_inertial = state.velocity_inertial_mps - atmosphere_velocity
        dcm_ib = quaternion_to_dcm(state.quaternion_ib)
        air_velocity_body = dcm_ib.T @ air_velocity_inertial
        aerodynamic = aerodynamic_state(
            air_velocity_body,
            state.angular_rate_body_radps,
            state.control_surface_rad,
            atmosphere.density_kgpm3,
            atmosphere.speed_of_sound_mps,
            configuration,
            self.aerodynamic_database,
        )
        propulsion = configuration.propulsion
        has_fuel = state.mass_kg > configuration.mass.dry_mass_kg + 1.0e-9
        density_ratio = max(0.0, atmosphere.density_kgpm3 / SEA_LEVEL_DENSITY_KGPM3)
        mach_margin = np.clip(
            (propulsion.maximum_operating_mach - aerodynamic.mach) / 0.20, 0.0, 1.0
        )
        altitude_margin = np.clip(
            (propulsion.maximum_operating_altitude_m - altitude) / 2_000.0, 0.0, 1.0
        )
        air_thrust = (
            propulsion.maximum_thrust_n
            * state.throttle
            * density_ratio**propulsion.thrust_density_exponent
            * mach_margin
            * altitude_margin
            if has_fuel
            else 0.0
        )
        rocket_enabled = command.rocket_assist and propulsion.rocket_assist_available and has_fuel
        rocket_thrust = propulsion.rocket_thrust_n if rocket_enabled else 0.0
        total_force = aerodynamic.force_body_n + np.array([air_thrust + rocket_thrust, 0.0, 0.0])
        gravity = -planet.gravitational_parameter_m3_s2 * state.position_inertial_m / radius**3
        if planet.j2 > 0.0 and radius > planet.radius_m:
            gravity += j2_acceleration_mps2(
                state.position_inertial_m,
                planet.gravitational_parameter_m3_s2,
                planet.radius_m,
                planet.j2,
            )
        mass_flow = 0.0
        if has_fuel:
            mass_flow += configuration.mass.maximum_fuel_flow_kgps * state.throttle
            if rocket_enabled:
                mass_flow += rocket_thrust / (
                    propulsion.rocket_specific_impulse_s * STANDARD_GRAVITY_MPS2
                )
        return AircraftLoads(
            altitude,
            latitude,
            longitude_ground,
            air_velocity_inertial,
            air_velocity_body,
            aerodynamic,
            float(air_thrust),
            float(rocket_thrust),
            total_force,
            aerodynamic.moment_body_nm,
            gravity,
            float(mass_flow),
            atmosphere.density_kgpm3,
            atmosphere.speed_of_sound_mps,
        )

    def derivative(
        self,
        time_s: float,
        values: FloatArray,
        command: AircraftControlCommand,
    ) -> FloatArray:
        """Evaluate the nonlinear 18-state derivative for a held pilot command."""
        state = AircraftState.from_array(values, normalize=True)
        loads = self.loads(time_s, values, command)
        dcm_ib = quaternion_to_dcm(state.quaternion_ib)
        effective_mass = max(state.mass_kg, self.configuration.mass.dry_mass_kg)
        acceleration = (
            dcm_ib @ loads.total_force_body_n / effective_mass + loads.gravity_inertial_mps2
        )
        inertia_reference = np.diag(self.configuration.mass.inertia_diagonal_kgm2)
        inertia_scale = effective_mass / self.configuration.mass.initial_mass_kg
        inertia = inertia_scale * inertia_reference
        inertia_rate = (
            -loads.mass_flow_kgps / self.configuration.mass.initial_mass_kg * inertia_reference
        )
        angular_momentum = inertia @ state.angular_rate_body_radps
        angular_acceleration = np.linalg.solve(
            inertia,
            loads.total_moment_body_nm
            - np.cross(state.angular_rate_body_radps, angular_momentum)
            - inertia_rate @ state.angular_rate_body_radps,
        )
        quaternion_rate = 0.5 * quaternion_multiply(
            state.quaternion_ib,
            np.concatenate(([0.0], state.angular_rate_body_radps)),
        )
        geometry = self.configuration.geometry
        target_surfaces = np.array(
            [
                command.roll * geometry.aileron_limit_rad,
                -command.pitch * geometry.elevator_limit_rad,
                -command.yaw * geometry.rudder_limit_rad,
            ]
        )
        surface_rate = np.clip(
            (target_surfaces - state.control_surface_rad) / geometry.control_time_constant_s,
            -geometry.control_rate_limit_radps,
            geometry.control_rate_limit_radps,
        )
        throttle_rate = (command.throttle - state.throttle) / geometry.throttle_time_constant_s
        return np.concatenate(
            (
                state.velocity_inertial_mps,
                acceleration,
                quaternion_rate,
                angular_acceleration,
                [-loads.mass_flow_kgps],
                surface_rate,
                [throttle_rate],
            )
        )

    def local_attitude_rad(self, values: npt.ArrayLike) -> tuple[float, float, float]:
        """Return local-NED roll, pitch, heading from the inertial quaternion."""
        state = AircraftState.from_array(values, normalize=True)
        dcm_in = local_ned_dcm_inertial(state.position_inertial_m)
        dcm_nb = dcm_in.T @ quaternion_to_dcm(state.quaternion_ib)
        return quaternion_to_euler321(dcm_to_quaternion(dcm_nb))


def aircraft_stall_speed_mps(
    mass_kg: float,
    density_kgpm3: float,
    configuration: AircraftSandboxConfiguration,
    load_factor: float = 1.0,
) -> float:
    """Return the level/loaded theoretical onset speed from configured ``CL_max``."""
    values = np.asarray([mass_kg, density_kgpm3, load_factor])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("stall-speed inputs must be positive and finite")
    return float(
        np.sqrt(
            2.0
            * mass_kg
            * STANDARD_GRAVITY_MPS2
            * load_factor
            / (
                density_kgpm3
                * configuration.geometry.wing_area_m2
                * configuration.aerodynamics.cl_maximum
            )
        )
    )
