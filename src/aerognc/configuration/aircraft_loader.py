"""Strict configuration boundary for the fictional civilian aircraft sandbox."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.loader import (
    ConfigurationError,
    _boolean,
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _string,
)


@dataclass(frozen=True, slots=True)
class AircraftPlanetDefinition:
    """Synthetic spherical rotating planet used by the aircraft sandbox."""

    name: str
    gravitational_parameter_m3_s2: float
    radius_m: float
    j2: float
    rotation_rate_radps: float
    atmosphere_density_scale: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.gravitational_parameter_m3_s2,
                self.radius_m,
                self.j2,
                self.rotation_rate_radps,
                self.atmosphere_density_scale,
            ]
        )
        if not self.name.strip() or not np.all(np.isfinite(values)):
            raise ValueError("aircraft planet values must be named and finite")
        if self.gravitational_parameter_m3_s2 <= 0.0 or self.radius_m <= 0.0:
            raise ValueError("aircraft planet gravity and radius must be positive")
        if not 0.0 <= self.j2 < 0.1 or self.rotation_rate_radps < 0.0:
            raise ValueError("aircraft planet J2/rotation values are outside the model domain")
        if not 0.0 <= self.atmosphere_density_scale <= 1.0e3:
            raise ValueError("aircraft atmosphere density scale must lie in [0, 1000]")


@dataclass(frozen=True, slots=True)
class AircraftGeometryDefinition:
    """Reference geometry and control-surface limits in SI units."""

    wing_area_m2: float
    wingspan_m: float
    mean_chord_m: float
    aileron_limit_rad: float
    elevator_limit_rad: float
    rudder_limit_rad: float
    control_time_constant_s: float
    control_rate_limit_radps: float
    throttle_time_constant_s: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.wing_area_m2,
                self.wingspan_m,
                self.mean_chord_m,
                self.aileron_limit_rad,
                self.elevator_limit_rad,
                self.rudder_limit_rad,
                self.control_time_constant_s,
                self.control_rate_limit_radps,
                self.throttle_time_constant_s,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("aircraft geometry and actuator values must be positive and finite")
        if np.any(values[3:6] > np.deg2rad(60.0)):
            raise ValueError("aircraft control deflections cannot exceed 60 deg")


@dataclass(frozen=True, slots=True)
class AircraftMassDefinition:
    """Initial/dry mass, reference inertia, and fuel-flow settings."""

    initial_mass_kg: float
    dry_mass_kg: float
    inertia_diagonal_kgm2: tuple[float, float, float]
    maximum_fuel_flow_kgps: float

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_diagonal_kgm2, dtype=np.float64)
        values = np.concatenate(
            (
                np.asarray([self.initial_mass_kg, self.dry_mass_kg, self.maximum_fuel_flow_kgps]),
                inertia,
            )
        )
        if inertia.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("aircraft mass/inertia values must be positive and finite")
        if self.dry_mass_kg >= self.initial_mass_kg:
            raise ValueError("aircraft dry mass must be lower than initial mass")


@dataclass(frozen=True, slots=True)
class AircraftAerodynamicsDefinition:
    """Dimensionless nonlinear aerodynamic stability/control derivatives."""

    cl_zero: float
    cl_alpha_per_rad: float
    cl_elevator_per_rad: float
    cl_pitch_rate: float
    cl_maximum: float
    stall_angle_rad: float
    post_stall_lift_fraction: float
    cd_zero: float
    induced_drag_factor: float
    stall_drag_increment: float
    side_force_beta_per_rad: float
    side_force_aileron_per_rad: float
    side_force_rudder_per_rad: float
    roll_beta_per_rad: float
    roll_rate: float
    roll_yaw_rate: float
    roll_aileron_per_rad: float
    roll_rudder_per_rad: float
    pitch_zero: float
    pitch_alpha_per_rad: float
    pitch_rate: float
    pitch_elevator_per_rad: float
    yaw_beta_per_rad: float
    yaw_roll_rate: float
    yaw_rate: float
    yaw_aileron_per_rad: float
    yaw_rudder_per_rad: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [getattr(self, field) for field in self.__dataclass_fields__], dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("aircraft aerodynamic coefficients must be finite")
        if self.cl_alpha_per_rad <= 0.0 or self.cl_maximum <= 0.0:
            raise ValueError("CL-alpha and CL maximum must be positive")
        if not np.deg2rad(3.0) <= self.stall_angle_rad <= np.deg2rad(45.0):
            raise ValueError("stall angle must lie in [3, 45] deg")
        if not 0.0 <= self.post_stall_lift_fraction <= 1.0:
            raise ValueError("post-stall lift fraction must lie in [0, 1]")
        if self.cd_zero <= 0.0 or self.induced_drag_factor < 0.0:
            raise ValueError("zero-lift drag must be positive and induced drag nonnegative")


@dataclass(frozen=True, slots=True)
class AircraftPropulsionDefinition:
    """Air-breathing engine plus optional educational high-altitude rocket assist."""

    maximum_thrust_n: float
    thrust_density_exponent: float
    maximum_operating_mach: float
    maximum_operating_altitude_m: float
    rocket_assist_available: bool
    rocket_thrust_n: float
    rocket_specific_impulse_s: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.maximum_thrust_n,
                self.thrust_density_exponent,
                self.maximum_operating_mach,
                self.maximum_operating_altitude_m,
                self.rocket_thrust_n,
                self.rocket_specific_impulse_s,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("aircraft propulsion values must be positive and finite")
        if self.thrust_density_exponent > 2.0:
            raise ValueError("thrust density exponent cannot exceed 2")


@dataclass(frozen=True, slots=True)
class AircraftInitialCondition:
    """Geodetic start and local flight condition."""

    latitude_rad: float
    longitude_rad: float
    altitude_m: float
    true_airspeed_mps: float
    heading_rad: float
    flight_path_angle_rad: float
    bank_angle_rad: float
    angle_of_attack_rad: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.latitude_rad,
                self.longitude_rad,
                self.altitude_m,
                self.true_airspeed_mps,
                self.heading_rad,
                self.flight_path_angle_rad,
                self.bank_angle_rad,
                self.angle_of_attack_rad,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("aircraft initial condition must be finite")
        if not -np.pi / 2.0 <= self.latitude_rad <= np.pi / 2.0:
            raise ValueError("aircraft latitude must lie in [-pi/2, pi/2]")
        if self.altitude_m < 0.0 or self.true_airspeed_mps <= 0.0:
            raise ValueError("aircraft altitude must be nonnegative and airspeed positive")
        if abs(self.flight_path_angle_rad) > np.deg2rad(60.0):
            raise ValueError("initial flight-path angle cannot exceed 60 deg")
        if abs(self.angle_of_attack_rad) > np.deg2rad(30.0):
            raise ValueError("initial angle of attack cannot exceed 30 deg")


@dataclass(frozen=True, slots=True)
class AircraftSandboxConfiguration:
    """Complete coefficient-driven fixed-wing sandbox configuration."""

    source_path: Path
    name: str
    safety_scope: str
    planet: AircraftPlanetDefinition
    geometry: AircraftGeometryDefinition
    mass: AircraftMassDefinition
    aerodynamics: AircraftAerodynamicsDefinition
    aerodynamic_backend: str
    aerodynamic_table_path: Path | None
    propulsion: AircraftPropulsionDefinition
    initial: AircraftInitialCondition
    initial_throttle: float
    duration_s: float
    integration_step_s: float
    output_step_s: float
    wind_north_mps: float
    wind_east_mps: float
    turbulence_std_ned_mps: tuple[float, float, float]
    turbulence_correlation_time_s: float
    wind_random_seed: int
    gust_start_time_s: float
    gust_duration_s: float
    gust_amplitude_ned_mps: tuple[float, float, float]
    output_directory: Path

    def __post_init__(self) -> None:
        timing = np.asarray([self.duration_s, self.integration_step_s, self.output_step_s])
        if not self.name.strip() or not self.safety_scope.strip():
            raise ValueError("aircraft scenario name and safety scope cannot be empty")
        if not np.all(np.isfinite(timing)) or np.any(timing <= 0.0):
            raise ValueError("aircraft simulation timing must be positive and finite")
        if self.duration_s > 86_400.0 or self.integration_step_s > 0.1:
            raise ValueError("aircraft batch duration/step exceed the interactive model domain")
        if self.integration_step_s > self.output_step_s:
            raise ValueError("aircraft integration step cannot exceed output step")
        ratio = self.output_step_s / self.integration_step_s
        if not np.isclose(ratio, round(ratio), atol=1.0e-10):
            raise ValueError("aircraft output step must be an integer multiple of integration step")
        if not 0.0 <= self.initial_throttle <= 1.0:
            raise ValueError("initial aircraft throttle must lie in [0, 1]")
        if self.aerodynamic_backend not in {"analytic", "table"}:
            raise ValueError("aircraft aerodynamic backend must be analytic or table")
        if self.aerodynamic_backend == "table" and self.aerodynamic_table_path is None:
            raise ValueError("table aerodynamic backend requires a CSV table path")
        wind_values = np.asarray(
            [
                self.wind_north_mps,
                self.wind_east_mps,
                *self.turbulence_std_ned_mps,
                self.turbulence_correlation_time_s,
                self.gust_start_time_s,
                self.gust_duration_s,
                *self.gust_amplitude_ned_mps,
            ]
        )
        if not np.all(np.isfinite(wind_values)):
            raise ValueError("aircraft wind values must be finite")
        if np.any(np.asarray(self.turbulence_std_ned_mps) < 0.0):
            raise ValueError("aircraft turbulence standard deviations must be nonnegative")
        if self.turbulence_correlation_time_s <= 0.0:
            raise ValueError("aircraft turbulence correlation time must be positive")
        if self.gust_start_time_s < 0.0 or self.gust_duration_s <= 0.0:
            raise ValueError("aircraft discrete-gust time values are outside their domain")
        if isinstance(self.wind_random_seed, bool) or self.wind_random_seed < 0:
            raise ValueError("aircraft wind random seed must be a nonnegative integer")


def _degrees(data: Mapping[str, object], key: str, context: str) -> float:
    return float(np.deg2rad(_number(data[key], f"{context}.{key}")))


def load_aircraft_configuration(path: str | Path) -> AircraftSandboxConfiguration:
    """Load one strict fictional-aircraft YAML file."""
    source = Path(path).resolve()
    root = _load_yaml(source)
    _keys(
        root,
        "aircraft",
        required={
            "metadata",
            "planet",
            "geometry",
            "mass",
            "aerodynamics",
            "propulsion",
            "initial_condition",
            "environment",
            "simulation",
            "output_directory",
        },
    )
    metadata = _mapping(root["metadata"], "aircraft.metadata")
    _keys(metadata, "aircraft.metadata", required={"name", "fictional", "safety_scope"})
    if not _boolean(metadata["fictional"], "aircraft.metadata.fictional"):
        raise ConfigurationError("aircraft.metadata.fictional must be true")
    safety_scope = _string(metadata["safety_scope"], "aircraft.metadata.safety_scope")
    folded = safety_scope.casefold()
    if not all(word in folded for word in ("fictional", "civilian", "synthetic")):
        raise ConfigurationError("aircraft safety scope must state fictional, civilian, synthetic")

    planet_data = _mapping(root["planet"], "aircraft.planet")
    _keys(
        planet_data,
        "aircraft.planet",
        required={
            "name",
            "gravitational_parameter_m3_s2",
            "radius_m",
            "j2",
            "rotation_rate_radps",
            "atmosphere_density_scale",
        },
    )
    planet = AircraftPlanetDefinition(
        _string(planet_data["name"], "aircraft.planet.name"),
        _number(
            planet_data["gravitational_parameter_m3_s2"],
            "aircraft.planet.gravitational_parameter_m3_s2",
            positive=True,
        ),
        _number(planet_data["radius_m"], "aircraft.planet.radius_m", positive=True),
        _number(planet_data["j2"], "aircraft.planet.j2", nonnegative=True),
        _number(
            planet_data["rotation_rate_radps"],
            "aircraft.planet.rotation_rate_radps",
            nonnegative=True,
        ),
        _number(
            planet_data["atmosphere_density_scale"],
            "aircraft.planet.atmosphere_density_scale",
            nonnegative=True,
        ),
    )

    geometry_data = _mapping(root["geometry"], "aircraft.geometry")
    _keys(
        geometry_data,
        "aircraft.geometry",
        required={
            "wing_area_m2",
            "wingspan_m",
            "mean_chord_m",
            "aileron_limit_deg",
            "elevator_limit_deg",
            "rudder_limit_deg",
            "control_time_constant_s",
            "control_rate_limit_degps",
            "throttle_time_constant_s",
        },
    )
    geometry = AircraftGeometryDefinition(
        _number(geometry_data["wing_area_m2"], "aircraft.geometry.wing_area_m2", positive=True),
        _number(geometry_data["wingspan_m"], "aircraft.geometry.wingspan_m", positive=True),
        _number(geometry_data["mean_chord_m"], "aircraft.geometry.mean_chord_m", positive=True),
        _degrees(geometry_data, "aileron_limit_deg", "aircraft.geometry"),
        _degrees(geometry_data, "elevator_limit_deg", "aircraft.geometry"),
        _degrees(geometry_data, "rudder_limit_deg", "aircraft.geometry"),
        _number(
            geometry_data["control_time_constant_s"],
            "aircraft.geometry.control_time_constant_s",
            positive=True,
        ),
        _degrees(geometry_data, "control_rate_limit_degps", "aircraft.geometry"),
        _number(
            geometry_data["throttle_time_constant_s"],
            "aircraft.geometry.throttle_time_constant_s",
            positive=True,
        ),
    )

    mass_data = _mapping(root["mass"], "aircraft.mass")
    _keys(
        mass_data,
        "aircraft.mass",
        required={
            "initial_mass_kg",
            "dry_mass_kg",
            "inertia_xx_kgm2",
            "inertia_yy_kgm2",
            "inertia_zz_kgm2",
            "maximum_fuel_flow_kgps",
        },
    )
    mass = AircraftMassDefinition(
        _number(mass_data["initial_mass_kg"], "aircraft.mass.initial_mass_kg", positive=True),
        _number(mass_data["dry_mass_kg"], "aircraft.mass.dry_mass_kg", positive=True),
        (
            _number(mass_data["inertia_xx_kgm2"], "aircraft.mass.inertia_xx_kgm2", positive=True),
            _number(mass_data["inertia_yy_kgm2"], "aircraft.mass.inertia_yy_kgm2", positive=True),
            _number(mass_data["inertia_zz_kgm2"], "aircraft.mass.inertia_zz_kgm2", positive=True),
        ),
        _number(
            mass_data["maximum_fuel_flow_kgps"],
            "aircraft.mass.maximum_fuel_flow_kgps",
            positive=True,
        ),
    )

    aero_data = _mapping(root["aerodynamics"], "aircraft.aerodynamics")
    aero_keys = {
        "cl_zero",
        "cl_alpha_per_rad",
        "cl_elevator_per_rad",
        "cl_pitch_rate",
        "cl_maximum",
        "stall_angle_deg",
        "post_stall_lift_fraction",
        "cd_zero",
        "induced_drag_factor",
        "stall_drag_increment",
        "side_force_beta_per_rad",
        "side_force_aileron_per_rad",
        "side_force_rudder_per_rad",
        "roll_beta_per_rad",
        "roll_rate",
        "roll_yaw_rate",
        "roll_aileron_per_rad",
        "roll_rudder_per_rad",
        "pitch_zero",
        "pitch_alpha_per_rad",
        "pitch_rate",
        "pitch_elevator_per_rad",
        "yaw_beta_per_rad",
        "yaw_roll_rate",
        "yaw_rate",
        "yaw_aileron_per_rad",
        "yaw_rudder_per_rad",
    }
    _keys(
        aero_data,
        "aircraft.aerodynamics",
        required=aero_keys,
        optional={"backend", "table_path"},
    )
    aero_values = {
        key: _number(aero_data[key], f"aircraft.aerodynamics.{key}") for key in aero_keys
    }
    stall_angle_rad = aero_values.pop("stall_angle_deg") * np.pi / 180.0
    aerodynamics = AircraftAerodynamicsDefinition(
        **aero_values,
        stall_angle_rad=stall_angle_rad,
    )
    aerodynamic_backend = _string(
        aero_data.get("backend", "analytic"), "aircraft.aerodynamics.backend"
    ).casefold()
    if aerodynamic_backend not in {"analytic", "table"}:
        raise ConfigurationError("aircraft.aerodynamics.backend must be analytic or table")
    table_path_value = aero_data.get("table_path")
    aerodynamic_table_path = (
        None
        if table_path_value is None
        else (
            source.parent / _string(table_path_value, "aircraft.aerodynamics.table_path")
        ).resolve()
    )
    if aerodynamic_backend == "table" and (
        aerodynamic_table_path is None or not aerodynamic_table_path.is_file()
    ):
        raise ConfigurationError(
            "aircraft.aerodynamics.table_path must name an existing CSV for table backend"
        )

    propulsion_data = _mapping(root["propulsion"], "aircraft.propulsion")
    _keys(
        propulsion_data,
        "aircraft.propulsion",
        required={
            "maximum_thrust_n",
            "thrust_density_exponent",
            "maximum_operating_mach",
            "maximum_operating_altitude_m",
            "rocket_assist_available",
            "rocket_thrust_n",
            "rocket_specific_impulse_s",
        },
    )
    propulsion = AircraftPropulsionDefinition(
        _number(
            propulsion_data["maximum_thrust_n"],
            "aircraft.propulsion.maximum_thrust_n",
            positive=True,
        ),
        _number(
            propulsion_data["thrust_density_exponent"],
            "aircraft.propulsion.thrust_density_exponent",
            positive=True,
        ),
        _number(
            propulsion_data["maximum_operating_mach"],
            "aircraft.propulsion.maximum_operating_mach",
            positive=True,
        ),
        _number(
            propulsion_data["maximum_operating_altitude_m"],
            "aircraft.propulsion.maximum_operating_altitude_m",
            positive=True,
        ),
        _boolean(
            propulsion_data["rocket_assist_available"],
            "aircraft.propulsion.rocket_assist_available",
        ),
        _number(
            propulsion_data["rocket_thrust_n"],
            "aircraft.propulsion.rocket_thrust_n",
            positive=True,
        ),
        _number(
            propulsion_data["rocket_specific_impulse_s"],
            "aircraft.propulsion.rocket_specific_impulse_s",
            positive=True,
        ),
    )

    initial_data = _mapping(root["initial_condition"], "aircraft.initial_condition")
    _keys(
        initial_data,
        "aircraft.initial_condition",
        required={
            "latitude_deg",
            "longitude_deg",
            "altitude_m",
            "true_airspeed_mps",
            "heading_deg",
            "flight_path_angle_deg",
            "bank_angle_deg",
            "angle_of_attack_deg",
            "throttle",
        },
    )
    initial = AircraftInitialCondition(
        _degrees(initial_data, "latitude_deg", "aircraft.initial_condition"),
        _degrees(initial_data, "longitude_deg", "aircraft.initial_condition"),
        _number(
            initial_data["altitude_m"], "aircraft.initial_condition.altitude_m", nonnegative=True
        ),
        _number(
            initial_data["true_airspeed_mps"],
            "aircraft.initial_condition.true_airspeed_mps",
            positive=True,
        ),
        _degrees(initial_data, "heading_deg", "aircraft.initial_condition"),
        _degrees(initial_data, "flight_path_angle_deg", "aircraft.initial_condition"),
        _degrees(initial_data, "bank_angle_deg", "aircraft.initial_condition"),
        _degrees(initial_data, "angle_of_attack_deg", "aircraft.initial_condition"),
    )

    environment_data = _mapping(root["environment"], "aircraft.environment")
    _keys(
        environment_data,
        "aircraft.environment",
        required={"wind_north_mps", "wind_east_mps"},
        optional={
            "turbulence_std_north_mps",
            "turbulence_std_east_mps",
            "turbulence_std_down_mps",
            "turbulence_correlation_time_s",
            "random_seed",
            "gust_start_time_s",
            "gust_duration_s",
            "gust_north_mps",
            "gust_east_mps",
            "gust_down_mps",
        },
    )
    simulation_data = _mapping(root["simulation"], "aircraft.simulation")
    _keys(
        simulation_data,
        "aircraft.simulation",
        required={"duration_s", "integration_step_s", "output_step_s"},
    )
    return AircraftSandboxConfiguration(
        source_path=source,
        name=_string(metadata["name"], "aircraft.metadata.name"),
        safety_scope=safety_scope,
        planet=planet,
        geometry=geometry,
        mass=mass,
        aerodynamics=aerodynamics,
        aerodynamic_backend=aerodynamic_backend,
        aerodynamic_table_path=aerodynamic_table_path,
        propulsion=propulsion,
        initial=initial,
        initial_throttle=_number(
            initial_data["throttle"], "aircraft.initial_condition.throttle", nonnegative=True
        ),
        duration_s=_number(
            simulation_data["duration_s"], "aircraft.simulation.duration_s", positive=True
        ),
        integration_step_s=_number(
            simulation_data["integration_step_s"],
            "aircraft.simulation.integration_step_s",
            positive=True,
        ),
        output_step_s=_number(
            simulation_data["output_step_s"],
            "aircraft.simulation.output_step_s",
            positive=True,
        ),
        wind_north_mps=_number(
            environment_data["wind_north_mps"], "aircraft.environment.wind_north_mps"
        ),
        wind_east_mps=_number(
            environment_data["wind_east_mps"], "aircraft.environment.wind_east_mps"
        ),
        turbulence_std_ned_mps=(
            _number(
                environment_data.get("turbulence_std_north_mps", 0.0),
                "aircraft.environment.turbulence_std_north_mps",
                nonnegative=True,
            ),
            _number(
                environment_data.get("turbulence_std_east_mps", 0.0),
                "aircraft.environment.turbulence_std_east_mps",
                nonnegative=True,
            ),
            _number(
                environment_data.get("turbulence_std_down_mps", 0.0),
                "aircraft.environment.turbulence_std_down_mps",
                nonnegative=True,
            ),
        ),
        turbulence_correlation_time_s=_number(
            environment_data.get("turbulence_correlation_time_s", 2.0),
            "aircraft.environment.turbulence_correlation_time_s",
            positive=True,
        ),
        wind_random_seed=_integer(
            environment_data.get("random_seed", 218),
            "aircraft.environment.random_seed",
            nonnegative=True,
        ),
        gust_start_time_s=_number(
            environment_data.get("gust_start_time_s", 0.0),
            "aircraft.environment.gust_start_time_s",
            nonnegative=True,
        ),
        gust_duration_s=_number(
            environment_data.get("gust_duration_s", 1.0),
            "aircraft.environment.gust_duration_s",
            positive=True,
        ),
        gust_amplitude_ned_mps=(
            _number(
                environment_data.get("gust_north_mps", 0.0),
                "aircraft.environment.gust_north_mps",
            ),
            _number(
                environment_data.get("gust_east_mps", 0.0),
                "aircraft.environment.gust_east_mps",
            ),
            _number(
                environment_data.get("gust_down_mps", 0.0),
                "aircraft.environment.gust_down_mps",
            ),
        ),
        output_directory=(
            source.parent / _string(root["output_directory"], "output_directory")
        ).resolve(),
    )
