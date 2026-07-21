"""Strict YAML-to-dataclass configuration boundary."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, NoReturn, cast

import numpy as np
import yaml

from aerognc.configuration.models import (
    EnvironmentDefinition,
    LaunchConfiguration,
    SimulationConfiguration,
    ThreeDofConfiguration,
    VehicleDefinition,
)
from aerognc.environment.atmosphere import StandardAtmosphere1976
from aerognc.environment.gravity import GravityMode, GravityModel
from aerognc.environment.wind import WindModel, WindProfile
from aerognc.mathematics.interpolation import OutOfRange
from aerognc.vehicle.actuators import ActuatorAllocator, ActuatorLimits
from aerognc.vehicle.aero_database import TabulatedAerodynamicDatabase
from aerognc.vehicle.aerodynamics import AerodynamicModel
from aerognc.vehicle.mass_properties import MassPropertiesModel
from aerognc.vehicle.propulsion import ThrustCurve


class ConfigurationError(ValueError):
    """Raised when a configuration is incomplete, ambiguous, or nonphysical."""


def _fail(context: str, message: str) -> NoReturn:
    raise ConfigurationError(f"{context}: {message}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(context, "expected a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail(context, "all keys must be strings")
    return cast(Mapping[str, object], value)


def _keys(
    mapping: Mapping[str, object],
    context: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    missing = required - mapping.keys()
    unknown = mapping.keys() - required - optional
    if missing:
        _fail(context, f"missing required keys: {', '.join(sorted(missing))}")
    if unknown:
        _fail(context, f"unknown keys: {', '.join(sorted(unknown))}")


def _number(
    value: object, context: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(context, "expected a number")
    result = float(value)
    if not np.isfinite(result):
        _fail(context, "must be finite")
    if positive and result <= 0.0:
        _fail(context, "must be positive")
    if nonnegative and result < 0.0:
        _fail(context, "must be nonnegative")
    return result


def _integer(value: object, context: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(context, "expected an integer")
    if nonnegative and value < 0:
        _fail(context, "must be nonnegative")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(context, "expected a nonempty string")
    return value.strip()


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(context, "expected true or false")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(context, "expected a sequence")
    return cast(Sequence[object], value)


def _number_tuple(value: object, context: str, *, length: int | None = None) -> tuple[float, ...]:
    sequence = _sequence(value, context)
    if length is not None and len(sequence) != length:
        _fail(context, f"expected exactly {length} values")
    return tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(sequence))


def _matrix3(value: object, context: str) -> tuple[tuple[float, float, float], ...]:
    rows = _sequence(value, context)
    if len(rows) != 3:
        _fail(context, "expected three rows")
    parsed = tuple(
        _number_tuple(row, f"{context}[{index}]", length=3) for index, row in enumerate(rows)
    )
    return cast(tuple[tuple[float, float, float], ...], parsed)


def _load_yaml(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {path}")
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {path}: {error}") from error
    return _mapping(loaded, str(path))


def _load_vehicle(path: Path) -> VehicleDefinition:
    root = _load_yaml(path)
    _keys(
        root,
        "vehicle",
        required={
            "metadata",
            "geometry",
            "propulsion",
            "mass_properties",
            "aerodynamics",
            "actuator",
        },
    )
    metadata = _mapping(root["metadata"], "vehicle.metadata")
    _keys(metadata, "vehicle.metadata", required={"name", "description", "fictional"})
    fictional = _boolean(metadata["fictional"], "vehicle.metadata.fictional")
    if not fictional:
        _fail("vehicle.metadata.fictional", "public baseline vehicles must be explicitly fictional")

    geometry = _mapping(root["geometry"], "vehicle.geometry")
    _keys(geometry, "vehicle.geometry", required={"reference_area_m2", "reference_length_m"})
    area_m2 = _number(
        geometry["reference_area_m2"], "vehicle.geometry.reference_area_m2", positive=True
    )
    length_m = _number(
        geometry["reference_length_m"], "vehicle.geometry.reference_length_m", positive=True
    )

    propulsion_data = _mapping(root["propulsion"], "vehicle.propulsion")
    _keys(
        propulsion_data, "vehicle.propulsion", required={"time_s", "thrust_n", "propellant_mass_kg"}
    )
    propulsion = ThrustCurve(
        _number_tuple(propulsion_data["time_s"], "vehicle.propulsion.time_s"),
        _number_tuple(propulsion_data["thrust_n"], "vehicle.propulsion.thrust_n"),
        _number(
            propulsion_data["propellant_mass_kg"],
            "vehicle.propulsion.propellant_mass_kg",
            positive=True,
        ),
    )

    mass_data = _mapping(root["mass_properties"], "vehicle.mass_properties")
    _keys(
        mass_data,
        "vehicle.mass_properties",
        required={
            "dry_mass_kg",
            "dry_cg_from_nose_m",
            "wet_cg_from_nose_m",
            "dry_inertia_body_kgm2",
            "wet_inertia_body_kgm2",
        },
    )
    mass_properties = MassPropertiesModel(
        _number(mass_data["dry_mass_kg"], "vehicle.mass_properties.dry_mass_kg", positive=True),
        propulsion,
        _number(
            mass_data["dry_cg_from_nose_m"],
            "vehicle.mass_properties.dry_cg_from_nose_m",
            nonnegative=True,
        ),
        _number(
            mass_data["wet_cg_from_nose_m"],
            "vehicle.mass_properties.wet_cg_from_nose_m",
            nonnegative=True,
        ),
        _matrix3(
            mass_data["dry_inertia_body_kgm2"], "vehicle.mass_properties.dry_inertia_body_kgm2"
        ),
        _matrix3(
            mass_data["wet_inertia_body_kgm2"], "vehicle.mass_properties.wet_inertia_body_kgm2"
        ),
    )

    aero = _mapping(root["aerodynamics"], "vehicle.aerodynamics")
    if "out_of_range" not in aero:
        _fail("vehicle.aerodynamics", "missing required key: out_of_range")
    policy = _string(aero["out_of_range"], "vehicle.aerodynamics.out_of_range")
    if policy not in {"error", "clamp", "extrapolate"}:
        _fail("vehicle.aerodynamics.out_of_range", "expected error, clamp, or extrapolate")
    if "database_file" in aero:
        _keys(
            aero,
            "vehicle.aerodynamics",
            required={"database_file", "out_of_range"},
        )
        database_path = path.parent / _string(
            aero["database_file"], "vehicle.aerodynamics.database_file"
        )
        try:
            database = TabulatedAerodynamicDatabase.from_csv(
                database_path,
                out_of_range=cast(OutOfRange, policy),
            )
        except ValueError as error:
            _fail("vehicle.aerodynamics.database_file", str(error))
        aerodynamics = AerodynamicModel(
            reference_area_m2=area_m2,
            reference_length_m=length_m,
            coefficient_provider=database,
        )
    else:
        required_aero = {
            "mach_points",
            "drag_coefficients",
            "out_of_range",
            "drag_alpha2_per_rad2",
            "side_beta_per_rad",
            "normal_alpha_per_rad",
            "roll_beta_per_rad",
            "pitch_alpha_per_rad",
            "yaw_beta_per_rad",
            "roll_rate",
            "pitch_rate",
            "yaw_rate",
        }
        _keys(aero, "vehicle.aerodynamics", required=required_aero)
        aerodynamics = AerodynamicModel(
            reference_area_m2=area_m2,
            reference_length_m=length_m,
            mach_points=_number_tuple(aero["mach_points"], "vehicle.aerodynamics.mach_points"),
            drag_coefficients=_number_tuple(
                aero["drag_coefficients"], "vehicle.aerodynamics.drag_coefficients"
            ),
            out_of_range=cast(OutOfRange, policy),
            drag_alpha2_per_rad2=_number(
                aero["drag_alpha2_per_rad2"], "vehicle.aerodynamics.drag_alpha2_per_rad2"
            ),
            side_beta_per_rad=_number(
                aero["side_beta_per_rad"], "vehicle.aerodynamics.side_beta_per_rad"
            ),
            normal_alpha_per_rad=_number(
                aero["normal_alpha_per_rad"], "vehicle.aerodynamics.normal_alpha_per_rad"
            ),
            roll_beta_per_rad=_number(
                aero["roll_beta_per_rad"], "vehicle.aerodynamics.roll_beta_per_rad"
            ),
            pitch_alpha_per_rad=_number(
                aero["pitch_alpha_per_rad"], "vehicle.aerodynamics.pitch_alpha_per_rad"
            ),
            yaw_beta_per_rad=_number(
                aero["yaw_beta_per_rad"], "vehicle.aerodynamics.yaw_beta_per_rad"
            ),
            roll_rate=_number(aero["roll_rate"], "vehicle.aerodynamics.roll_rate"),
            pitch_rate=_number(aero["pitch_rate"], "vehicle.aerodynamics.pitch_rate"),
            yaw_rate=_number(aero["yaw_rate"], "vehicle.aerodynamics.yaw_rate"),
        )

    actuator = _mapping(root["actuator"], "vehicle.actuator")
    _keys(
        actuator,
        "vehicle.actuator",
        required={
            "time_constant_s",
            "position_limit_deg",
            "rate_limit_degps",
            "command_delay_s",
            "moment_per_command_nm_per_rad",
        },
    )
    position_limit_rad = np.deg2rad(
        _number(
            actuator["position_limit_deg"], "vehicle.actuator.position_limit_deg", positive=True
        )
    )
    limits = ActuatorLimits(
        time_constant_s=_number(
            actuator["time_constant_s"], "vehicle.actuator.time_constant_s", positive=True
        ),
        position_limit_rad=float(position_limit_rad),
        rate_limit_radps=float(
            np.deg2rad(
                _number(
                    actuator["rate_limit_degps"], "vehicle.actuator.rate_limit_degps", positive=True
                )
            )
        ),
        command_delay_s=_number(
            actuator["command_delay_s"], "vehicle.actuator.command_delay_s", nonnegative=True
        ),
    )
    allocator = ActuatorAllocator(
        _number_tuple(
            actuator["moment_per_command_nm_per_rad"],
            "vehicle.actuator.moment_per_command_nm_per_rad",
            length=3,
        ),
        [position_limit_rad] * 3,
    )
    return VehicleDefinition(
        name=_string(metadata["name"], "vehicle.metadata.name"),
        description=_string(metadata["description"], "vehicle.metadata.description"),
        fictional=fictional,
        propulsion=propulsion,
        mass_properties=mass_properties,
        aerodynamics=aerodynamics,
        actuator_limits=limits,
        actuator_allocator=allocator,
    )


def load_three_dof_configuration(path: str | Path) -> ThreeDofConfiguration:
    """Load and validate one complete 3-DOF YAML scenario."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "scenario",
        required={"metadata", "vehicle_file", "simulation", "launch", "environment"},
    )
    metadata = _mapping(root["metadata"], "scenario.metadata")
    _keys(metadata, "scenario.metadata", required={"safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "scenario.metadata.safety_scope")
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        _fail(
            "scenario.metadata.safety_scope", "must explicitly state fictional and civilian scope"
        )

    simulation_data = _mapping(root["simulation"], "scenario.simulation")
    _keys(
        simulation_data,
        "scenario.simulation",
        required={"name", "step_s", "maximum_time_s", "random_seed", "output_directory"},
    )
    simulation = SimulationConfiguration(
        name=_string(simulation_data["name"], "scenario.simulation.name"),
        step_s=_number(simulation_data["step_s"], "scenario.simulation.step_s", positive=True),
        maximum_time_s=_number(
            simulation_data["maximum_time_s"], "scenario.simulation.maximum_time_s", positive=True
        ),
        random_seed=_integer(
            simulation_data["random_seed"], "scenario.simulation.random_seed", nonnegative=True
        ),
        output_directory=Path(
            _string(simulation_data["output_directory"], "scenario.simulation.output_directory")
        ),
    )
    if simulation.step_s > 0.1:
        _fail("scenario.simulation.step_s", "must be no greater than 0.1 s")

    launch_data = _mapping(root["launch"], "scenario.launch")
    _keys(
        launch_data,
        "scenario.launch",
        required={
            "position_ned_m",
            "initial_speed_mps",
            "elevation_deg",
            "azimuth_deg",
            "thrust_alignment",
        },
    )
    alignment = _string(launch_data["thrust_alignment"], "scenario.launch.thrust_alignment")
    if alignment not in {"velocity", "launch_axis"}:
        _fail("scenario.launch.thrust_alignment", "expected velocity or launch_axis")
    launch = LaunchConfiguration(
        position_ned_m=cast(
            tuple[float, float, float],
            _number_tuple(
                launch_data["position_ned_m"], "scenario.launch.position_ned_m", length=3
            ),
        ),
        initial_speed_mps=_number(
            launch_data["initial_speed_mps"], "scenario.launch.initial_speed_mps", nonnegative=True
        ),
        elevation_deg=_number(launch_data["elevation_deg"], "scenario.launch.elevation_deg"),
        azimuth_deg=_number(launch_data["azimuth_deg"], "scenario.launch.azimuth_deg"),
        thrust_alignment=cast(Literal["velocity", "launch_axis"], alignment),
    )
    if not 0.0 < launch.elevation_deg <= 90.0:
        _fail("scenario.launch.elevation_deg", "must be in (0, 90] degrees")

    environment = _mapping(root["environment"], "scenario.environment")
    _keys(environment, "scenario.environment", required={"atmosphere", "gravity", "wind"})
    atmosphere_data = _mapping(environment["atmosphere"], "scenario.environment.atmosphere")
    _keys(
        atmosphere_data,
        "scenario.environment.atmosphere",
        required={"minimum_altitude_m", "maximum_altitude_m"},
    )
    atmosphere = StandardAtmosphere1976(
        _number(
            atmosphere_data["minimum_altitude_m"],
            "scenario.environment.atmosphere.minimum_altitude_m",
        ),
        _number(
            atmosphere_data["maximum_altitude_m"],
            "scenario.environment.atmosphere.maximum_altitude_m",
        ),
    )
    gravity_data = _mapping(environment["gravity"], "scenario.environment.gravity")
    _keys(
        gravity_data,
        "scenario.environment.gravity",
        required={"model", "sea_level_mps2", "earth_radius_m"},
    )
    gravity_mode = _string(gravity_data["model"], "scenario.environment.gravity.model")
    if gravity_mode not in {"constant", "inverse_square"}:
        _fail("scenario.environment.gravity.model", "expected constant or inverse_square")
    gravity = GravityModel(
        mode=cast(GravityMode, gravity_mode),
        sea_level_mps2=_number(
            gravity_data["sea_level_mps2"],
            "scenario.environment.gravity.sea_level_mps2",
            positive=True,
        ),
        earth_radius_m=_number(
            gravity_data["earth_radius_m"],
            "scenario.environment.gravity.earth_radius_m",
            positive=True,
        ),
    )
    wind_data = _mapping(environment["wind"], "scenario.environment.wind")
    _keys(
        wind_data,
        "scenario.environment.wind",
        required={
            "altitudes_m",
            "velocities_ned_mps",
            "gust_std_ned_mps",
            "correlation_time_s",
            "sample_step_s",
        },
    )
    velocity_rows = _sequence(
        wind_data["velocities_ned_mps"], "scenario.environment.wind.velocities_ned_mps"
    )
    profile = WindProfile(
        _number_tuple(wind_data["altitudes_m"], "scenario.environment.wind.altitudes_m"),
        [
            _number_tuple(row, f"scenario.environment.wind.velocities_ned_mps[{index}]", length=3)
            for index, row in enumerate(velocity_rows)
        ],
    )
    wind = WindModel(
        profile,
        gust_std_ned_mps=_number_tuple(
            wind_data["gust_std_ned_mps"], "scenario.environment.wind.gust_std_ned_mps", length=3
        ),
        correlation_time_s=_number(
            wind_data["correlation_time_s"],
            "scenario.environment.wind.correlation_time_s",
            positive=True,
        ),
        sample_step_s=_number(
            wind_data["sample_step_s"], "scenario.environment.wind.sample_step_s", positive=True
        ),
        horizon_s=simulation.maximum_time_s + simulation.step_s,
        seed=simulation.random_seed,
    )
    vehicle_file = Path(_string(root["vehicle_file"], "scenario.vehicle_file"))
    if not vehicle_file.is_absolute():
        vehicle_file = source_path.parent / vehicle_file
    vehicle = _load_vehicle(vehicle_file.resolve())
    return ThreeDofConfiguration(
        source_path=source_path,
        safety_scope=safety_scope,
        simulation=simulation,
        launch=launch,
        environment=EnvironmentDefinition(atmosphere, gravity, wind),
        vehicle=vehicle,
    )
