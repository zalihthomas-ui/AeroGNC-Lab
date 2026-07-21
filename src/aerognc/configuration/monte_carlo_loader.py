"""Validated coupled Monte Carlo verification configuration."""

from dataclasses import dataclass
from pathlib import Path

from aerognc.configuration.loader import (
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _string,
)


@dataclass(frozen=True, slots=True)
class MonteCarloDispersions:
    """One-sigma independent dispersion definitions."""

    initial_speed_std_mps: float
    initial_elevation_std_deg: float
    vehicle_mass_scale_std: float
    thrust_scale_std: float
    thrust_misalignment_std_deg: float
    aerodynamic_scale_std: float
    wind_scale_std: float
    sensor_noise_scale_std: float
    sensor_bias_scale_std: float
    actuator_delay_scale_std: float
    controller_gain_scale_std: float


@dataclass(frozen=True, slots=True)
class MonteCarloRequirements:
    """Positive-margin pass/fail thresholds."""

    minimum_apogee_m: float
    maximum_dynamic_pressure_pa: float
    maximum_landing_range_m: float
    maximum_navigation_rms_m: float
    maximum_control_settling_time_s: float


@dataclass(frozen=True, slots=True)
class MonteCarloConfiguration:
    """Paths, execution settings, dispersions, and requirement thresholds."""

    source_path: Path
    name: str
    safety_scope: str
    base_scenario_path: Path
    navigation_config_path: Path
    attitude_config_path: Path
    output_directory: Path
    sample_count: int
    workers: int
    master_seed: int
    dispersions: MonteCarloDispersions
    requirements: MonteCarloRequirements


def _resolved_relative(source_path: Path, value: object, context: str) -> Path:
    path = Path(_string(value, context))
    return path if path.is_absolute() else (source_path.parent / path).resolve()


def load_monte_carlo_configuration(path: str | Path) -> MonteCarloConfiguration:
    """Load one coupled flight/navigation/control ensemble definition."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "monte_carlo",
        required={
            "metadata",
            "base_scenario_file",
            "navigation_config_file",
            "attitude_config_file",
            "execution",
            "dispersions",
            "requirements",
        },
    )
    metadata = _mapping(root["metadata"], "monte_carlo.metadata")
    _keys(metadata, "monte_carlo.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "monte_carlo.metadata.safety_scope")
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        raise ValueError("monte_carlo safety_scope must state fictional and civilian")
    execution = _mapping(root["execution"], "monte_carlo.execution")
    _keys(
        execution,
        "monte_carlo.execution",
        required={"sample_count", "workers", "master_seed", "output_directory"},
    )
    sample_count = _integer(
        execution["sample_count"], "monte_carlo.execution.sample_count", nonnegative=True
    )
    workers = _integer(execution["workers"], "monte_carlo.execution.workers", nonnegative=True)
    if sample_count <= 0 or workers <= 0:
        raise ValueError("Monte Carlo sample_count and workers must be positive")

    dispersion_data = _mapping(root["dispersions"], "monte_carlo.dispersions")
    dispersion_names = {
        "initial_speed_std_mps",
        "initial_elevation_std_deg",
        "vehicle_mass_scale_std",
        "thrust_scale_std",
        "thrust_misalignment_std_deg",
        "aerodynamic_scale_std",
        "wind_scale_std",
        "sensor_noise_scale_std",
        "sensor_bias_scale_std",
        "actuator_delay_scale_std",
        "controller_gain_scale_std",
    }
    _keys(dispersion_data, "monte_carlo.dispersions", required=dispersion_names)
    dispersion_values = {
        name: _number(dispersion_data[name], f"monte_carlo.dispersions.{name}", nonnegative=True)
        for name in dispersion_names
    }
    dispersions = MonteCarloDispersions(**dispersion_values)

    requirement_data = _mapping(root["requirements"], "monte_carlo.requirements")
    requirement_names = {
        "minimum_apogee_m",
        "maximum_dynamic_pressure_pa",
        "maximum_landing_range_m",
        "maximum_navigation_rms_m",
        "maximum_control_settling_time_s",
    }
    _keys(requirement_data, "monte_carlo.requirements", required=requirement_names)
    requirement_values = {
        name: _number(requirement_data[name], f"monte_carlo.requirements.{name}", positive=True)
        for name in requirement_names
    }
    requirements = MonteCarloRequirements(**requirement_values)
    return MonteCarloConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "monte_carlo.metadata.name"),
        safety_scope=safety_scope,
        base_scenario_path=_resolved_relative(
            source_path, root["base_scenario_file"], "monte_carlo.base_scenario_file"
        ),
        navigation_config_path=_resolved_relative(
            source_path,
            root["navigation_config_file"],
            "monte_carlo.navigation_config_file",
        ),
        attitude_config_path=_resolved_relative(
            source_path,
            root["attitude_config_file"],
            "monte_carlo.attitude_config_file",
        ),
        output_directory=Path(
            _string(execution["output_directory"], "monte_carlo.execution.output_directory")
        ),
        sample_count=sample_count,
        workers=workers,
        master_seed=_integer(
            execution["master_seed"], "monte_carlo.execution.master_seed", nonnegative=True
        ),
        dispersions=dispersions,
        requirements=requirements,
    )
