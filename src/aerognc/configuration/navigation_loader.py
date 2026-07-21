"""Validated sensor and vertical-navigation demonstration configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aerognc.configuration.loader import (
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _matrix3,
    _number,
    _number_tuple,
    _sequence,
    _string,
    load_three_dof_configuration,
)
from aerognc.configuration.models import ThreeDofConfiguration
from aerognc.gnc.ekf import VerticalFilterTuning
from aerognc.vehicle.sensors import SensorErrorParameters


@dataclass(frozen=True, slots=True)
class NavigationDemoConfiguration:
    """Truth scenario, seeded sensors, filter tuning, and output location."""

    source_path: Path
    name: str
    safety_scope: str
    output_directory: Path
    random_seed: int
    base: ThreeDofConfiguration
    accelerometer: SensorErrorParameters
    barometer: SensorErrorParameters
    gnss: SensorErrorParameters
    filter_tuning: VerticalFilterTuning
    initial_state: tuple[float, float, float]
    initial_covariance: tuple[tuple[float, float, float], ...]


def _dropout_intervals(value: object, context: str) -> tuple[tuple[float, float], ...]:
    rows = _sequence(value, context)
    result: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        pair = _number_tuple(row, f"{context}[{index}]", length=2)
        result.append((pair[0], pair[1]))
    return tuple(result)


def _sensor_parameters(value: object, context: str, dimension: int) -> SensorErrorParameters:
    data = _mapping(value, context)
    _keys(
        data,
        context,
        required={
            "sample_rate_hz",
            "noise_std",
            "constant_bias",
            "bias_drift_std_per_sqrt_s",
            "quantisation",
            "delay_s",
            "dropout_probability",
            "dropout_intervals_s",
        },
    )
    return SensorErrorParameters(
        sample_rate_hz=_number(data["sample_rate_hz"], f"{context}.sample_rate_hz", positive=True),
        noise_std=_number_tuple(data["noise_std"], f"{context}.noise_std", length=dimension),
        constant_bias=_number_tuple(
            data["constant_bias"], f"{context}.constant_bias", length=dimension
        ),
        bias_drift_std_per_sqrt_s=_number_tuple(
            data["bias_drift_std_per_sqrt_s"],
            f"{context}.bias_drift_std_per_sqrt_s",
            length=dimension,
        ),
        quantisation=_number_tuple(
            data["quantisation"], f"{context}.quantisation", length=dimension
        ),
        delay_s=_number(data["delay_s"], f"{context}.delay_s", nonnegative=True),
        dropout_probability=_number(
            data["dropout_probability"], f"{context}.dropout_probability", nonnegative=True
        ),
        dropout_intervals_s=_dropout_intervals(
            data["dropout_intervals_s"], f"{context}.dropout_intervals_s"
        ),
    )


def load_navigation_demo_configuration(path: str | Path) -> NavigationDemoConfiguration:
    """Load one synthetic navigation demonstration."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "navigation_demo",
        required={"metadata", "base_scenario_file", "simulation", "sensors", "filter"},
    )
    metadata = _mapping(root["metadata"], "navigation_demo.metadata")
    _keys(metadata, "navigation_demo.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "navigation_demo.metadata.safety_scope")
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        raise ValueError("navigation_demo safety_scope must state fictional and civilian")
    base_path = Path(_string(root["base_scenario_file"], "navigation_demo.base_scenario_file"))
    if not base_path.is_absolute():
        base_path = source_path.parent / base_path
    base = load_three_dof_configuration(base_path)
    simulation = _mapping(root["simulation"], "navigation_demo.simulation")
    _keys(
        simulation,
        "navigation_demo.simulation",
        required={"output_directory", "random_seed"},
    )
    sensors = _mapping(root["sensors"], "navigation_demo.sensors")
    _keys(sensors, "navigation_demo.sensors", required={"accelerometer", "barometer", "gnss"})
    filter_data = _mapping(root["filter"], "navigation_demo.filter")
    _keys(
        filter_data,
        "navigation_demo.filter",
        required={
            "initial_state",
            "initial_covariance",
            "acceleration_process_std_mps2",
            "bias_random_walk_std_mps2_per_sqrt_s",
            "barometer_std_m",
            "gnss_altitude_std_m",
            "gnss_vertical_velocity_std_mps",
        },
    )
    covariance = _matrix3(
        filter_data["initial_covariance"], "navigation_demo.filter.initial_covariance"
    )
    return NavigationDemoConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "navigation_demo.metadata.name"),
        safety_scope=safety_scope,
        output_directory=Path(
            _string(simulation["output_directory"], "navigation_demo.simulation.output_directory")
        ),
        random_seed=_integer(
            simulation["random_seed"], "navigation_demo.simulation.random_seed", nonnegative=True
        ),
        base=base,
        accelerometer=_sensor_parameters(
            sensors["accelerometer"], "navigation_demo.sensors.accelerometer", 3
        ),
        barometer=_sensor_parameters(sensors["barometer"], "navigation_demo.sensors.barometer", 1),
        gnss=_sensor_parameters(sensors["gnss"], "navigation_demo.sensors.gnss", 6),
        filter_tuning=VerticalFilterTuning(
            _number(
                filter_data["acceleration_process_std_mps2"],
                "navigation_demo.filter.acceleration_process_std_mps2",
                positive=True,
            ),
            _number(
                filter_data["bias_random_walk_std_mps2_per_sqrt_s"],
                "navigation_demo.filter.bias_random_walk_std_mps2_per_sqrt_s",
                positive=True,
            ),
            _number(
                filter_data["barometer_std_m"],
                "navigation_demo.filter.barometer_std_m",
                positive=True,
            ),
            _number(
                filter_data["gnss_altitude_std_m"],
                "navigation_demo.filter.gnss_altitude_std_m",
                positive=True,
            ),
            _number(
                filter_data["gnss_vertical_velocity_std_mps"],
                "navigation_demo.filter.gnss_vertical_velocity_std_mps",
                positive=True,
            ),
        ),
        initial_state=cast(
            tuple[float, float, float],
            _number_tuple(
                filter_data["initial_state"], "navigation_demo.filter.initial_state", length=3
            ),
        ),
        initial_covariance=covariance,
    )
