"""Strict launch-window optimization configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerognc.configuration.loader import (
    ConfigurationError,
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _string,
)
from aerognc.configuration.planetary_catalog import PlanetaryCatalog, load_planetary_catalog


@dataclass(frozen=True, slots=True)
class LaunchWindowConfiguration:
    """Validated direct-transfer search bounds, constraints, and output settings."""

    source_path: Path
    name: str
    safety_scope: str
    catalog: PlanetaryCatalog
    departure_body: str
    destination_body: str
    departure_bounds_s: tuple[float, float]
    arrival_bounds_s: tuple[float, float]
    departure_parking_altitude_m: float
    destination_parking_altitude_m: float
    maximum_c3_m2_s2: float
    maximum_arrival_excess_speed_mps: float
    maximum_total_delta_v_mps: float
    departure_grid_count: int
    arrival_grid_count: int
    maximum_refinement_iterations: int
    epoch_tolerance_s: float
    output_directory: Path


def load_launch_window_configuration(path: str | Path) -> LaunchWindowConfiguration:
    """Load a fictional launch-window search with strict unknown-key rejection."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "launch_window",
        required={
            "metadata",
            "catalog_path",
            "route",
            "parking_orbits",
            "constraints",
            "optimizer",
            "output_directory",
        },
    )
    metadata = _mapping(root["metadata"], "launch_window.metadata")
    _keys(metadata, "launch_window.metadata", required={"name", "safety_scope", "fictional"})
    if metadata["fictional"] is not True:
        raise ConfigurationError("launch_window.metadata.fictional: must be true")
    safety_scope = _string(metadata["safety_scope"], "launch_window.metadata.safety_scope")
    if not all(word in safety_scope.casefold() for word in ("fictional", "civilian", "synthetic")):
        raise ConfigurationError(
            "launch_window.metadata.safety_scope: must say fictional, civilian, synthetic"
        )
    catalog = load_planetary_catalog(
        source_path.parent / _string(root["catalog_path"], "launch_window.catalog_path")
    )
    route = _mapping(root["route"], "launch_window.route")
    _keys(
        route,
        "launch_window.route",
        required={
            "departure_body",
            "destination_body",
            "departure_day_bounds",
            "arrival_day_bounds",
        },
    )
    departure_body = _string(route["departure_body"], "launch_window.route.departure_body")
    destination_body = _string(route["destination_body"], "launch_window.route.destination_body")
    if departure_body == destination_body:
        raise ConfigurationError("launch_window.route: departure and destination must differ")
    for name in (departure_body, destination_body):
        try:
            catalog.body(name)
        except KeyError as error:
            raise ConfigurationError(f"launch_window.route: unknown body {name!r}") from error
    departure_day_bounds = _number_tuple(
        route["departure_day_bounds"], "launch_window.route.departure_day_bounds", length=2
    )
    arrival_day_bounds = _number_tuple(
        route["arrival_day_bounds"], "launch_window.route.arrival_day_bounds", length=2
    )
    if (
        departure_day_bounds[0] < 0.0
        or departure_day_bounds[1] <= departure_day_bounds[0]
        or arrival_day_bounds[1] <= arrival_day_bounds[0]
        or arrival_day_bounds[0] <= departure_day_bounds[0]
    ):
        raise ConfigurationError("launch_window.route: epoch bounds are invalid")

    parking = _mapping(root["parking_orbits"], "launch_window.parking_orbits")
    _keys(
        parking,
        "launch_window.parking_orbits",
        required={"departure_altitude_m", "destination_altitude_m"},
    )
    constraints = _mapping(root["constraints"], "launch_window.constraints")
    _keys(
        constraints,
        "launch_window.constraints",
        required={
            "maximum_c3_m2_s2",
            "maximum_arrival_excess_speed_mps",
            "maximum_total_delta_v_mps",
        },
    )
    optimizer = _mapping(root["optimizer"], "launch_window.optimizer")
    _keys(
        optimizer,
        "launch_window.optimizer",
        required={
            "departure_grid_count",
            "arrival_grid_count",
            "maximum_refinement_iterations",
            "epoch_tolerance_s",
        },
    )
    integer_values = tuple(
        _integer(optimizer[key], f"launch_window.optimizer.{key}", nonnegative=True)
        for key in (
            "departure_grid_count",
            "arrival_grid_count",
            "maximum_refinement_iterations",
        )
    )
    if integer_values[0] < 3 or integer_values[1] < 3 or integer_values[2] < 1:
        raise ConfigurationError("launch_window.optimizer: grid >=3 and iterations >=1 required")
    return LaunchWindowConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "launch_window.metadata.name"),
        safety_scope=safety_scope,
        catalog=catalog,
        departure_body=departure_body,
        destination_body=destination_body,
        departure_bounds_s=(
            departure_day_bounds[0] * 86_400.0,
            departure_day_bounds[1] * 86_400.0,
        ),
        arrival_bounds_s=(
            arrival_day_bounds[0] * 86_400.0,
            arrival_day_bounds[1] * 86_400.0,
        ),
        departure_parking_altitude_m=_number(
            parking["departure_altitude_m"],
            "launch_window.parking_orbits.departure_altitude_m",
            positive=True,
        ),
        destination_parking_altitude_m=_number(
            parking["destination_altitude_m"],
            "launch_window.parking_orbits.destination_altitude_m",
            positive=True,
        ),
        maximum_c3_m2_s2=_number(
            constraints["maximum_c3_m2_s2"],
            "launch_window.constraints.maximum_c3_m2_s2",
            positive=True,
        ),
        maximum_arrival_excess_speed_mps=_number(
            constraints["maximum_arrival_excess_speed_mps"],
            "launch_window.constraints.maximum_arrival_excess_speed_mps",
            positive=True,
        ),
        maximum_total_delta_v_mps=_number(
            constraints["maximum_total_delta_v_mps"],
            "launch_window.constraints.maximum_total_delta_v_mps",
            positive=True,
        ),
        departure_grid_count=integer_values[0],
        arrival_grid_count=integer_values[1],
        maximum_refinement_iterations=integer_values[2],
        epoch_tolerance_s=_number(
            optimizer["epoch_tolerance_s"],
            "launch_window.optimizer.epoch_tolerance_s",
            positive=True,
        ),
        output_directory=Path(_string(root["output_directory"], "launch_window.output_directory")),
    )
