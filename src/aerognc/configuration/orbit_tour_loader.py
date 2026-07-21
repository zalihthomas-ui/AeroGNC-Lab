"""Strict configuration for the fictional civilian orbit-assisted tour."""

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
    _string,
)
from aerognc.configuration.planetary_catalog import PlanetaryCatalog, load_planetary_catalog


@dataclass(frozen=True, slots=True)
class OrbitTourConfiguration:
    """Validated inputs and acceptance limits for a capture-dwell-departure tour."""

    source_path: Path
    name: str
    safety_scope: str
    catalog: PlanetaryCatalog
    departure_body: str
    assist_body: str
    destination_body: str
    departure_time_s: float
    assist_arrival_time_s: float
    destination_arrival_time_s: float
    departure_parking_altitude_m: float
    assist_parking_altitude_m: float
    assist_dwell_revolutions: int
    destination_parking_altitude_m: float
    spacecraft_name: str
    initial_mass_kg: float
    dry_mass_kg: float
    specific_impulse_s: float
    maximum_total_delta_v_mps: float
    minimum_final_mass_kg: float
    first_leg_samples: int
    parking_orbit_samples: int
    second_leg_samples: int
    output_directory: Path


def load_orbit_tour_configuration(path: str | Path) -> OrbitTourConfiguration:
    """Load a public-safe orbit-assisted transfer from strict YAML."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "orbit_tour",
        required={
            "metadata",
            "catalog_path",
            "route",
            "parking_orbits",
            "spacecraft",
            "constraints",
            "simulation",
        },
    )
    metadata = _mapping(root["metadata"], "orbit_tour.metadata")
    _keys(metadata, "orbit_tour.metadata", required={"name", "safety_scope", "fictional"})
    if metadata["fictional"] is not True:
        raise ConfigurationError("orbit_tour.metadata.fictional: must be true")
    safety_scope = _string(metadata["safety_scope"], "orbit_tour.metadata.safety_scope")
    folded = safety_scope.casefold()
    if not all(word in folded for word in ("fictional", "civilian", "synthetic")):
        raise ConfigurationError("orbit-tour safety_scope must say fictional, civilian, synthetic")
    catalog_path = source_path.parent / _string(root["catalog_path"], "orbit_tour.catalog_path")
    catalog = load_planetary_catalog(catalog_path)

    route = _mapping(root["route"], "orbit_tour.route")
    _keys(
        route,
        "orbit_tour.route",
        required={
            "departure_body",
            "assist_body",
            "destination_body",
            "departure_day",
            "assist_arrival_day",
            "destination_arrival_day",
        },
    )
    departure_body = _string(route["departure_body"], "orbit_tour.route.departure_body")
    assist_body = _string(route["assist_body"], "orbit_tour.route.assist_body")
    destination_body = _string(route["destination_body"], "orbit_tour.route.destination_body")
    if len({departure_body, assist_body, destination_body}) != 3:
        raise ConfigurationError("orbit_tour.route: departure, assist, and destination must differ")
    for body_name in (departure_body, assist_body, destination_body):
        try:
            catalog.body(body_name)
        except KeyError as error:
            raise ConfigurationError(
                f"orbit_tour.route: unknown catalog body {body_name!r}"
            ) from error
    departure_day = _number(
        route["departure_day"], "orbit_tour.route.departure_day", nonnegative=True
    )
    assist_day = _number(
        route["assist_arrival_day"], "orbit_tour.route.assist_arrival_day", positive=True
    )
    destination_day = _number(
        route["destination_arrival_day"],
        "orbit_tour.route.destination_arrival_day",
        positive=True,
    )
    if not departure_day < assist_day < destination_day:
        raise ConfigurationError("orbit_tour.route: epochs must be strictly ordered")

    parking = _mapping(root["parking_orbits"], "orbit_tour.parking_orbits")
    _keys(
        parking,
        "orbit_tour.parking_orbits",
        required={
            "departure_altitude_m",
            "assist_altitude_m",
            "assist_dwell_revolutions",
            "destination_altitude_m",
        },
    )
    departure_altitude_m = _number(
        parking["departure_altitude_m"],
        "orbit_tour.parking_orbits.departure_altitude_m",
        positive=True,
    )
    assist_altitude_m = _number(
        parking["assist_altitude_m"],
        "orbit_tour.parking_orbits.assist_altitude_m",
        positive=True,
    )
    destination_altitude_m = _number(
        parking["destination_altitude_m"],
        "orbit_tour.parking_orbits.destination_altitude_m",
        positive=True,
    )
    dwell_revolutions = _integer(
        parking["assist_dwell_revolutions"],
        "orbit_tour.parking_orbits.assist_dwell_revolutions",
        nonnegative=True,
    )
    if dwell_revolutions < 1:
        raise ConfigurationError(
            "orbit_tour.parking_orbits.assist_dwell_revolutions: must be at least 1"
        )

    spacecraft = _mapping(root["spacecraft"], "orbit_tour.spacecraft")
    _keys(
        spacecraft,
        "orbit_tour.spacecraft",
        required={"name", "initial_mass_kg", "dry_mass_kg", "specific_impulse_s"},
    )
    initial_mass_kg = _number(
        spacecraft["initial_mass_kg"], "orbit_tour.spacecraft.initial_mass_kg", positive=True
    )
    dry_mass_kg = _number(
        spacecraft["dry_mass_kg"], "orbit_tour.spacecraft.dry_mass_kg", positive=True
    )
    if dry_mass_kg > initial_mass_kg:
        raise ConfigurationError("orbit_tour.spacecraft: dry mass cannot exceed initial mass")

    constraints = _mapping(root["constraints"], "orbit_tour.constraints")
    _keys(
        constraints,
        "orbit_tour.constraints",
        required={"maximum_total_delta_v_mps", "minimum_final_mass_kg"},
    )
    minimum_final_mass_kg = _number(
        constraints["minimum_final_mass_kg"],
        "orbit_tour.constraints.minimum_final_mass_kg",
        positive=True,
    )
    if minimum_final_mass_kg < dry_mass_kg or minimum_final_mass_kg > initial_mass_kg:
        raise ConfigurationError(
            "orbit_tour.constraints.minimum_final_mass_kg: must lie between dry and initial mass"
        )

    simulation = _mapping(root["simulation"], "orbit_tour.simulation")
    _keys(
        simulation,
        "orbit_tour.simulation",
        required={
            "first_leg_samples",
            "parking_orbit_samples",
            "second_leg_samples",
            "output_directory",
        },
    )
    sample_values = tuple(
        _integer(simulation[key], f"orbit_tour.simulation.{key}", nonnegative=True)
        for key in ("first_leg_samples", "parking_orbit_samples", "second_leg_samples")
    )
    if any(value < 20 or value > 10_000 for value in sample_values):
        raise ConfigurationError("orbit_tour.simulation: each sample count must be 20..10000")
    return OrbitTourConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "orbit_tour.metadata.name"),
        safety_scope=safety_scope,
        catalog=catalog,
        departure_body=departure_body,
        assist_body=assist_body,
        destination_body=destination_body,
        departure_time_s=departure_day * 86_400.0,
        assist_arrival_time_s=assist_day * 86_400.0,
        destination_arrival_time_s=destination_day * 86_400.0,
        departure_parking_altitude_m=departure_altitude_m,
        assist_parking_altitude_m=assist_altitude_m,
        assist_dwell_revolutions=dwell_revolutions,
        destination_parking_altitude_m=destination_altitude_m,
        spacecraft_name=_string(spacecraft["name"], "orbit_tour.spacecraft.name"),
        initial_mass_kg=initial_mass_kg,
        dry_mass_kg=dry_mass_kg,
        specific_impulse_s=_number(
            spacecraft["specific_impulse_s"],
            "orbit_tour.spacecraft.specific_impulse_s",
            positive=True,
        ),
        maximum_total_delta_v_mps=_number(
            constraints["maximum_total_delta_v_mps"],
            "orbit_tour.constraints.maximum_total_delta_v_mps",
            positive=True,
        ),
        minimum_final_mass_kg=minimum_final_mass_kg,
        first_leg_samples=sample_values[0],
        parking_orbit_samples=sample_values[1],
        second_leg_samples=sample_values[2],
        output_directory=Path(
            _string(simulation["output_directory"], "orbit_tour.simulation.output_directory")
        ),
    )
