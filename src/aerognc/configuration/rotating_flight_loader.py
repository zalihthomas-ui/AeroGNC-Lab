"""Strict configuration boundary for rotating-planet ascent studies."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import yaml

from aerognc.configuration.loader import ConfigurationError, load_three_dof_configuration
from aerognc.configuration.models import ThreeDofConfiguration
from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.mathematics.geodesy import GeodeticPosition, LaunchSite, ReferenceEllipsoid


@dataclass(frozen=True, slots=True)
class RotatingAscentConfiguration:
    """Validated rotating-planet wrapper around a fictional 3-DOF vehicle case."""

    source_path: Path
    safety_scope: str
    base_configuration: ThreeDofConfiguration
    planet: RotatingOblatePlanet
    launch_site: LaunchSite
    output_directory: Path


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{context}: expected a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact_keys(mapping: Mapping[str, object], expected: set[str], context: str) -> None:
    missing = expected - mapping.keys()
    unknown = mapping.keys() - expected
    if missing:
        raise ConfigurationError(f"{context}: missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigurationError(f"{context}: unknown keys: {', '.join(sorted(unknown))}")


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context}: expected a number")
    result = float(value)
    if not np.isfinite(result):
        raise ConfigurationError(f"{context}: must be finite")
    return result


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context}: expected nonempty text")
    return value.strip()


def load_rotating_ascent_configuration(path: str | Path) -> RotatingAscentConfiguration:
    """Load a synthetic rotating-oblate-planet ascent scenario."""
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {source_path}")
    try:
        loaded: object = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {source_path}: {error}") from error
    root = _mapping(loaded, str(source_path))
    _exact_keys(
        root,
        {"metadata", "base_configuration", "planet", "launch_site", "output_directory"},
        "rotating_ascent",
    )

    metadata = _mapping(root["metadata"], "rotating_ascent.metadata")
    _exact_keys(metadata, {"safety_scope", "fictional"}, "rotating_ascent.metadata")
    if metadata["fictional"] is not True:
        raise ConfigurationError(
            "rotating_ascent.metadata.fictional: public scenarios must be fictional"
        )
    safety_scope = _text(metadata["safety_scope"], "rotating_ascent.metadata.safety_scope")

    base_path = source_path.parent / _text(
        root["base_configuration"], "rotating_ascent.base_configuration"
    )
    base_configuration = load_three_dof_configuration(base_path)

    planet_data = _mapping(root["planet"], "rotating_ascent.planet")
    _exact_keys(
        planet_data,
        {
            "name",
            "fictional",
            "semi_major_axis_m",
            "flattening",
            "gravitational_parameter_m3ps2",
            "rotation_rate_radps",
            "j2",
        },
        "rotating_ascent.planet",
    )
    if planet_data["fictional"] is not True:
        raise ConfigurationError("rotating_ascent.planet.fictional: must be true")
    ellipsoid = ReferenceEllipsoid(
        _number(planet_data["semi_major_axis_m"], "rotating_ascent.planet.semi_major_axis_m"),
        _number(planet_data["flattening"], "rotating_ascent.planet.flattening"),
    )
    planet = RotatingOblatePlanet(
        name=_text(planet_data["name"], "rotating_ascent.planet.name"),
        ellipsoid=ellipsoid,
        gravitational_parameter_m3ps2=_number(
            planet_data["gravitational_parameter_m3ps2"],
            "rotating_ascent.planet.gravitational_parameter_m3ps2",
        ),
        rotation_rate_radps=_number(
            planet_data["rotation_rate_radps"], "rotating_ascent.planet.rotation_rate_radps"
        ),
        j2=_number(planet_data["j2"], "rotating_ascent.planet.j2"),
    )

    site_data = _mapping(root["launch_site"], "rotating_ascent.launch_site")
    _exact_keys(
        site_data,
        {"latitude_deg", "longitude_deg", "altitude_m", "azimuth_deg", "elevation_deg"},
        "rotating_ascent.launch_site",
    )
    latitude_deg = _number(site_data["latitude_deg"], "rotating_ascent.launch_site.latitude_deg")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ConfigurationError("rotating_ascent.launch_site.latitude_deg: outside [-90, 90]")
    launch_site = LaunchSite(
        geodetic=GeodeticPosition(
            latitude_rad=np.deg2rad(latitude_deg),
            longitude_rad=np.deg2rad(
                _number(site_data["longitude_deg"], "rotating_ascent.launch_site.longitude_deg")
            ),
            altitude_m=_number(site_data["altitude_m"], "rotating_ascent.launch_site.altitude_m"),
        ),
        azimuth_rad=np.deg2rad(
            _number(site_data["azimuth_deg"], "rotating_ascent.launch_site.azimuth_deg")
        ),
        elevation_rad=np.deg2rad(
            _number(site_data["elevation_deg"], "rotating_ascent.launch_site.elevation_deg")
        ),
    )
    output_directory = Path(_text(root["output_directory"], "rotating_ascent.output_directory"))
    return RotatingAscentConfiguration(
        source_path=source_path,
        safety_scope=safety_scope,
        base_configuration=base_configuration,
        planet=planet,
        launch_site=launch_site,
        output_directory=output_directory,
    )
