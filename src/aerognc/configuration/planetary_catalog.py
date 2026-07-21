"""Strict loader for the fictional planetary catalog used by Mission Designer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from aerognc.astrodynamics.bodies import BodyRole, CircularOrbitBody, PrimaryBody
from aerognc.configuration.loader import (
    ConfigurationError,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _sequence,
    _string,
)


@dataclass(frozen=True, slots=True)
class PlanetaryCatalog:
    """Synthetic central body and selectable Keplerian research worlds."""

    name: str
    safety_scope: str
    primary: PrimaryBody
    bodies: tuple[CircularOrbitBody, ...]

    def body(self, name: str, *, role: BodyRole = "background") -> CircularOrbitBody:
        """Return a named body with a mission-specific role."""
        match = next((body for body in self.bodies if body.name == name), None)
        if match is None:
            raise KeyError(f"planetary catalog has no body named {name!r}")
        return replace(match, role=role)


def _hex_color(value: object, context: str) -> str:
    color = _string(value, context).upper()
    if len(color) != 7 or not color.startswith("#"):
        raise ConfigurationError(f"{context}: expected #RRGGBB colour")
    try:
        int(color[1:], 16)
    except ValueError as error:
        raise ConfigurationError(f"{context}: expected #RRGGBB colour") from error
    return color


def load_planetary_catalog(path: str | Path) -> PlanetaryCatalog:
    """Load a public-safe synthetic planetary system from YAML."""
    root = _load_yaml(Path(path).resolve())
    _keys(root, "planetary_catalog", required={"metadata", "primary", "bodies"})
    metadata = _mapping(root["metadata"], "planetary_catalog.metadata")
    _keys(metadata, "planetary_catalog.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "planetary_catalog.metadata.safety_scope")
    folded = safety_scope.casefold()
    if not all(word in folded for word in ("fictional", "civilian", "synthetic")):
        raise ConfigurationError(
            "planetary catalog safety scope must say fictional, civilian, synthetic"
        )
    primary_data = _mapping(root["primary"], "planetary_catalog.primary")
    _keys(
        primary_data,
        "planetary_catalog.primary",
        required={"name", "gravitational_parameter_m3_s2", "radius_m", "color"},
    )
    primary = PrimaryBody(
        _string(primary_data["name"], "planetary_catalog.primary.name"),
        _number(
            primary_data["gravitational_parameter_m3_s2"],
            "planetary_catalog.primary.gravitational_parameter_m3_s2",
            positive=True,
        ),
        _number(primary_data["radius_m"], "planetary_catalog.primary.radius_m", positive=True),
        _hex_color(primary_data["color"], "planetary_catalog.primary.color"),
    )
    bodies: list[CircularOrbitBody] = []
    for index, value in enumerate(_sequence(root["bodies"], "planetary_catalog.bodies")):
        context = f"planetary_catalog.bodies[{index}]"
        data = _mapping(value, context)
        _keys(
            data,
            context,
            required={"name", "gravitational_parameter_m3_s2", "radius_m", "color", "orbit"},
        )
        orbit = _mapping(data["orbit"], f"{context}.orbit")
        _keys(
            orbit,
            f"{context}.orbit",
            required={
                "semi_major_axis_m",
                "eccentricity",
                "phase_at_epoch_deg",
                "inclination_deg",
                "ascending_node_deg",
                "argument_of_periapsis_deg",
            },
        )
        bodies.append(
            CircularOrbitBody(
                name=_string(data["name"], f"{context}.name"),
                role="background",
                gravitational_parameter_m3_s2=_number(
                    data["gravitational_parameter_m3_s2"],
                    f"{context}.gravitational_parameter_m3_s2",
                    positive=True,
                ),
                radius_m=_number(data["radius_m"], f"{context}.radius_m", positive=True),
                semi_major_axis_m=_number(
                    orbit["semi_major_axis_m"], f"{context}.orbit.semi_major_axis_m", positive=True
                ),
                eccentricity=_number(
                    orbit["eccentricity"], f"{context}.orbit.eccentricity", nonnegative=True
                ),
                phase_at_epoch_rad=np.deg2rad(
                    _number(orbit["phase_at_epoch_deg"], f"{context}.orbit.phase_at_epoch_deg")
                ),
                inclination_rad=np.deg2rad(
                    _number(orbit["inclination_deg"], f"{context}.orbit.inclination_deg")
                ),
                ascending_node_rad=np.deg2rad(
                    _number(orbit["ascending_node_deg"], f"{context}.orbit.ascending_node_deg")
                ),
                argument_of_periapsis_rad=np.deg2rad(
                    _number(
                        orbit["argument_of_periapsis_deg"],
                        f"{context}.orbit.argument_of_periapsis_deg",
                    )
                ),
                color=_hex_color(data["color"], f"{context}.color"),
            )
        )
    if len(bodies) < 3 or len({body.name.casefold() for body in bodies}) != len(bodies):
        raise ConfigurationError("planetary catalog requires at least three uniquely named bodies")
    return PlanetaryCatalog(
        _string(metadata["name"], "planetary_catalog.metadata.name"),
        safety_scope,
        primary,
        tuple(bodies),
    )
