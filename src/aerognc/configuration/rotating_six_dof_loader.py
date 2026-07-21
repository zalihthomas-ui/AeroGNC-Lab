"""Strict composition boundary for rotating-planet quaternion 6-DOF flight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerognc.configuration.loader import _keys, _load_yaml, _mapping, _string
from aerognc.configuration.rotating_flight_loader import (
    RotatingAscentConfiguration,
    load_rotating_ascent_configuration,
)
from aerognc.configuration.six_dof_loader import SixDofConfiguration, load_six_dof_configuration


@dataclass(frozen=True, slots=True)
class RotatingSixDofConfiguration:
    """Validated composition of rigid-body and rotating-planet configurations."""

    source_path: Path
    name: str
    safety_scope: str
    six_dof: SixDofConfiguration
    rotating_planet: RotatingAscentConfiguration
    output_directory: Path


def load_rotating_six_dof_configuration(path: str | Path) -> RotatingSixDofConfiguration:
    """Load a public-safe rotating-planet rigid-body scenario."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "rotating_six_dof",
        required={
            "metadata",
            "six_dof_configuration",
            "rotating_planet_configuration",
            "output_directory",
        },
    )
    metadata = _mapping(root["metadata"], "rotating_six_dof.metadata")
    _keys(metadata, "rotating_six_dof.metadata", required={"name", "safety_scope", "fictional"})
    if metadata["fictional"] is not True:
        raise ValueError("rotating_six_dof.metadata.fictional must be true")
    safety_scope = _string(metadata["safety_scope"], "rotating_six_dof.metadata.safety_scope")
    lowered = safety_scope.lower()
    if "fictional" not in lowered or "civilian" not in lowered:
        raise ValueError("rotating_six_dof safety_scope must state fictional and civilian")

    six_path = source_path.parent / _string(
        root["six_dof_configuration"], "rotating_six_dof.six_dof_configuration"
    )
    rotating_path = source_path.parent / _string(
        root["rotating_planet_configuration"],
        "rotating_six_dof.rotating_planet_configuration",
    )
    six_dof = load_six_dof_configuration(six_path)
    rotating_planet = load_rotating_ascent_configuration(rotating_path)
    if six_dof.base.source_path != rotating_planet.base_configuration.source_path:
        raise ValueError("rotating 6-DOF components must reference the same base vehicle scenario")
    return RotatingSixDofConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "rotating_six_dof.metadata.name"),
        safety_scope=safety_scope,
        six_dof=six_dof,
        rotating_planet=rotating_planet,
        output_directory=Path(
            _string(root["output_directory"], "rotating_six_dof.output_directory")
        ),
    )
