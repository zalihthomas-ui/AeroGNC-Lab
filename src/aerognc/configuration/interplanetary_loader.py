"""Strict configuration for fictional civilian interplanetary missions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from aerognc.astrodynamics.bodies import BodyRole, CircularOrbitBody, PrimaryBody
from aerognc.astrodynamics.maneuvers import (
    FiniteBurn,
    ImpulsiveManeuver,
    ManeuverFrame,
    SpacecraftManeuver,
)
from aerognc.astrodynamics.perturbations import PerturbationSettings
from aerognc.configuration.loader import (
    ConfigurationError,
    _boolean,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _sequence,
    _string,
)

SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class SpacecraftInjection:
    """Spacecraft state at the departure body's sphere-of-influence boundary."""

    name: str
    mass_kg: float
    reference_body: str
    position_offset_rtn_m: tuple[float, float, float]
    velocity_offset_rtn_mps: tuple[float, float, float]
    dry_mass_kg: float


@dataclass(frozen=True, slots=True)
class InterplanetaryConfiguration:
    """Complete restricted N-body mission and visualisation definition."""

    source_path: Path
    name: str
    description: str
    safety_scope: str
    primary: PrimaryBody
    bodies: tuple[CircularOrbitBody, ...]
    spacecraft: SpacecraftInjection
    duration_s: float
    step_s: float
    assist_encounter_radius_m: float
    destination_arrival_radius_m: float
    output_directory: Path
    snapshot_time_s: float
    maneuvers: tuple[SpacecraftManeuver, ...] = ()
    force_model: PerturbationSettings = field(default_factory=PerturbationSettings)

    def body_with_role(self, role: BodyRole) -> CircularOrbitBody:
        """Return the unique body assigned a mission role."""
        body = next((candidate for candidate in self.bodies if candidate.role == role), None)
        if body is None:
            raise KeyError(f"mission has no body with role: {role}")
        return body


def _color(value: object, context: str) -> str:
    color = _string(value, context)
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None:
        raise ConfigurationError(f"{context}: expected a six-digit hexadecimal colour")
    return color.upper()


def _load_body(value: object, index: int) -> CircularOrbitBody:
    context = f"interplanetary.bodies[{index}]"
    data = _mapping(value, context)
    _keys(
        data,
        context,
        required={
            "name",
            "role",
            "gravitational_parameter_m3_s2",
            "radius_m",
            "color",
            "orbit",
        },
    )
    role_value = _string(data["role"], f"{context}.role")
    if role_value not in {"departure", "assist", "destination", "background"}:
        raise ConfigurationError(
            f"{context}.role: expected departure, assist, destination, or background"
        )
    orbit = _mapping(data["orbit"], f"{context}.orbit")
    _keys(
        orbit,
        f"{context}.orbit",
        required={
            "semi_major_axis_m",
            "phase_at_epoch_deg",
            "inclination_deg",
            "ascending_node_deg",
        },
        optional={"eccentricity", "argument_of_periapsis_deg"},
    )
    return CircularOrbitBody(
        name=_string(data["name"], f"{context}.name"),
        role=cast(BodyRole, role_value),
        gravitational_parameter_m3_s2=_number(
            data["gravitational_parameter_m3_s2"],
            f"{context}.gravitational_parameter_m3_s2",
            positive=True,
        ),
        radius_m=_number(data["radius_m"], f"{context}.radius_m", positive=True),
        semi_major_axis_m=_number(
            orbit["semi_major_axis_m"], f"{context}.orbit.semi_major_axis_m", positive=True
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
        color=_color(data["color"], f"{context}.color"),
        eccentricity=_number(
            orbit.get("eccentricity", 0.0),
            f"{context}.orbit.eccentricity",
            nonnegative=True,
        ),
        argument_of_periapsis_rad=np.deg2rad(
            _number(
                orbit.get("argument_of_periapsis_deg", 0.0),
                f"{context}.orbit.argument_of_periapsis_deg",
            )
        ),
    )


def _load_maneuver(value: object, index: int) -> SpacecraftManeuver:
    context = f"interplanetary.maneuvers[{index}]"
    data = _mapping(value, context)
    maneuver_type = _string(data.get("type"), f"{context}.type")
    if maneuver_type == "impulse":
        _keys(
            data,
            context,
            required={"name", "type", "epoch_days", "delta_velocity_mps"},
            optional={"frame", "specific_impulse_s"},
        )
        delta_velocity = cast(
            tuple[float, float, float],
            _number_tuple(data["delta_velocity_mps"], f"{context}.delta_velocity_mps", length=3),
        )
        frame = _string(data.get("frame", "rtn"), f"{context}.frame")
        if frame not in {"inertial", "rtn"}:
            raise ConfigurationError(f"{context}.frame: expected inertial or rtn")
        return ImpulsiveManeuver(
            name=_string(data["name"], f"{context}.name"),
            epoch_s=_number(data["epoch_days"], f"{context}.epoch_days", nonnegative=True)
            * SECONDS_PER_DAY,
            delta_velocity_mps=delta_velocity,
            frame=cast(ManeuverFrame, frame),
            specific_impulse_s=_number(
                data.get("specific_impulse_s", 320.0),
                f"{context}.specific_impulse_s",
                positive=True,
            ),
        )
    if maneuver_type == "finite_burn":
        _keys(
            data,
            context,
            required={"name", "type", "start_time_days", "duration_s", "thrust_n", "direction"},
            optional={"frame", "specific_impulse_s"},
        )
        direction = cast(
            tuple[float, float, float],
            _number_tuple(data["direction"], f"{context}.direction", length=3),
        )
        frame = _string(data.get("frame", "rtn"), f"{context}.frame")
        if frame not in {"inertial", "rtn"}:
            raise ConfigurationError(f"{context}.frame: expected inertial or rtn")
        return FiniteBurn(
            name=_string(data["name"], f"{context}.name"),
            start_time_s=_number(
                data["start_time_days"], f"{context}.start_time_days", nonnegative=True
            )
            * SECONDS_PER_DAY,
            duration_s=_number(data["duration_s"], f"{context}.duration_s", positive=True),
            thrust_n=_number(data["thrust_n"], f"{context}.thrust_n", positive=True),
            direction=direction,
            frame=cast(ManeuverFrame, frame),
            specific_impulse_s=_number(
                data.get("specific_impulse_s", 320.0),
                f"{context}.specific_impulse_s",
                positive=True,
            ),
        )
    raise ConfigurationError(f"{context}.type: expected impulse or finite_burn")


def load_interplanetary_configuration(path: str | Path) -> InterplanetaryConfiguration:
    """Load and validate one synthetic restricted N-body gravity-assist scenario."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "interplanetary",
        required={"metadata", "primary", "bodies", "spacecraft", "mission"},
        optional={"force_model", "maneuvers"},
    )
    metadata = _mapping(root["metadata"], "interplanetary.metadata")
    _keys(
        metadata,
        "interplanetary.metadata",
        required={"name", "description", "safety_scope"},
    )
    safety_scope = _string(metadata["safety_scope"], "interplanetary.metadata.safety_scope")
    safety_words = safety_scope.casefold()
    if not all(word in safety_words for word in ("fictional", "civilian", "synthetic")):
        raise ConfigurationError(
            "interplanetary.metadata.safety_scope: must state fictional, civilian, and synthetic"
        )

    primary_data = _mapping(root["primary"], "interplanetary.primary")
    _keys(
        primary_data,
        "interplanetary.primary",
        required={"name", "gravitational_parameter_m3_s2", "radius_m", "color"},
    )
    primary = PrimaryBody(
        name=_string(primary_data["name"], "interplanetary.primary.name"),
        gravitational_parameter_m3_s2=_number(
            primary_data["gravitational_parameter_m3_s2"],
            "interplanetary.primary.gravitational_parameter_m3_s2",
            positive=True,
        ),
        radius_m=_number(
            primary_data["radius_m"], "interplanetary.primary.radius_m", positive=True
        ),
        color=_color(primary_data["color"], "interplanetary.primary.color"),
    )
    body_rows = _sequence(root["bodies"], "interplanetary.bodies")
    if len(body_rows) < 3:
        raise ConfigurationError("interplanetary.bodies: at least three bodies are required")
    bodies = tuple(_load_body(row, index) for index, row in enumerate(body_rows))
    names = [body.name.casefold() for body in bodies]
    if len(names) != len(set(names)):
        raise ConfigurationError("interplanetary.bodies: names must be unique")
    for role in ("departure", "assist", "destination"):
        if sum(body.role == role for body in bodies) != 1:
            raise ConfigurationError(f"interplanetary.bodies: exactly one {role} body is required")

    spacecraft_data = _mapping(root["spacecraft"], "interplanetary.spacecraft")
    _keys(
        spacecraft_data,
        "interplanetary.spacecraft",
        required={
            "name",
            "mass_kg",
            "reference_body",
            "position_offset_rtn_m",
            "velocity_offset_rtn_mps",
        },
        optional={"dry_mass_kg"},
    )
    position_offset = cast(
        tuple[float, float, float],
        _number_tuple(
            spacecraft_data["position_offset_rtn_m"],
            "interplanetary.spacecraft.position_offset_rtn_m",
            length=3,
        ),
    )
    velocity_offset = cast(
        tuple[float, float, float],
        _number_tuple(
            spacecraft_data["velocity_offset_rtn_mps"],
            "interplanetary.spacecraft.velocity_offset_rtn_mps",
            length=3,
        ),
    )
    spacecraft = SpacecraftInjection(
        name=_string(spacecraft_data["name"], "interplanetary.spacecraft.name"),
        mass_kg=_number(
            spacecraft_data["mass_kg"], "interplanetary.spacecraft.mass_kg", positive=True
        ),
        reference_body=_string(
            spacecraft_data["reference_body"], "interplanetary.spacecraft.reference_body"
        ),
        position_offset_rtn_m=position_offset,
        velocity_offset_rtn_mps=velocity_offset,
        dry_mass_kg=_number(
            spacecraft_data.get("dry_mass_kg", spacecraft_data["mass_kg"]),
            "interplanetary.spacecraft.dry_mass_kg",
            positive=True,
        ),
    )
    if spacecraft.dry_mass_kg > spacecraft.mass_kg:
        raise ConfigurationError("interplanetary.spacecraft.dry_mass_kg cannot exceed mass_kg")
    departure = next(body for body in bodies if body.role == "departure")
    if spacecraft.reference_body != departure.name:
        raise ConfigurationError(
            "interplanetary.spacecraft.reference_body must be the departure body"
        )
    if np.linalg.norm(position_offset) <= departure.radius_m:
        raise ConfigurationError(
            "interplanetary.spacecraft.position_offset_rtn_m must start outside the departure body"
        )

    mission = _mapping(root["mission"], "interplanetary.mission")
    _keys(
        mission,
        "interplanetary.mission",
        required={
            "duration_days",
            "step_s",
            "assist_encounter_radius_m",
            "destination_arrival_radius_m",
            "output_directory",
            "snapshot_time_days",
        },
    )
    duration_s = (
        _number(mission["duration_days"], "interplanetary.mission.duration_days", positive=True)
        * SECONDS_PER_DAY
    )
    step_s = _number(mission["step_s"], "interplanetary.mission.step_s", positive=True)
    if step_s > 21_600.0:
        raise ConfigurationError("interplanetary.mission.step_s cannot exceed six hours")
    if duration_s > 20.0 * 365.25 * SECONDS_PER_DAY:
        raise ConfigurationError("interplanetary mission duration cannot exceed twenty years")
    assist = next(body for body in bodies if body.role == "assist")
    destination = next(body for body in bodies if body.role == "destination")
    encounter_radius_m = _number(
        mission["assist_encounter_radius_m"],
        "interplanetary.mission.assist_encounter_radius_m",
        positive=True,
    )
    arrival_radius_m = _number(
        mission["destination_arrival_radius_m"],
        "interplanetary.mission.destination_arrival_radius_m",
        positive=True,
    )
    if encounter_radius_m <= assist.radius_m:
        raise ConfigurationError("assist encounter radius must exceed the assist-body radius")
    if arrival_radius_m <= destination.radius_m:
        raise ConfigurationError("destination arrival radius must exceed the body radius")
    snapshot_time_s = (
        _number(
            mission["snapshot_time_days"],
            "interplanetary.mission.snapshot_time_days",
            nonnegative=True,
        )
        * SECONDS_PER_DAY
    )
    if snapshot_time_s > duration_s:
        raise ConfigurationError("snapshot time must lie inside the mission duration")
    maneuvers = tuple(
        _load_maneuver(row, index)
        for index, row in enumerate(
            _sequence(root.get("maneuvers", []), "interplanetary.maneuvers")
        )
    )
    maneuver_names = [maneuver.name.casefold() for maneuver in maneuvers]
    if len(maneuver_names) != len(set(maneuver_names)):
        raise ConfigurationError("interplanetary.maneuvers: maneuver names must be unique")
    for maneuver in maneuvers:
        start_time_s = (
            maneuver.epoch_s if isinstance(maneuver, ImpulsiveManeuver) else maneuver.start_time_s
        )
        end_time_s = (
            start_time_s if isinstance(maneuver, ImpulsiveManeuver) else maneuver.end_time_s
        )
        if start_time_s > duration_s or end_time_s > duration_s:
            raise ConfigurationError(
                "interplanetary.maneuvers: all maneuvers must lie in mission duration"
            )
    force_data = _mapping(root.get("force_model", {}), "interplanetary.force_model")
    _keys(
        force_data,
        "interplanetary.force_model",
        required=set(),
        optional={
            "j2",
            "reference_radius_m",
            "radiation_area_m2",
            "reflectivity_coefficient",
            "include_relativity",
        },
    )
    force_model = PerturbationSettings(
        j2=_number(
            force_data.get("j2", 0.0),
            "interplanetary.force_model.j2",
            nonnegative=True,
        ),
        reference_radius_m=_number(
            force_data.get("reference_radius_m", primary.radius_m),
            "interplanetary.force_model.reference_radius_m",
            positive=True,
        ),
        radiation_area_m2=_number(
            force_data.get("radiation_area_m2", 0.0),
            "interplanetary.force_model.radiation_area_m2",
            nonnegative=True,
        ),
        reflectivity_coefficient=_number(
            force_data.get("reflectivity_coefficient", 1.0),
            "interplanetary.force_model.reflectivity_coefficient",
            positive=True,
        ),
        include_relativity=_boolean(
            force_data.get("include_relativity", False),
            "interplanetary.force_model.include_relativity",
        ),
    )
    return InterplanetaryConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "interplanetary.metadata.name"),
        description=_string(metadata["description"], "interplanetary.metadata.description"),
        safety_scope=safety_scope,
        primary=primary,
        bodies=bodies,
        spacecraft=spacecraft,
        duration_s=duration_s,
        step_s=step_s,
        assist_encounter_radius_m=encounter_radius_m,
        destination_arrival_radius_m=arrival_radius_m,
        output_directory=Path(
            _string(mission["output_directory"], "interplanetary.mission.output_directory")
        ),
        snapshot_time_s=snapshot_time_s,
        maneuvers=maneuvers,
        force_model=force_model,
    )
