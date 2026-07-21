"""Strict configuration boundary for the public-safe satellite orbit sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from aerognc.configuration.loader import (
    ConfigurationError,
    _boolean,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _sequence,
    _string,
)

OrbitModelName = Literal[
    "free",
    "two_body",
    "restricted_three_body",
    "full_n_body",
    "perturbed_decay",
]
OrbitSpeedMode = Literal["circular", "escape", "custom"]
ORBIT_MODEL_NAMES: tuple[OrbitModelName, ...] = (
    "free",
    "two_body",
    "restricted_three_body",
    "full_n_body",
    "perturbed_decay",
)
ORBIT_SPEED_MODES: tuple[OrbitSpeedMode, ...] = ("circular", "escape", "custom")


@dataclass(frozen=True, slots=True)
class OrbitPrimaryDefinition:
    """Spherical rotating synthetic primary used by the sandbox."""

    name: str
    gravitational_parameter_m3_s2: float
    radius_m: float
    j2: float
    rotation_rate_radps: float
    color: str

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.gravitational_parameter_m3_s2,
                self.radius_m,
                self.j2,
                self.rotation_rate_radps,
            ]
        )
        if not self.name.strip() or not np.all(np.isfinite(values)):
            raise ValueError("orbit primary values must be named and finite")
        if self.gravitational_parameter_m3_s2 <= 0.0 or self.radius_m <= 0.0:
            raise ValueError("orbit primary gravity and radius must be positive")
        if not 0.0 <= self.j2 < 0.1 or self.rotation_rate_radps < 0.0:
            raise ValueError("orbit primary J2/rotation values are outside the model domain")


@dataclass(frozen=True, slots=True)
class OrbitSecondaryDefinition:
    """Configured circular secondary for restricted/full N-body examples."""

    name: str
    gravitational_parameter_m3_s2: float
    radius_m: float
    orbital_radius_m: float
    phase_at_epoch_rad: float
    inclination_rad: float
    color: str

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.gravitational_parameter_m3_s2,
                self.radius_m,
                self.orbital_radius_m,
                self.phase_at_epoch_rad,
                self.inclination_rad,
            ]
        )
        if not self.name.strip() or not np.all(np.isfinite(values)):
            raise ValueError("orbit secondary values must be named and finite")
        if np.any(values[:3] <= 0.0) or abs(self.inclination_rad) > np.pi:
            raise ValueError("orbit secondary geometry is outside the model domain")


@dataclass(frozen=True, slots=True)
class SatelliteDefinition:
    """Satellite mass, drag, and ideal correction-engine properties."""

    name: str
    initial_mass_kg: float
    dry_mass_kg: float
    drag_area_m2: float
    drag_coefficient: float
    correction_specific_impulse_s: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.initial_mass_kg,
                self.dry_mass_kg,
                self.drag_area_m2,
                self.drag_coefficient,
                self.correction_specific_impulse_s,
            ]
        )
        if not self.name.strip() or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("satellite values must be named, positive, and finite")
        if self.dry_mass_kg >= self.initial_mass_kg:
            raise ValueError("satellite dry mass must be lower than initial mass")


@dataclass(frozen=True, slots=True)
class OrbitInitialCondition:
    """User-readable initial orbit geometry."""

    altitude_m: float
    speed_mode: OrbitSpeedMode
    custom_speed_mps: float
    inclination_rad: float
    ascending_node_rad: float
    phase_rad: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.altitude_m,
                self.custom_speed_mps,
                self.inclination_rad,
                self.ascending_node_rad,
                self.phase_rad,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("initial orbit values must be finite")
        if self.altitude_m <= 0.0 or self.custom_speed_mps < 0.0:
            raise ValueError("initial altitude must be positive and custom speed nonnegative")
        if self.speed_mode not in ORBIT_SPEED_MODES:
            raise ValueError(f"speed_mode must be one of {ORBIT_SPEED_MODES}")
        if not 0.0 <= self.inclination_rad <= np.pi:
            raise ValueError("initial inclination must lie in [0, pi]")


@dataclass(frozen=True, slots=True)
class OrbitCorrectionDefinition:
    """Optional idealized recircularization policy; disabled by default."""

    enabled: bool
    trigger_altitude_m: float
    maximum_burns: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.trigger_altitude_m) or self.trigger_altitude_m <= 0.0:
            raise ValueError("correction trigger altitude must be positive and finite")
        if (
            isinstance(self.maximum_burns, bool)
            or not isinstance(self.maximum_burns, int)
            or self.maximum_burns < 0
        ):
            raise ValueError("maximum correction burns must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class OrbitSandboxConfiguration:
    """Complete near-planet orbit-sandbox configuration."""

    source_path: Path
    name: str
    safety_scope: str
    model: OrbitModelName
    primary: OrbitPrimaryDefinition
    secondaries: tuple[OrbitSecondaryDefinition, ...]
    satellite: SatelliteDefinition
    initial: OrbitInitialCondition
    correction: OrbitCorrectionDefinition
    duration_s: float
    integration_step_s: float
    output_step_s: float
    reentry_altitude_m: float
    escape_radius_multiplier: float
    atmosphere_density_scale: float
    output_directory: Path

    def __post_init__(self) -> None:
        if self.model not in ORBIT_MODEL_NAMES:
            raise ValueError(f"orbit model must be one of {ORBIT_MODEL_NAMES}")
        if not self.secondaries:
            raise ValueError("orbit sandbox requires at least one synthetic secondary")
        names = [item.name.casefold() for item in self.secondaries]
        if len(names) != len(set(names)):
            raise ValueError("orbit secondary names must be unique")
        for body in self.secondaries:
            if body.orbital_radius_m <= self.primary.radius_m + body.radius_m:
                raise ValueError(f"secondary {body.name} intersects the primary")
        timing = np.array([self.duration_s, self.integration_step_s, self.output_step_s])
        if not np.all(np.isfinite(timing)) or np.any(timing <= 0.0):
            raise ValueError("orbit duration and steps must be positive and finite")
        if self.duration_s > 366.0 * 86_400.0:
            raise ValueError("interactive orbit duration cannot exceed 366 days")
        if self.integration_step_s > 300.0 or self.integration_step_s > self.output_step_s:
            raise ValueError(
                "orbit integration step must be at most 300 s and no larger than output step"
            )
        ratio = self.output_step_s / self.integration_step_s
        if not np.isclose(ratio, round(ratio), atol=1.0e-10):
            raise ValueError("output_step_s must be an integer multiple of integration_step_s")
        if not np.isfinite(self.reentry_altitude_m) or self.reentry_altitude_m < 0.0:
            raise ValueError("reentry altitude must be finite and nonnegative")
        if self.initial.altitude_m <= self.reentry_altitude_m:
            raise ValueError("initial altitude must exceed the reentry threshold")
        if not np.isfinite(self.escape_radius_multiplier) or self.escape_radius_multiplier <= 1.0:
            raise ValueError("escape radius multiplier must exceed one")
        if not np.isfinite(self.atmosphere_density_scale) or not (
            0.0 <= self.atmosphere_density_scale <= 1.0e6
        ):
            raise ValueError("atmosphere density scale must lie in [0, 1e6]")


def _color(value: object, context: str) -> str:
    color = _string(value, context).upper()
    if len(color) != 7 or not color.startswith("#"):
        raise ConfigurationError(f"{context}: expected #RRGGBB colour")
    try:
        int(color[1:], 16)
    except ValueError as error:
        raise ConfigurationError(f"{context}: expected #RRGGBB colour") from error
    return color


def _orbit_model(value: object, context: str) -> OrbitModelName:
    model = _string(value, context)
    if model not in ORBIT_MODEL_NAMES:
        raise ConfigurationError(f"{context}: expected one of {ORBIT_MODEL_NAMES}")
    return model


def _speed_mode(value: object, context: str) -> OrbitSpeedMode:
    mode = _string(value, context)
    if mode not in ORBIT_SPEED_MODES:
        raise ConfigurationError(f"{context}: expected one of {ORBIT_SPEED_MODES}")
    return mode


def load_orbit_sandbox_configuration(path: str | Path) -> OrbitSandboxConfiguration:
    """Load one strict synthetic orbit-sandbox YAML file."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "orbit_sandbox",
        required={
            "metadata",
            "model",
            "primary",
            "secondaries",
            "satellite",
            "initial_condition",
            "correction",
            "simulation",
            "output_directory",
        },
    )
    metadata = _mapping(root["metadata"], "orbit_sandbox.metadata")
    _keys(metadata, "orbit_sandbox.metadata", required={"name", "safety_scope", "fictional"})
    if _boolean(metadata["fictional"], "orbit_sandbox.metadata.fictional") is not True:
        raise ConfigurationError("orbit_sandbox.metadata.fictional must be true")
    safety_scope = _string(metadata["safety_scope"], "orbit_sandbox.metadata.safety_scope")
    folded = safety_scope.casefold()
    if not all(word in folded for word in ("fictional", "civilian", "synthetic")):
        raise ConfigurationError("orbit safety scope must state fictional, civilian, and synthetic")

    primary_data = _mapping(root["primary"], "orbit_sandbox.primary")
    _keys(
        primary_data,
        "orbit_sandbox.primary",
        required={
            "name",
            "gravitational_parameter_m3_s2",
            "radius_m",
            "j2",
            "rotation_rate_radps",
            "color",
        },
    )
    primary = OrbitPrimaryDefinition(
        _string(primary_data["name"], "orbit_sandbox.primary.name"),
        _number(
            primary_data["gravitational_parameter_m3_s2"],
            "orbit_sandbox.primary.gravitational_parameter_m3_s2",
            positive=True,
        ),
        _number(primary_data["radius_m"], "orbit_sandbox.primary.radius_m", positive=True),
        _number(primary_data["j2"], "orbit_sandbox.primary.j2", nonnegative=True),
        _number(
            primary_data["rotation_rate_radps"],
            "orbit_sandbox.primary.rotation_rate_radps",
            nonnegative=True,
        ),
        _color(primary_data["color"], "orbit_sandbox.primary.color"),
    )

    secondaries: list[OrbitSecondaryDefinition] = []
    for index, item in enumerate(_sequence(root["secondaries"], "orbit_sandbox.secondaries")):
        context = f"orbit_sandbox.secondaries[{index}]"
        data = _mapping(item, context)
        _keys(
            data,
            context,
            required={
                "name",
                "gravitational_parameter_m3_s2",
                "radius_m",
                "orbital_radius_m",
                "phase_at_epoch_deg",
                "inclination_deg",
                "color",
            },
        )
        secondaries.append(
            OrbitSecondaryDefinition(
                _string(data["name"], f"{context}.name"),
                _number(
                    data["gravitational_parameter_m3_s2"],
                    f"{context}.gravitational_parameter_m3_s2",
                    positive=True,
                ),
                _number(data["radius_m"], f"{context}.radius_m", positive=True),
                _number(data["orbital_radius_m"], f"{context}.orbital_radius_m", positive=True),
                np.deg2rad(_number(data["phase_at_epoch_deg"], f"{context}.phase_at_epoch_deg")),
                np.deg2rad(_number(data["inclination_deg"], f"{context}.inclination_deg")),
                _color(data["color"], f"{context}.color"),
            )
        )

    satellite_data = _mapping(root["satellite"], "orbit_sandbox.satellite")
    _keys(
        satellite_data,
        "orbit_sandbox.satellite",
        required={
            "name",
            "initial_mass_kg",
            "dry_mass_kg",
            "drag_area_m2",
            "drag_coefficient",
            "correction_specific_impulse_s",
        },
    )
    satellite = SatelliteDefinition(
        _string(satellite_data["name"], "orbit_sandbox.satellite.name"),
        _number(
            satellite_data["initial_mass_kg"],
            "orbit_sandbox.satellite.initial_mass_kg",
            positive=True,
        ),
        _number(
            satellite_data["dry_mass_kg"],
            "orbit_sandbox.satellite.dry_mass_kg",
            positive=True,
        ),
        _number(
            satellite_data["drag_area_m2"],
            "orbit_sandbox.satellite.drag_area_m2",
            positive=True,
        ),
        _number(
            satellite_data["drag_coefficient"],
            "orbit_sandbox.satellite.drag_coefficient",
            positive=True,
        ),
        _number(
            satellite_data["correction_specific_impulse_s"],
            "orbit_sandbox.satellite.correction_specific_impulse_s",
            positive=True,
        ),
    )

    initial_data = _mapping(root["initial_condition"], "orbit_sandbox.initial_condition")
    _keys(
        initial_data,
        "orbit_sandbox.initial_condition",
        required={
            "altitude_km",
            "speed_mode",
            "custom_speed_mps",
            "inclination_deg",
            "ascending_node_deg",
            "phase_deg",
        },
    )
    initial = OrbitInitialCondition(
        1_000.0
        * _number(
            initial_data["altitude_km"],
            "orbit_sandbox.initial_condition.altitude_km",
            positive=True,
        ),
        _speed_mode(initial_data["speed_mode"], "orbit_sandbox.initial_condition.speed_mode"),
        _number(
            initial_data["custom_speed_mps"],
            "orbit_sandbox.initial_condition.custom_speed_mps",
            nonnegative=True,
        ),
        np.deg2rad(
            _number(
                initial_data["inclination_deg"], "orbit_sandbox.initial_condition.inclination_deg"
            )
        ),
        np.deg2rad(
            _number(
                initial_data["ascending_node_deg"],
                "orbit_sandbox.initial_condition.ascending_node_deg",
            )
        ),
        np.deg2rad(_number(initial_data["phase_deg"], "orbit_sandbox.initial_condition.phase_deg")),
    )

    correction_data = _mapping(root["correction"], "orbit_sandbox.correction")
    _keys(
        correction_data,
        "orbit_sandbox.correction",
        required={"enabled", "trigger_altitude_km", "maximum_burns"},
    )
    maximum_burns_value = correction_data["maximum_burns"]
    if isinstance(maximum_burns_value, bool) or not isinstance(maximum_burns_value, int):
        raise ConfigurationError("orbit_sandbox.correction.maximum_burns: expected an integer")
    correction = OrbitCorrectionDefinition(
        _boolean(correction_data["enabled"], "orbit_sandbox.correction.enabled"),
        1_000.0
        * _number(
            correction_data["trigger_altitude_km"],
            "orbit_sandbox.correction.trigger_altitude_km",
            positive=True,
        ),
        maximum_burns_value,
    )

    simulation = _mapping(root["simulation"], "orbit_sandbox.simulation")
    _keys(
        simulation,
        "orbit_sandbox.simulation",
        required={
            "duration_days",
            "integration_step_s",
            "output_step_s",
            "reentry_altitude_km",
            "escape_radius_multiplier",
            "atmosphere_density_scale",
        },
    )
    return OrbitSandboxConfiguration(
        source_path,
        _string(metadata["name"], "orbit_sandbox.metadata.name"),
        safety_scope,
        _orbit_model(root["model"], "orbit_sandbox.model"),
        primary,
        tuple(secondaries),
        satellite,
        initial,
        correction,
        86_400.0
        * _number(
            simulation["duration_days"], "orbit_sandbox.simulation.duration_days", positive=True
        ),
        _number(
            simulation["integration_step_s"],
            "orbit_sandbox.simulation.integration_step_s",
            positive=True,
        ),
        _number(
            simulation["output_step_s"], "orbit_sandbox.simulation.output_step_s", positive=True
        ),
        1_000.0
        * _number(
            simulation["reentry_altitude_km"],
            "orbit_sandbox.simulation.reentry_altitude_km",
            nonnegative=True,
        ),
        _number(
            simulation["escape_radius_multiplier"],
            "orbit_sandbox.simulation.escape_radius_multiplier",
            positive=True,
        ),
        _number(
            simulation["atmosphere_density_scale"],
            "orbit_sandbox.simulation.atmosphere_density_scale",
            nonnegative=True,
        ),
        Path(_string(root["output_directory"], "orbit_sandbox.output_directory")),
    )
