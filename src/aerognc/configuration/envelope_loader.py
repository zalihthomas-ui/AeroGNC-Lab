"""Strict configuration loading for flight-envelope control analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from aerognc.configuration.loader import (
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _string,
    load_three_dof_configuration,
)
from aerognc.configuration.models import ThreeDofConfiguration


@dataclass(frozen=True, slots=True)
class FlightEnvelopeConfiguration:
    """Validated inputs for synthetic ascent-envelope analysis."""

    source_path: Path
    name: str
    safety_scope: str
    base_scenario: ThreeDofConfiguration
    mach_points: tuple[float, ...]
    altitude_points_m: tuple[float, ...]
    mass_points_kg: tuple[float, ...]
    normal_control_derivative_per_rad: float
    pitch_control_derivative_per_rad: float
    pitch_rate_derivative: float
    disturbance_moment_nm: float
    angle_of_attack_limit_rad: float
    state_weight_diagonal: tuple[float, float]
    input_weight: float
    uncertainty_sample_count: int
    aerodynamic_derivative_sigma_fraction: float
    control_effectiveness_sigma_fraction: float
    inertia_sigma_fraction: float
    random_seed: int
    minimum_control_authority_fraction: float
    minimum_closed_loop_damping_ratio: float
    output_directory: Path


def _increasing_positive_tuple(value: object, context: str) -> tuple[float, ...]:
    values = _number_tuple(value, context)
    if len(values) < 2 or any(item <= 0.0 for item in values):
        raise ValueError(f"{context} must contain at least two positive values")
    if any(after <= before for before, after in pairwise(values)):
        raise ValueError(f"{context} must be strictly increasing")
    return values


def _fraction(value: object, context: str, *, allow_zero: bool = True) -> float:
    parsed = _number(value, context, nonnegative=allow_zero, positive=not allow_zero)
    if parsed >= 1.0:
        raise ValueError(f"{context} must be less than 1")
    return parsed


def load_flight_envelope_configuration(path: str | Path) -> FlightEnvelopeConfiguration:
    """Load an envelope-analysis YAML file and reject ambiguous inputs."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "flight_envelope",
        required={
            "metadata",
            "base_scenario",
            "grid",
            "aerodynamic_derivatives",
            "trim",
            "lqr",
            "robustness",
            "requirements",
            "output_directory",
        },
    )
    metadata = _mapping(root["metadata"], "flight_envelope.metadata")
    _keys(metadata, "flight_envelope.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "flight_envelope.metadata.safety_scope")
    folded_scope = safety_scope.casefold()
    if not all(word in folded_scope for word in ("fictional", "civilian", "synthetic")):
        raise ValueError("safety_scope must explicitly say fictional, civilian, and synthetic")

    base_scenario_path = Path(_string(root["base_scenario"], "flight_envelope.base_scenario"))
    if not base_scenario_path.is_absolute():
        base_scenario_path = source_path.parent / base_scenario_path
    base_scenario = load_three_dof_configuration(base_scenario_path.resolve())

    grid = _mapping(root["grid"], "flight_envelope.grid")
    _keys(grid, "flight_envelope.grid", required={"mach", "altitude_m", "mass_kg"})
    mach_points = _increasing_positive_tuple(grid["mach"], "flight_envelope.grid.mach")
    altitude_points = _number_tuple(grid["altitude_m"], "flight_envelope.grid.altitude_m")
    if len(altitude_points) < 2 or any(item < 0.0 for item in altitude_points):
        raise ValueError("flight_envelope.grid.altitude_m must contain nonnegative values")
    if any(after <= before for before, after in pairwise(altitude_points)):
        raise ValueError("flight_envelope.grid.altitude_m must be strictly increasing")
    mass_points = _increasing_positive_tuple(grid["mass_kg"], "flight_envelope.grid.mass_kg")
    dry_mass = base_scenario.vehicle.mass_properties.dry_mass_kg
    wet_mass = base_scenario.vehicle.mass_properties.wet_mass_kg
    if mass_points[0] < dry_mass or mass_points[-1] > wet_mass:
        raise ValueError(
            "flight-envelope masses must stay within configured dry and wet vehicle mass"
        )

    derivatives = _mapping(
        root["aerodynamic_derivatives"], "flight_envelope.aerodynamic_derivatives"
    )
    _keys(
        derivatives,
        "flight_envelope.aerodynamic_derivatives",
        required={
            "normal_control_per_rad",
            "pitch_control_per_rad",
            "pitch_rate",
        },
    )
    normal_control = _number(
        derivatives["normal_control_per_rad"],
        "flight_envelope.aerodynamic_derivatives.normal_control_per_rad",
    )
    pitch_control = _number(
        derivatives["pitch_control_per_rad"],
        "flight_envelope.aerodynamic_derivatives.pitch_control_per_rad",
    )
    if abs(normal_control) <= 1.0e-12 or abs(pitch_control) <= 1.0e-12:
        raise ValueError("control derivatives must be nonzero")

    trim = _mapping(root["trim"], "flight_envelope.trim")
    _keys(trim, "flight_envelope.trim", required={"disturbance_moment_nm", "alpha_limit_deg"})
    lqr = _mapping(root["lqr"], "flight_envelope.lqr")
    _keys(lqr, "flight_envelope.lqr", required={"state_weight_diagonal", "input_weight"})
    weights = cast(
        tuple[float, float],
        _number_tuple(
            lqr["state_weight_diagonal"],
            "flight_envelope.lqr.state_weight_diagonal",
            length=2,
        ),
    )
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("flight-envelope LQR state weights must be positive")

    robustness = _mapping(root["robustness"], "flight_envelope.robustness")
    _keys(
        robustness,
        "flight_envelope.robustness",
        required={
            "sample_count",
            "aerodynamic_derivative_sigma_fraction",
            "control_effectiveness_sigma_fraction",
            "inertia_sigma_fraction",
            "random_seed",
        },
    )
    requirements = _mapping(root["requirements"], "flight_envelope.requirements")
    _keys(
        requirements,
        "flight_envelope.requirements",
        required={"minimum_control_authority_fraction", "minimum_closed_loop_damping_ratio"},
    )
    sample_count = _integer(
        robustness["sample_count"], "flight_envelope.robustness.sample_count", nonnegative=True
    )
    if sample_count < 10:
        raise ValueError("flight_envelope.robustness.sample_count must be at least 10")

    return FlightEnvelopeConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "flight_envelope.metadata.name"),
        safety_scope=safety_scope,
        base_scenario=base_scenario,
        mach_points=mach_points,
        altitude_points_m=altitude_points,
        mass_points_kg=mass_points,
        normal_control_derivative_per_rad=normal_control,
        pitch_control_derivative_per_rad=pitch_control,
        pitch_rate_derivative=_number(
            derivatives["pitch_rate"], "flight_envelope.aerodynamic_derivatives.pitch_rate"
        ),
        disturbance_moment_nm=_number(
            trim["disturbance_moment_nm"], "flight_envelope.trim.disturbance_moment_nm"
        ),
        angle_of_attack_limit_rad=_number(
            trim["alpha_limit_deg"], "flight_envelope.trim.alpha_limit_deg", positive=True
        )
        * 3.141592653589793
        / 180.0,
        state_weight_diagonal=weights,
        input_weight=_number(
            lqr["input_weight"], "flight_envelope.lqr.input_weight", positive=True
        ),
        uncertainty_sample_count=sample_count,
        aerodynamic_derivative_sigma_fraction=_fraction(
            robustness["aerodynamic_derivative_sigma_fraction"],
            "flight_envelope.robustness.aerodynamic_derivative_sigma_fraction",
        ),
        control_effectiveness_sigma_fraction=_fraction(
            robustness["control_effectiveness_sigma_fraction"],
            "flight_envelope.robustness.control_effectiveness_sigma_fraction",
        ),
        inertia_sigma_fraction=_fraction(
            robustness["inertia_sigma_fraction"],
            "flight_envelope.robustness.inertia_sigma_fraction",
        ),
        random_seed=_integer(
            robustness["random_seed"],
            "flight_envelope.robustness.random_seed",
            nonnegative=True,
        ),
        minimum_control_authority_fraction=_fraction(
            requirements["minimum_control_authority_fraction"],
            "flight_envelope.requirements.minimum_control_authority_fraction",
            allow_zero=False,
        ),
        minimum_closed_loop_damping_ratio=_fraction(
            requirements["minimum_closed_loop_damping_ratio"],
            "flight_envelope.requirements.minimum_closed_loop_damping_ratio",
            allow_zero=False,
        ),
        output_directory=Path(
            _string(root["output_directory"], "flight_envelope.output_directory")
        ),
    )
