"""Strict configuration for public-safe constrained ascent guidance."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

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
class AscentGuidanceConfiguration:
    """Validated reference, constraint, optimizer, and simulation inputs."""

    source_path: Path
    name: str
    safety_scope: str
    base_scenario: ThreeDofConfiguration
    reference_time_s: tuple[float, ...]
    reference_elevation_rad: tuple[float, ...]
    reference_throttle: tuple[float, ...]
    maximum_dynamic_pressure_pa: float
    maximum_proper_load_factor: float
    maximum_angle_of_attack_rad: float
    minimum_alpha_constraint_dynamic_pressure_pa: float
    dynamic_pressure_soft_fraction: float
    ballistic_apogee_reserve_m: float
    pitch_time_constant_s: float
    pitch_rate_limit_radps: float
    desired_apogee_m: float
    apogee_tolerance_m: float
    elevation_offset_bounds_rad: tuple[float, float]
    throttle_scale_bounds: tuple[float, float]
    initial_elevation_step_rad: float
    initial_throttle_step: float
    maximum_optimizer_iterations: int
    simulation_step_s: float
    maximum_time_s: float
    output_directory: Path


def _bounded_pair(value: object, context: str) -> tuple[float, float]:
    lower, upper = _number_tuple(value, context, length=2)
    if lower >= upper:
        raise ValueError(f"{context} lower bound must be below upper bound")
    return lower, upper


def load_ascent_guidance_configuration(path: str | Path) -> AscentGuidanceConfiguration:
    """Load a constrained ascent case with strict keys and SI conversion at the boundary."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "ascent_guidance",
        required={
            "metadata",
            "base_scenario",
            "reference",
            "constraints",
            "governor",
            "pitch_response",
            "performance",
            "optimization",
            "simulation",
            "output_directory",
        },
    )
    metadata = _mapping(root["metadata"], "ascent_guidance.metadata")
    _keys(metadata, "ascent_guidance.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "ascent_guidance.metadata.safety_scope")
    scope = safety_scope.casefold()
    if not all(word in scope for word in ("fictional", "civilian", "synthetic")):
        raise ValueError("ascent-guidance safety_scope must say fictional, civilian, synthetic")
    forbidden_scope_terms = ("intercept", "homing", "weapon", "target vehicle")
    if any(term in scope for term in forbidden_scope_terms):
        raise ValueError("ascent-guidance safety_scope contains an excluded operational topic")

    scenario_path = Path(_string(root["base_scenario"], "ascent_guidance.base_scenario"))
    if not scenario_path.is_absolute():
        scenario_path = source_path.parent / scenario_path
    base_scenario = load_three_dof_configuration(scenario_path.resolve())

    reference = _mapping(root["reference"], "ascent_guidance.reference")
    _keys(
        reference,
        "ascent_guidance.reference",
        required={"time_s", "elevation_deg", "throttle"},
    )
    times = _number_tuple(reference["time_s"], "ascent_guidance.reference.time_s")
    elevations_deg = _number_tuple(
        reference["elevation_deg"], "ascent_guidance.reference.elevation_deg"
    )
    throttle = _number_tuple(reference["throttle"], "ascent_guidance.reference.throttle")
    if len(times) < 2 or len(times) != len(elevations_deg) or len(times) != len(throttle):
        raise ValueError("reference time, elevation, and throttle arrays must share two points")
    if times[0] != 0.0 or any(after <= before for before, after in pairwise(times)):
        raise ValueError("reference time_s must begin at zero and be strictly increasing")
    if any(not 0.0 <= item <= 1.0 for item in throttle):
        raise ValueError("reference throttle values must lie in [0, 1]")
    if any(not 0.0 < item <= 90.0 for item in elevations_deg):
        raise ValueError("reference elevations must lie in (0, 90] deg")

    constraints = _mapping(root["constraints"], "ascent_guidance.constraints")
    _keys(
        constraints,
        "ascent_guidance.constraints",
        required={
            "maximum_dynamic_pressure_pa",
            "maximum_proper_load_factor",
            "maximum_angle_of_attack_deg",
            "minimum_alpha_constraint_dynamic_pressure_pa",
        },
    )
    governor = _mapping(root["governor"], "ascent_guidance.governor")
    _keys(
        governor,
        "ascent_guidance.governor",
        required={"dynamic_pressure_soft_fraction", "ballistic_apogee_reserve_m"},
    )
    soft_fraction = _number(
        governor["dynamic_pressure_soft_fraction"],
        "ascent_guidance.governor.dynamic_pressure_soft_fraction",
        positive=True,
    )
    if soft_fraction >= 1.0:
        raise ValueError("dynamic_pressure_soft_fraction must be less than one")

    pitch = _mapping(root["pitch_response"], "ascent_guidance.pitch_response")
    _keys(
        pitch,
        "ascent_guidance.pitch_response",
        required={"time_constant_s", "rate_limit_degps"},
    )
    performance = _mapping(root["performance"], "ascent_guidance.performance")
    _keys(
        performance,
        "ascent_guidance.performance",
        required={"desired_apogee_m", "apogee_tolerance_m"},
    )
    optimization = _mapping(root["optimization"], "ascent_guidance.optimization")
    _keys(
        optimization,
        "ascent_guidance.optimization",
        required={
            "elevation_offset_bounds_deg",
            "throttle_scale_bounds",
            "initial_elevation_step_deg",
            "initial_throttle_step",
            "maximum_iterations",
        },
    )
    elevation_bounds_deg = _bounded_pair(
        optimization["elevation_offset_bounds_deg"],
        "ascent_guidance.optimization.elevation_offset_bounds_deg",
    )
    throttle_bounds = _bounded_pair(
        optimization["throttle_scale_bounds"],
        "ascent_guidance.optimization.throttle_scale_bounds",
    )
    if throttle_bounds[0] < 0.0 or throttle_bounds[1] > 1.0:
        raise ValueError("throttle_scale_bounds must stay within [0, 1]")
    maximum_iterations = _integer(
        optimization["maximum_iterations"],
        "ascent_guidance.optimization.maximum_iterations",
        nonnegative=True,
    )
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least one")

    simulation = _mapping(root["simulation"], "ascent_guidance.simulation")
    _keys(simulation, "ascent_guidance.simulation", required={"step_s", "maximum_time_s"})
    step_s = _number(simulation["step_s"], "ascent_guidance.simulation.step_s", positive=True)
    if step_s > 0.1:
        raise ValueError("ascent-guidance simulation step must be no greater than 0.1 s")
    maximum_time_s = _number(
        simulation["maximum_time_s"], "ascent_guidance.simulation.maximum_time_s", positive=True
    )
    if maximum_time_s <= times[-1]:
        raise ValueError("maximum simulation time must exceed the reference duration")

    maximum_dynamic_pressure = _number(
        constraints["maximum_dynamic_pressure_pa"],
        "ascent_guidance.constraints.maximum_dynamic_pressure_pa",
        positive=True,
    )
    minimum_alpha_dynamic_pressure = _number(
        constraints["minimum_alpha_constraint_dynamic_pressure_pa"],
        "ascent_guidance.constraints.minimum_alpha_constraint_dynamic_pressure_pa",
        nonnegative=True,
    )
    if minimum_alpha_dynamic_pressure >= maximum_dynamic_pressure:
        raise ValueError("minimum alpha-constraint pressure must be below the max-Q limit")

    return AscentGuidanceConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "ascent_guidance.metadata.name"),
        safety_scope=safety_scope,
        base_scenario=base_scenario,
        reference_time_s=times,
        reference_elevation_rad=tuple(float(np.deg2rad(value)) for value in elevations_deg),
        reference_throttle=throttle,
        maximum_dynamic_pressure_pa=maximum_dynamic_pressure,
        maximum_proper_load_factor=_number(
            constraints["maximum_proper_load_factor"],
            "ascent_guidance.constraints.maximum_proper_load_factor",
            positive=True,
        ),
        maximum_angle_of_attack_rad=float(
            np.deg2rad(
                _number(
                    constraints["maximum_angle_of_attack_deg"],
                    "ascent_guidance.constraints.maximum_angle_of_attack_deg",
                    positive=True,
                )
            )
        ),
        minimum_alpha_constraint_dynamic_pressure_pa=minimum_alpha_dynamic_pressure,
        dynamic_pressure_soft_fraction=soft_fraction,
        ballistic_apogee_reserve_m=_number(
            governor["ballistic_apogee_reserve_m"],
            "ascent_guidance.governor.ballistic_apogee_reserve_m",
            nonnegative=True,
        ),
        pitch_time_constant_s=_number(
            pitch["time_constant_s"],
            "ascent_guidance.pitch_response.time_constant_s",
            positive=True,
        ),
        pitch_rate_limit_radps=float(
            np.deg2rad(
                _number(
                    pitch["rate_limit_degps"],
                    "ascent_guidance.pitch_response.rate_limit_degps",
                    positive=True,
                )
            )
        ),
        desired_apogee_m=_number(
            performance["desired_apogee_m"],
            "ascent_guidance.performance.desired_apogee_m",
            positive=True,
        ),
        apogee_tolerance_m=_number(
            performance["apogee_tolerance_m"],
            "ascent_guidance.performance.apogee_tolerance_m",
            positive=True,
        ),
        elevation_offset_bounds_rad=(
            float(np.deg2rad(elevation_bounds_deg[0])),
            float(np.deg2rad(elevation_bounds_deg[1])),
        ),
        throttle_scale_bounds=throttle_bounds,
        initial_elevation_step_rad=float(
            np.deg2rad(
                _number(
                    optimization["initial_elevation_step_deg"],
                    "ascent_guidance.optimization.initial_elevation_step_deg",
                    positive=True,
                )
            )
        ),
        initial_throttle_step=_number(
            optimization["initial_throttle_step"],
            "ascent_guidance.optimization.initial_throttle_step",
            positive=True,
        ),
        maximum_optimizer_iterations=maximum_iterations,
        simulation_step_s=step_s,
        maximum_time_s=maximum_time_s,
        output_directory=Path(
            _string(root["output_directory"], "ascent_guidance.output_directory")
        ),
    )
