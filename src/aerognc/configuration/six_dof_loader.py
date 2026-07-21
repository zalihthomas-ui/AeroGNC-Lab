"""Validated configured 6-DOF ascent scenario."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from aerognc.configuration.loader import (
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _sequence,
    _string,
    load_three_dof_configuration,
)
from aerognc.configuration.models import ThreeDofConfiguration
from aerognc.gnc.control_loops import QuaternionAttitudePD
from aerognc.gnc.guidance import AttitudeReferenceSchedule


@dataclass(frozen=True, slots=True)
class SixDofConfiguration:
    """Rigid-body initial condition, reference schedule, and composed base models."""

    source_path: Path
    name: str
    safety_scope: str
    step_s: float
    duration_s: float
    output_directory: Path
    initial_position_ned_m: tuple[float, float, float]
    initial_speed_mps: float
    initial_euler321_deg: tuple[float, float, float]
    initial_angular_rate_body_degps: tuple[float, float, float]
    base: ThreeDofConfiguration
    reference_schedule: AttitudeReferenceSchedule
    controller: QuaternionAttitudePD


def load_six_dof_configuration(path: str | Path) -> SixDofConfiguration:
    """Load one fictional closed-loop rigid-body ascent YAML file."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "six_dof",
        required={
            "metadata",
            "base_scenario_file",
            "simulation",
            "initial",
            "attitude_reference",
            "controller",
        },
    )
    metadata = _mapping(root["metadata"], "six_dof.metadata")
    _keys(metadata, "six_dof.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "six_dof.metadata.safety_scope")
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        raise ValueError("six_dof safety_scope must explicitly state fictional and civilian")

    base_path = Path(_string(root["base_scenario_file"], "six_dof.base_scenario_file"))
    if not base_path.is_absolute():
        base_path = source_path.parent / base_path
    base = load_three_dof_configuration(base_path)

    simulation = _mapping(root["simulation"], "six_dof.simulation")
    _keys(
        simulation,
        "six_dof.simulation",
        required={"step_s", "duration_s", "output_directory"},
    )
    step_s = _number(simulation["step_s"], "six_dof.simulation.step_s", positive=True)
    duration_s = _number(simulation["duration_s"], "six_dof.simulation.duration_s", positive=True)
    if step_s > 0.02:
        raise ValueError("six_dof.simulation.step_s must be no greater than 0.02 s")
    if duration_s > base.simulation.maximum_time_s:
        raise ValueError("six_dof duration exceeds the base environment horizon")

    initial = _mapping(root["initial"], "six_dof.initial")
    _keys(
        initial,
        "six_dof.initial",
        required={
            "position_ned_m",
            "speed_mps",
            "euler321_deg",
            "angular_rate_body_degps",
        },
    )
    position = cast(
        tuple[float, float, float],
        _number_tuple(initial["position_ned_m"], "six_dof.initial.position_ned_m", length=3),
    )
    euler = cast(
        tuple[float, float, float],
        _number_tuple(initial["euler321_deg"], "six_dof.initial.euler321_deg", length=3),
    )
    rate = cast(
        tuple[float, float, float],
        _number_tuple(
            initial["angular_rate_body_degps"],
            "six_dof.initial.angular_rate_body_degps",
            length=3,
        ),
    )

    reference = _mapping(root["attitude_reference"], "six_dof.attitude_reference")
    _keys(
        reference,
        "six_dof.attitude_reference",
        required={"time_s", "euler321_deg"},
    )
    reference_rows = _sequence(reference["euler321_deg"], "six_dof.attitude_reference.euler321_deg")
    reference_schedule = AttitudeReferenceSchedule(
        _number_tuple(reference["time_s"], "six_dof.attitude_reference.time_s"),
        [
            _number_tuple(row, f"six_dof.attitude_reference.euler321_deg[{index}]", length=3)
            for index, row in enumerate(reference_rows)
        ],
    )
    if reference_schedule.time_s[0] > 0.0 or reference_schedule.time_s[-1] < duration_s:
        raise ValueError("six_dof attitude reference must cover the complete simulation interval")

    controller_data = _mapping(root["controller"], "six_dof.controller")
    _keys(
        controller_data,
        "six_dof.controller",
        required={
            "proportional_moment_nm_per_rad",
            "rate_damping_nms_per_rad",
            "moment_limit_nm",
        },
    )
    controller = QuaternionAttitudePD(
        _number_tuple(
            controller_data["proportional_moment_nm_per_rad"],
            "six_dof.controller.proportional_moment_nm_per_rad",
            length=3,
        ),
        _number_tuple(
            controller_data["rate_damping_nms_per_rad"],
            "six_dof.controller.rate_damping_nms_per_rad",
            length=3,
        ),
        _number_tuple(
            controller_data["moment_limit_nm"],
            "six_dof.controller.moment_limit_nm",
            length=3,
        ),
    )
    physical_limit = np.abs(base.vehicle.actuator_allocator.moment_per_command_nm) * np.abs(
        base.vehicle.actuator_allocator.command_limit_rad
    )
    if np.any(controller.moment_limit_nm > physical_limit + 1.0e-12):
        raise ValueError("six_dof controller moment limit exceeds actuator authority")
    return SixDofConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "six_dof.metadata.name"),
        safety_scope=safety_scope,
        step_s=step_s,
        duration_s=duration_s,
        output_directory=Path(
            _string(simulation["output_directory"], "six_dof.simulation.output_directory")
        ),
        initial_position_ned_m=position,
        initial_speed_mps=_number(
            initial["speed_mps"], "six_dof.initial.speed_mps", nonnegative=True
        ),
        initial_euler321_deg=euler,
        initial_angular_rate_body_degps=rate,
        base=base,
        reference_schedule=reference_schedule,
        controller=controller,
    )
