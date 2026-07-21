"""Validated scalar attitude-control benchmark configuration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.loader import (
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _string,
)
from aerognc.gnc.pid import PIDGains
from aerognc.vehicle.actuators import ActuatorLimits


@dataclass(frozen=True, slots=True)
class AttitudeControlConfiguration:
    """Complete single-axis fictional attitude-control comparison setup."""

    source_path: Path
    name: str
    safety_scope: str
    step_s: float
    duration_s: float
    output_directory: Path
    command_time_s: float
    reference_angle_rad: float
    disturbance_start_s: float
    disturbance_end_s: float
    disturbance_moment_nm: float
    inertia_kgm2: float
    passive_damping_nms: float
    actuator_limits: ActuatorLimits
    actuator_effectiveness_nm_per_rad: float
    attitude_pid: PIDGains
    rate_pid: PIDGains
    rate_command_limit_radps: float
    moment_command_limit_nm: float
    state_feedback_poles: tuple[float, float]


def _pid_gains(value: object, context: str, output_limit: float) -> PIDGains:
    data = _mapping(value, context)
    _keys(
        data,
        context,
        required={"kp", "ki", "kd", "integral_limit", "derivative_filter_tau_s"},
    )
    integral_limit = _number(data["integral_limit"], f"{context}.integral_limit", positive=True)
    return PIDGains(
        proportional=_number(data["kp"], f"{context}.kp"),
        integral=_number(data["ki"], f"{context}.ki"),
        derivative=_number(data["kd"], f"{context}.kd"),
        output_min=-output_limit,
        output_max=output_limit,
        integral_min=-integral_limit,
        integral_max=integral_limit,
        derivative_filter_tau_s=_number(
            data["derivative_filter_tau_s"],
            f"{context}.derivative_filter_tau_s",
            nonnegative=True,
        ),
        anti_windup_gain=1.0,
    )


def load_attitude_control_configuration(path: str | Path) -> AttitudeControlConfiguration:
    """Load the public-safe single-axis attitude benchmark YAML."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "attitude_control",
        required={
            "metadata",
            "simulation",
            "command",
            "disturbance",
            "plant",
            "actuator",
            "cascaded_pid",
            "state_feedback",
        },
    )
    metadata = _mapping(root["metadata"], "attitude_control.metadata")
    _keys(metadata, "attitude_control.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "attitude_control.metadata.safety_scope")
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        raise ValueError("attitude control safety_scope must state fictional and civilian")

    simulation = _mapping(root["simulation"], "attitude_control.simulation")
    _keys(
        simulation,
        "attitude_control.simulation",
        required={"step_s", "duration_s", "output_directory"},
    )
    step_s = _number(simulation["step_s"], "attitude_control.simulation.step_s", positive=True)
    duration_s = _number(
        simulation["duration_s"], "attitude_control.simulation.duration_s", positive=True
    )
    command = _mapping(root["command"], "attitude_control.command")
    _keys(command, "attitude_control.command", required={"time_s", "reference_angle_deg"})
    command_time_s = _number(command["time_s"], "attitude_control.command.time_s", nonnegative=True)
    reference_angle_rad = float(
        np.deg2rad(
            _number(
                command["reference_angle_deg"],
                "attitude_control.command.reference_angle_deg",
            )
        )
    )

    disturbance = _mapping(root["disturbance"], "attitude_control.disturbance")
    _keys(
        disturbance,
        "attitude_control.disturbance",
        required={"start_s", "end_s", "moment_nm"},
    )
    disturbance_start_s = _number(
        disturbance["start_s"], "attitude_control.disturbance.start_s", nonnegative=True
    )
    disturbance_end_s = _number(
        disturbance["end_s"], "attitude_control.disturbance.end_s", nonnegative=True
    )
    if not command_time_s < disturbance_start_s < disturbance_end_s < duration_s:
        raise ValueError(
            "attitude control timing must satisfy command < disturbance start < end < duration"
        )

    plant = _mapping(root["plant"], "attitude_control.plant")
    _keys(plant, "attitude_control.plant", required={"inertia_kgm2", "passive_damping_nms"})
    inertia = _number(plant["inertia_kgm2"], "attitude_control.plant.inertia_kgm2", positive=True)
    damping = _number(
        plant["passive_damping_nms"],
        "attitude_control.plant.passive_damping_nms",
        nonnegative=True,
    )

    actuator = _mapping(root["actuator"], "attitude_control.actuator")
    _keys(
        actuator,
        "attitude_control.actuator",
        required={
            "time_constant_s",
            "position_limit_deg",
            "rate_limit_degps",
            "command_delay_s",
            "effectiveness_nm_per_rad",
        },
    )
    position_limit_rad = float(
        np.deg2rad(
            _number(
                actuator["position_limit_deg"],
                "attitude_control.actuator.position_limit_deg",
                positive=True,
            )
        )
    )
    actuator_limits = ActuatorLimits(
        time_constant_s=_number(
            actuator["time_constant_s"],
            "attitude_control.actuator.time_constant_s",
            positive=True,
        ),
        position_limit_rad=position_limit_rad,
        rate_limit_radps=float(
            np.deg2rad(
                _number(
                    actuator["rate_limit_degps"],
                    "attitude_control.actuator.rate_limit_degps",
                    positive=True,
                )
            )
        ),
        command_delay_s=_number(
            actuator["command_delay_s"],
            "attitude_control.actuator.command_delay_s",
            nonnegative=True,
        ),
    )
    effectiveness = _number(
        actuator["effectiveness_nm_per_rad"],
        "attitude_control.actuator.effectiveness_nm_per_rad",
        positive=True,
    )
    physical_moment_limit = effectiveness * position_limit_rad

    cascaded = _mapping(root["cascaded_pid"], "attitude_control.cascaded_pid")
    _keys(
        cascaded,
        "attitude_control.cascaded_pid",
        required={"rate_command_limit_radps", "moment_command_limit_nm", "attitude", "rate"},
    )
    rate_limit = _number(
        cascaded["rate_command_limit_radps"],
        "attitude_control.cascaded_pid.rate_command_limit_radps",
        positive=True,
    )
    moment_limit = _number(
        cascaded["moment_command_limit_nm"],
        "attitude_control.cascaded_pid.moment_command_limit_nm",
        positive=True,
    )
    if moment_limit > physical_moment_limit + 1.0e-12:
        raise ValueError("PID moment limit exceeds actuator physical authority")

    feedback = _mapping(root["state_feedback"], "attitude_control.state_feedback")
    _keys(feedback, "attitude_control.state_feedback", required={"poles_radps"})
    poles = _number_tuple(
        feedback["poles_radps"], "attitude_control.state_feedback.poles_radps", length=2
    )
    if any(pole >= 0.0 for pole in poles):
        raise ValueError("state-feedback poles must have negative real values")
    return AttitudeControlConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "attitude_control.metadata.name"),
        safety_scope=safety_scope,
        step_s=step_s,
        duration_s=duration_s,
        output_directory=Path(
            _string(
                simulation["output_directory"],
                "attitude_control.simulation.output_directory",
            )
        ),
        command_time_s=command_time_s,
        reference_angle_rad=reference_angle_rad,
        disturbance_start_s=disturbance_start_s,
        disturbance_end_s=disturbance_end_s,
        disturbance_moment_nm=_number(
            disturbance["moment_nm"], "attitude_control.disturbance.moment_nm"
        ),
        inertia_kgm2=inertia,
        passive_damping_nms=damping,
        actuator_limits=actuator_limits,
        actuator_effectiveness_nm_per_rad=effectiveness,
        attitude_pid=_pid_gains(
            cascaded["attitude"], "attitude_control.cascaded_pid.attitude", rate_limit
        ),
        rate_pid=_pid_gains(cascaded["rate"], "attitude_control.cascaded_pid.rate", moment_limit),
        rate_command_limit_radps=rate_limit,
        moment_command_limit_nm=moment_limit,
        state_feedback_poles=(poles[0], poles[1]),
    )
