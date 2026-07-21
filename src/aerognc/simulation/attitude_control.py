"""Closed-loop scalar attitude benchmark with the same bounded actuator model."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from aerognc.configuration.control_loader import AttitudeControlConfiguration
from aerognc.gnc.control_loops import CascadedAttitudeController
from aerognc.gnc.pid import PIDController
from aerognc.gnc.state_feedback import StateFeedbackController, ackermann_gain
from aerognc.mathematics.integrators import rk4_step
from aerognc.mathematics.vectors import FloatArray
from aerognc.vehicle.actuators import FirstOrderActuator
from aerognc.verification.metrics import StepResponseMetrics, step_response_metrics

ControllerKind = Literal["cascaded_pid", "state_feedback"]


@dataclass(frozen=True, slots=True)
class AttitudeControlResult:
    """Signals and metrics for one controller benchmark."""

    controller_name: ControllerKind
    time_s: FloatArray
    reference_angle_rad: FloatArray
    angle_rad: FloatArray
    angular_rate_radps: FloatArray
    rate_command_radps: FloatArray
    commanded_moment_nm: FloatArray
    actuator_position_rad: FloatArray
    achieved_moment_nm: FloatArray
    disturbance_moment_nm: FloatArray
    saturated: np.ndarray
    metrics: StepResponseMetrics


def simulate_attitude_control(
    configuration: AttitudeControlConfiguration,
    controller_kind: ControllerKind,
) -> AttitudeControlResult:
    """Run one deterministic closed-loop attitude benchmark."""
    sample_count = int(np.floor(configuration.duration_s / configuration.step_s + 0.5)) + 1
    time_s = np.linspace(0.0, configuration.duration_s, sample_count)
    reference = np.where(
        time_s >= configuration.command_time_s, configuration.reference_angle_rad, 0.0
    )
    disturbance = np.where(
        (time_s >= configuration.disturbance_start_s) & (time_s < configuration.disturbance_end_s),
        configuration.disturbance_moment_nm,
        0.0,
    )
    angle = np.zeros(sample_count)
    rate = np.zeros(sample_count)
    rate_command = np.zeros(sample_count)
    commanded_moment = np.zeros(sample_count)
    actuator_position = np.zeros(sample_count)
    achieved_moment = np.zeros(sample_count)
    saturated = np.zeros(sample_count, dtype=bool)
    actuator = FirstOrderActuator(configuration.actuator_limits)

    cascaded = CascadedAttitudeController(
        PIDController(configuration.attitude_pid),
        PIDController(configuration.rate_pid),
        configuration.rate_command_limit_radps,
        configuration.moment_command_limit_nm,
    )
    system_matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
    input_matrix = np.array([[0.0], [1.0 / configuration.inertia_kgm2]])
    gain = ackermann_gain(system_matrix, input_matrix, configuration.state_feedback_poles)
    feedback = StateFeedbackController(gain, configuration.moment_command_limit_nm)
    controller_execution_s = 0.0

    for index in range(sample_count):
        controller_start = perf_counter()
        if controller_kind == "cascaded_pid":
            moment_command, rate_reference = cascaded.update(
                float(reference[index]),
                float(angle[index]),
                float(rate[index]),
                configuration.step_s,
            )
            rate_command[index] = rate_reference
        elif controller_kind == "state_feedback":
            moment_command = feedback.command([angle[index], rate[index]], [reference[index], 0.0])
            rate_command[index] = 0.0
        else:
            raise ValueError(f"unknown controller_kind: {controller_kind}")
        controller_execution_s += perf_counter() - controller_start
        commanded_moment[index] = moment_command
        actuator_command_rad = moment_command / configuration.actuator_effectiveness_nm_per_rad
        actuator_position[index] = actuator.update(actuator_command_rad, configuration.step_s)
        achieved_moment[index] = (
            configuration.actuator_effectiveness_nm_per_rad * actuator_position[index]
        )
        saturated[index] = actuator.saturated or np.isclose(
            abs(moment_command), configuration.moment_command_limit_nm
        )
        if index == sample_count - 1:
            continue

        total_external_moment = achieved_moment[index] + disturbance[index]

        def plant_derivative(
            _time_s: float,
            state: FloatArray,
            applied_moment_nm: float = total_external_moment,
        ) -> FloatArray:
            return np.array(
                [
                    state[1],
                    (applied_moment_nm - configuration.passive_damping_nms * state[1])
                    / configuration.inertia_kgm2,
                ]
            )

        next_state = rk4_step(
            plant_derivative,
            float(time_s[index]),
            [angle[index], rate[index]],
            configuration.step_s,
        )
        angle[index + 1], rate[index + 1] = next_state

    metrics = step_response_metrics(
        time_s,
        reference,
        angle,
        achieved_moment,
        saturated,
        command_time_s=configuration.command_time_s,
        disturbance_start_time_s=configuration.disturbance_start_s,
        disturbance_end_time_s=configuration.disturbance_end_s,
        execution_time_s=controller_execution_s,
    )
    return AttitudeControlResult(
        controller_name=controller_kind,
        time_s=time_s,
        reference_angle_rad=reference,
        angle_rad=angle,
        angular_rate_radps=rate,
        rate_command_radps=rate_command,
        commanded_moment_nm=commanded_moment,
        actuator_position_rad=actuator_position,
        achieved_moment_nm=achieved_moment,
        disturbance_moment_nm=disturbance,
        saturated=saturated,
        metrics=metrics,
    )


def compare_attitude_controllers(
    configuration: AttitudeControlConfiguration,
) -> tuple[AttitudeControlResult, AttitudeControlResult]:
    """Run cascaded PID and manual-pole-placement state feedback."""
    return (
        simulate_attitude_control(configuration, "cascaded_pid"),
        simulate_attitude_control(configuration, "state_feedback"),
    )


def write_attitude_control_outputs(
    results: tuple[AttitudeControlResult, AttitudeControlResult],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic comparison signals and quantitative metric JSON."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "attitude_control_comparison.csv"
    first, second = results
    if not np.array_equal(first.time_s, second.time_s):
        raise ValueError("controller comparison time bases must be identical")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "time_s",
                "reference_angle_rad",
                "cascaded_pid_angle_rad",
                "cascaded_pid_rate_radps",
                "cascaded_pid_achieved_moment_nm",
                "cascaded_pid_actuator_position_rad",
                "cascaded_pid_saturated",
                "state_feedback_angle_rad",
                "state_feedback_rate_radps",
                "state_feedback_achieved_moment_nm",
                "state_feedback_actuator_position_rad",
                "state_feedback_saturated",
                "disturbance_moment_nm",
            ]
        )
        for index, time_s in enumerate(first.time_s):
            writer.writerow(
                [
                    f"{time_s:.10g}",
                    f"{first.reference_angle_rad[index]:.10g}",
                    f"{first.angle_rad[index]:.10g}",
                    f"{first.angular_rate_radps[index]:.10g}",
                    f"{first.achieved_moment_nm[index]:.10g}",
                    f"{first.actuator_position_rad[index]:.10g}",
                    int(first.saturated[index]),
                    f"{second.angle_rad[index]:.10g}",
                    f"{second.angular_rate_radps[index]:.10g}",
                    f"{second.achieved_moment_nm[index]:.10g}",
                    f"{second.actuator_position_rad[index]:.10g}",
                    int(second.saturated[index]),
                    f"{first.disturbance_moment_nm[index]:.10g}",
                ]
            )
    metrics_path = output / "attitude_control_metrics.json"
    metrics_payload = {result.controller_name: result.metrics.as_dict() for result in results}
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, metrics_path
