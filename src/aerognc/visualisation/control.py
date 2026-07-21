"""Closed-loop attitude-controller comparison figures."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.configuration.control_loader import AttitudeControlConfiguration
from aerognc.simulation.attitude_control import AttitudeControlResult
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style


def plot_attitude_control_comparison(
    results: tuple[AttitudeControlResult, AttitudeControlResult],
    configuration: AttitudeControlConfiguration,
    output_directory: str | Path,
) -> Path:
    """Plot tracking, error, rate, moment, and actuator-limit behaviour."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pid, feedback = results
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True)
        time_s = pid.time_s
        axes[0, 0].plot(
            time_s,
            np.rad2deg(pid.reference_angle_rad),
            color=NAVY,
            linestyle="--",
            label="Reference",
        )
        axes[0, 0].plot(time_s, np.rad2deg(pid.angle_rad), color=BLUE, label="Cascaded PID")
        axes[0, 0].plot(
            time_s, np.rad2deg(feedback.angle_rad), color=ORANGE, label="State feedback"
        )
        axes[0, 0].set(ylabel="Pitch angle [deg]", title="Attitude tracking")
        axes[0, 0].legend(ncol=3, fontsize=8)
        axes[0, 1].plot(
            time_s,
            np.rad2deg(pid.reference_angle_rad - pid.angle_rad),
            color=BLUE,
            label="Cascaded PID",
        )
        axes[0, 1].plot(
            time_s,
            np.rad2deg(feedback.reference_angle_rad - feedback.angle_rad),
            color=ORANGE,
            label="State feedback",
        )
        axes[0, 1].axhline(0.0, color=NAVY, linewidth=0.8)
        axes[0, 1].set(ylabel="Tracking error [deg]", title="Tracking error")
        axes[1, 0].plot(
            time_s, np.rad2deg(pid.angular_rate_radps), color=BLUE, label="Cascaded PID"
        )
        axes[1, 0].plot(
            time_s,
            np.rad2deg(feedback.angular_rate_radps),
            color=ORANGE,
            label="State feedback",
        )
        axes[1, 0].set(xlabel="Time [s]", ylabel="Pitch rate [deg/s]", title="Angular rate")
        axes[1, 1].plot(time_s, pid.achieved_moment_nm, color=BLUE, label="Cascaded PID")
        axes[1, 1].plot(time_s, feedback.achieved_moment_nm, color=ORANGE, label="State feedback")
        axes[1, 1].plot(time_s, pid.disturbance_moment_nm, color=RED, label="Disturbance")
        axes[1, 1].set(xlabel="Time [s]", ylabel="Moment [N m]", title="Actuation and disturbance")
        axes[1, 1].legend(ncol=3, fontsize=8)
        for axis in axes.flat:
            axis.axvspan(
                configuration.disturbance_start_s,
                configuration.disturbance_end_s,
                color=GREEN,
                alpha=0.08,
                linewidth=0.0,
            )
        figure.suptitle(
            "Fictional single-axis attitude-control comparison", color=NAVY, fontweight="bold"
        )
        path = output / "attitude_control_comparison.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
