"""Configured 6-DOF ascent visualisation."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerognc.simulation.logging import SimulationResult
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style


def plot_six_dof_results(
    result: SimulationResult,
    output_directory: str | Path,
    *,
    filename: str = "six_dof_ascent.png",
) -> Path:
    """Plot rigid-body trajectory, quaternion-attitude tracking, rates, and aero angles."""
    if Path(filename).name != filename or not filename.lower().endswith(".png"):
        raise ValueError("6-DOF plot filename must be a local PNG name")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    time_s = result.time_s
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True)
        axes[0, 0].plot(time_s, result.columns["altitude_m"] / 1_000.0, color=NAVY)
        axes[0, 0].set(ylabel="Altitude [km]", title="Ascent altitude")
        axes[0, 1].plot(time_s, result.columns["attitude_error_deg"], color=RED)
        axes[0, 1].set(ylabel="Quaternion error [deg]", title="Attitude tracking error")
        axes[1, 0].plot(time_s, result.columns["roll_rate_degps"], color=BLUE, label="p")
        axes[1, 0].plot(time_s, result.columns["pitch_rate_degps"], color=ORANGE, label="q")
        axes[1, 0].plot(time_s, result.columns["yaw_rate_degps"], color=GREEN, label="r")
        axes[1, 0].set(xlabel="Time [s]", ylabel="Body rate [deg/s]", title="Angular rates")
        axes[1, 0].legend(ncol=3)
        axes[1, 1].plot(time_s, result.columns["alpha_deg"], color=BLUE, label="Angle of attack")
        axes[1, 1].plot(time_s, result.columns["beta_deg"], color=ORANGE, label="Sideslip")
        axes[1, 1].set(
            xlabel="Time [s]", ylabel="Aerodynamic angle [deg]", title="Aerodynamic angles"
        )
        axes[1, 1].legend(ncol=2, fontsize=8)
        for axis in axes.flat:
            for event in result.events:
                axis.axvline(event.time_s, color=NAVY, linewidth=0.7, linestyle="--", alpha=0.5)
        figure.suptitle(
            f"{result.scenario_name.replace('_', ' ')} — quaternion 6-DOF ascent",
            color=NAVY,
            fontweight="bold",
        )
        path = output / filename
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
