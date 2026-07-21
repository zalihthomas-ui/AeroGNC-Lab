"""Publication-style point-mass trajectory figures."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from aerognc.simulation.logging import SimulationResult
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, RED, engineering_style


def _mark_events(axis: Axes, result: SimulationResult) -> None:
    for event in result.events:
        axis.axvline(event.time_s, color=GREY, linewidth=0.7, linestyle="--", alpha=0.55)


def plot_three_dof_results(
    result: SimulationResult, output_directory: str | Path
) -> tuple[Path, ...]:
    """Generate high-resolution kinematics, loads, and trajectory PNG figures."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), constrained_layout=True)
        time_s = result.time_s
        axes[0, 0].plot(time_s, result.columns["altitude_m"] / 1_000.0, color=NAVY)
        axes[0, 0].set(ylabel="Altitude [km]", title="Altitude")
        axes[0, 1].plot(
            time_s, result.columns["vertical_velocity_up_mps"], label="Vertical", color=BLUE
        )
        axes[0, 1].plot(time_s, result.columns["total_velocity_mps"], label="Total", color=ORANGE)
        axes[0, 1].set(ylabel="Velocity [m/s]", title="Velocity")
        axes[0, 1].legend(ncol=2)
        axes[1, 0].plot(time_s, result.columns["acceleration_magnitude_mps2"] / 9.80665, color=RED)
        axes[1, 0].set(xlabel="Time [s]", ylabel="Acceleration [g]", title="Total acceleration")
        axes[1, 1].plot(time_s, result.columns["flight_path_angle_deg"], color=GREEN)
        axes[1, 1].set(
            xlabel="Time [s]", ylabel="Flight-path angle [deg]", title="Flight-path angle"
        )
        for axis in axes.flat:
            _mark_events(axis, result)
        figure.suptitle(
            f"{result.scenario_name} — point-mass kinematics", color=NAVY, fontweight="bold"
        )
        path = output / "three_dof_kinematics.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        paths.append(path)

        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), constrained_layout=True)
        axes[0, 0].plot(time_s, result.columns["mach"], color=NAVY)
        axes[0, 0].set(ylabel="Mach number [-]", title="Mach number")
        axes[0, 1].plot(time_s, result.columns["dynamic_pressure_pa"] / 1_000.0, color=BLUE)
        axes[0, 1].set(ylabel="Dynamic pressure [kPa]", title="Dynamic pressure")
        axes[1, 0].plot(time_s, result.columns["mass_kg"], color=GREEN)
        axes[1, 0].set(xlabel="Time [s]", ylabel="Mass [kg]", title="Vehicle mass")
        axes[1, 1].plot(time_s, result.columns["thrust_n"] / 1_000.0, label="Thrust", color=ORANGE)
        axes[1, 1].plot(time_s, result.columns["drag_n"] / 1_000.0, label="Drag", color=RED)
        axes[1, 1].set(xlabel="Time [s]", ylabel="Force [kN]", title="Axial loads")
        axes[1, 1].legend(ncol=2)
        for axis in axes.flat:
            _mark_events(axis, result)
        figure.suptitle(
            f"{result.scenario_name} — atmosphere and loads", color=NAVY, fontweight="bold"
        )
        path = output / "three_dof_loads.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        paths.append(path)

        figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        axis.plot(
            result.columns["ground_range_m"] / 1_000.0,
            result.columns["altitude_m"] / 1_000.0,
            color=NAVY,
        )
        for event in result.events:
            horizontal = np.interp(event.time_s, result.time_s, result.columns["ground_range_m"])
            altitude = np.interp(event.time_s, result.time_s, result.columns["altitude_m"])
            axis.scatter(horizontal / 1_000.0, altitude / 1_000.0, s=28, color=ORANGE, zorder=3)
            axis.annotate(
                event.name.replace("_", " ").title(),
                (horizontal / 1_000.0, altitude / 1_000.0),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set(
            xlabel="Ground range [km]",
            ylabel="Altitude [km]",
            title=f"{result.scenario_name} — NED ground-range trajectory",
        )
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
        path = output / "three_dof_trajectory.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)
