"""Publication-style multistage and recovery evidence plot."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerognc.simulation.logging import SimulationResult
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style


def plot_multistage_recovery(result: SimulationResult, output_directory: str | Path) -> Path:
    """Plot trajectory, propulsion/mass, recovery area, and opening load."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    time_s = result.time_s
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True)
        axes[0, 0].plot(time_s, result.columns["altitude_m"], color=NAVY)
        axes[0, 0].set(ylabel="Altitude [m]", title="Vertical trajectory")
        axes[0, 1].plot(time_s, result.columns["vertical_velocity_mps"], color=BLUE)
        axes[0, 1].axhline(0.0, color=NAVY, linewidth=0.7)
        axes[0, 1].set(ylabel="Vertical velocity [m/s]", title="Ascent and descent")
        mass_axis = axes[1, 0]
        thrust_axis = mass_axis.twinx()
        mass_axis.plot(time_s, result.columns["mass_kg"], color=GREEN, label="Mass")
        mass_axis.plot(
            time_s,
            result.columns["retained_dry_mass_floor_kg"],
            color=NAVY,
            linestyle="--",
            label="Dry floor",
        )
        thrust_axis.plot(time_s, result.columns["thrust_n"], color=ORANGE, label="Thrust")
        mass_axis.set(xlabel="Time [s]", ylabel="Mass [kg]", title="Stage accounting")
        thrust_axis.set_ylabel("Thrust [N]")
        mass_axis.legend(loc="upper right", fontsize=8)
        area_axis = axes[1, 1]
        load_axis = area_axis.twinx()
        area_axis.plot(
            time_s, result.columns["recovery_drag_area_m2"], color=BLUE, label="Drag area"
        )
        load_axis.plot(time_s, result.columns["opening_load_n"], color=RED, label="Opening load")
        area_axis.set(
            xlabel="Time [s]", ylabel="Recovery area [m$^2$]", title="Recovery deployment"
        )
        load_axis.set_ylabel("Opening load [N]")
        for axis in axes.flat:
            for event in result.events:
                axis.axvline(event.time_s, color=NAVY, linewidth=0.6, linestyle=":", alpha=0.35)
        figure.suptitle(
            f"{result.scenario_name.replace('_', ' ')} — fictional multistage recovery",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "multistage_recovery.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
