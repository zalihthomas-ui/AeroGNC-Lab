"""Publication-style rotating-planet ascent diagnostics."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerognc.simulation.logging import SimulationResult
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, engineering_style


def plot_rotating_ascent(result: SimulationResult, output_directory: str | Path) -> Path:
    """Plot geodetic trajectory, rotating-frame velocity, and load diagnostics."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    time_s = result.time_s
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
        axes[0, 0].plot(time_s, result.columns["altitude_m"] / 1_000.0, color=NAVY)
        axes[0, 0].set(title="Ellipsoidal altitude above launch site", ylabel="Altitude [km]")

        axes[0, 1].plot(
            time_s,
            -result.columns["velocity_down_mps"],
            color=BLUE,
            label="Vertical up",
        )
        axes[0, 1].plot(
            time_s,
            result.columns["total_velocity_mps"],
            color=ORANGE,
            label="ECEF magnitude",
        )
        axes[0, 1].set(title="Rotating-frame velocity", ylabel="Velocity [m/s]")
        axes[0, 1].legend(ncol=2)

        axes[1, 0].plot(
            result.columns["east_m"] / 1_000.0,
            result.columns["north_m"] / 1_000.0,
            color=GREEN,
        )
        axes[1, 0].scatter([0.0], [0.0], color=ORANGE, s=24, label="Launch site")
        axes[1, 0].set(
            title="Local tangent-plane ground track",
            xlabel="East [km]",
            ylabel="North [km]",
            aspect="equal",
        )
        axes[1, 0].legend()

        axes[1, 1].plot(time_s, result.columns["mach"], color=NAVY, label="Mach")
        pressure_axis = axes[1, 1].twinx()
        pressure_axis.grid(False)
        pressure_axis.plot(
            time_s,
            result.columns["dynamic_pressure_pa"] / 1_000.0,
            color=ORANGE,
            label="Dynamic pressure",
        )
        axes[1, 1].set(title="Atmospheric loading", xlabel="Time [s]", ylabel="Mach [-]")
        pressure_axis.set_ylabel("Dynamic pressure [kPa]")
        handles_a, labels_a = axes[1, 1].get_legend_handles_labels()
        handles_b, labels_b = pressure_axis.get_legend_handles_labels()
        axes[1, 1].legend(handles_a + handles_b, labels_a + labels_b, loc="best")

        for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
            for event in result.events:
                axis.axvline(event.time_s, color=GREY, linestyle="--", linewidth=0.7, alpha=0.6)
        figure.suptitle(
            "Fictional rotating-oblate-planet research ascent",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "rotating_planet_ascent.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
