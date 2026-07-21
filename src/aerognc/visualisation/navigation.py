"""Truth/measurement/estimate navigation figures."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerognc.simulation.navigation_demo import NavigationDemoResult
from aerognc.visualisation.style import BLUE, GREY, NAVY, ORANGE, RED, engineering_style


def plot_navigation_demo(result: NavigationDemoResult, output_directory: str | Path) -> Path:
    """Plot truth, sparse measurements, estimate errors, and two-sigma bounds."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    time_s = result.time_s
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True)
        axes[0, 0].plot(time_s, result.true_altitude_m, color=NAVY, label="Truth")
        axes[0, 0].scatter(
            time_s,
            result.measured_barometric_altitude_m,
            color=GREY,
            s=5,
            alpha=0.45,
            label="Barometer",
        )
        axes[0, 0].plot(time_s, result.estimated_altitude_m, color=ORANGE, label="Estimate")
        axes[0, 0].set(ylabel="Altitude [m]", title="Vertical position")
        axes[0, 0].legend(ncol=3, fontsize=8)
        axes[0, 1].plot(time_s, result.true_vertical_velocity_up_mps, color=NAVY, label="Truth")
        axes[0, 1].scatter(
            time_s,
            result.measured_gnss_vertical_velocity_up_mps,
            color=GREY,
            s=7,
            alpha=0.55,
            label="GNSS-like",
        )
        axes[0, 1].plot(
            time_s, result.estimated_vertical_velocity_up_mps, color=BLUE, label="Estimate"
        )
        axes[0, 1].set(ylabel="Upward velocity [m/s]", title="Vertical velocity")
        axes[0, 1].legend(ncol=3, fontsize=8)
        altitude_error = result.estimated_altitude_m - result.true_altitude_m
        axes[1, 0].plot(time_s, altitude_error, color=RED, label="Error")
        axes[1, 0].fill_between(
            time_s,
            -2.0 * result.altitude_sigma_m,
            2.0 * result.altitude_sigma_m,
            color=BLUE,
            alpha=0.16,
            label="±2 sigma",
        )
        axes[1, 0].set(xlabel="Time [s]", ylabel="Altitude error [m]", title="Altitude error")
        axes[1, 0].legend(ncol=2)
        velocity_error = (
            result.estimated_vertical_velocity_up_mps - result.true_vertical_velocity_up_mps
        )
        axes[1, 1].plot(time_s, velocity_error, color=RED, label="Error")
        axes[1, 1].fill_between(
            time_s,
            -2.0 * result.velocity_sigma_mps,
            2.0 * result.velocity_sigma_mps,
            color=BLUE,
            alpha=0.16,
            label="±2 sigma",
        )
        axes[1, 1].set(
            xlabel="Time [s]", ylabel="Velocity error [m/s]", title="Vertical-velocity error"
        )
        axes[1, 1].legend(ncol=2)
        figure.suptitle(
            "Synthetic delayed-sensor vertical navigation", color=NAVY, fontweight="bold"
        )
        path = output / "navigation_filter.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
