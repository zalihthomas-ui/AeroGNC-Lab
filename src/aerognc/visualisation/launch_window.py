"""Publication-style launch-window grid and refined optimum figure."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.verification.launch_window import LaunchWindowRun
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, RED, engineering_style


def plot_launch_window_optimization(
    run: LaunchWindowRun,
    output_directory: str | Path,
) -> Path:
    """Plot the complete grid, feasibility, refined history, and delta-v split."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    optimization = run.optimization
    departure_days = optimization.departure_grid_s / 86_400.0
    arrival_days = optimization.arrival_grid_s / 86_400.0
    delta_v_kmps = optimization.total_delta_v_grid_mps.T / 1_000.0
    with engineering_style():
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
        contour = axes[0].contourf(
            departure_days,
            arrival_days,
            delta_v_kmps,
            levels=16,
            cmap="viridis_r",
        )
        figure.colorbar(contour, ax=axes[0], label="Injection + capture Δv [km/s]")
        infeasible = ~optimization.feasible_grid.T
        axes[0].contourf(
            departure_days,
            arrival_days,
            infeasible.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
        )
        history_departure = [item.departure_time_s / 86_400.0 for item in optimization.history]
        history_arrival = [item.arrival_time_s / 86_400.0 for item in optimization.history]
        axes[0].plot(
            history_departure,
            history_arrival,
            color=RED,
            marker="o",
            ms=3,
            lw=1.0,
            label="Bounded refinement",
        )
        optimum = optimization.optimum
        axes[0].scatter(
            optimum.departure_time_s / 86_400.0,
            optimum.arrival_time_s / 86_400.0,
            color=ORANGE,
            edgecolor=NAVY,
            marker="*",
            s=160,
            label="Selected opportunity",
        )
        axes[0].set(
            xlabel="Departure epoch [synthetic catalog day]",
            ylabel="Arrival epoch [synthetic catalog day]",
            title="Complete constrained launch-window screen",
        )
        axes[0].legend(fontsize=8)

        component_names = ("Parking-orbit\ninjection", "Destination\ncapture")
        component_values = (
            np.array([optimum.injection_delta_v_mps, optimum.capture_delta_v_mps]) / 1_000.0
        )
        axes[1].bar(component_names, component_values, color=(BLUE, GREEN), width=0.55)
        for index, value in enumerate(component_values):
            axes[1].text(index, value + 0.08, f"{value:.3f}", ha="center")
        axes[1].axhline(0.0, color=GREY, lw=0.8)
        axes[1].set(
            ylabel="Ideal impulsive Δv [km/s]",
            title=(
                f"Refined total {optimum.total_delta_v_mps / 1_000.0:.3f} km/s\n"
                f"day {optimum.departure_time_s / 86_400.0:.3f} → "
                f"{optimum.arrival_time_s / 86_400.0:.3f}"
            ),
        )
        axes[1].text(
            0.5,
            0.97,
            (
                f"C3 {optimum.transfer.departure_c3_m2_s2 / 1.0e6:.2f} km²/s²\n"
                f"arrival v∞ {optimum.transfer.arrival_excess_speed_mps / 1_000.0:.2f} km/s\n"
                f"endpoint error {run.endpoint_error_m:.3e} m\n"
                f"{optimization.evaluation_count} deterministic evaluations"
            ),
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            color=NAVY,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": GREY},
        )
        figure.suptitle(
            "Fictional civilian launch-window optimization",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "launch_window_optimization.png"
        figure.savefig(path, dpi=220, pad_inches=0.12)
        plt.close(figure)
    return path
