"""Publication-style launch-window and mission-plan visualisations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from aerognc.astrodynamics.mission_design import PorkchopGrid


def plot_porkchop_grid(
    grid: PorkchopGrid,
    *,
    departure_epoch_day: float = 0.0,
    title: str = "Synthetic launch-window analysis",
) -> Figure:
    """Create C3 contours with arrival-speed lines and infeasible shading."""
    departure_days = grid.departure_time_s / 86_400.0 - departure_epoch_day
    arrival_days = grid.arrival_time_s / 86_400.0 - departure_epoch_day
    c3_km2_s2 = grid.departure_c3_m2_s2 / 1.0e6
    arrival_speed_kmps = grid.arrival_excess_speed_mps / 1_000.0
    finite_c3 = np.ma.masked_invalid(c3_km2_s2.T)
    finite_arrival = np.ma.masked_invalid(arrival_speed_kmps.T)

    figure, axis = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    figure.patch.set_facecolor("#07111F")
    axis.set_facecolor("#0D1B2A")
    filled = axis.contourf(
        departure_days,
        arrival_days,
        finite_c3,
        levels=18,
        cmap="viridis",
    )
    arrival_contours = axis.contour(
        departure_days,
        arrival_days,
        finite_arrival,
        levels=8,
        colors="#F2D0A4",
        linewidths=0.8,
    )
    axis.clabel(arrival_contours, fmt="%.1f km/s", fontsize=8, colors="#F2D0A4")
    axis.contourf(
        departure_days,
        arrival_days,
        (~grid.feasible).T.astype(float),
        levels=[0.5, 1.5],
        colors=["#7E2F43"],
        alpha=0.25,
    )
    if np.any(grid.feasible):
        departure_index, arrival_index = grid.best_indices()
        axis.scatter(
            departure_days[departure_index],
            arrival_days[arrival_index],
            marker="*",
            s=180,
            color="#5FD19A",
            edgecolor="white",
            linewidth=0.8,
            label="Best feasible grid point",
            zorder=5,
        )
        axis.legend(loc="best")
    colorbar = figure.colorbar(filled, ax=axis, pad=0.02)
    colorbar.set_label("Departure C3 [km²/s²]", color="#D9E8F2")
    colorbar.ax.tick_params(colors="#D9E8F2")
    axis.set(
        title=title,
        xlabel="Departure epoch [catalog day]",
        ylabel="Arrival epoch [catalog day]",
    )
    axis.grid(color="#49657A", alpha=0.3, linewidth=0.6)
    axis.tick_params(colors="#D9E8F2")
    axis.xaxis.label.set_color("#D9E8F2")
    axis.yaxis.label.set_color("#D9E8F2")
    axis.title.set_color("#D9E8F2")
    for spine in axis.spines.values():
        spine.set_color("#49657A")
    return figure


def show_porkchop_grid(grid: PorkchopGrid, *, title: str) -> None:
    """Open an interactive launch-window figure."""
    plot_porkchop_grid(grid, title=title)
    plt.show()
