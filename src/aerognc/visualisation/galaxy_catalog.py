"""Publication-style view of the confirmed-exoplanet catalog snapshot."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.catalogs import ConfirmedExoplanet, ExoplanetCatalog
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, engineering_style


def plot_milky_way_catalog(
    catalog: ExoplanetCatalog,
    selection: tuple[ConfirmedExoplanet, ...],
    output_directory: str | Path,
) -> Path:
    """Plot positions, discovery history, methods, and field completeness."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    positioned = tuple(planet for planet in selection if planet.has_3d_position)
    _, positions = catalog.galactic_positions_pc(positioned)
    years = np.array(
        [planet.discovery_year or 0 for planet in positioned],
        dtype=np.float64,
    )
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), constrained_layout=True)
        if positions.size:
            scatter = axes[0, 0].scatter(
                positions[:, 0],
                positions[:, 1],
                c=years,
                s=8,
                alpha=0.58,
                linewidths=0.0,
                cmap="viridis",
            )
            figure.colorbar(scatter, ax=axes[0, 0], label="Discovery year")
            axes[0, 0].scatter(0.0, 0.0, color=ORANGE, marker="*", s=120, label="Sun")
            axes[0, 0].legend(loc="upper right")
        else:
            axes[0, 0].text(0.5, 0.5, "No complete 3D positions", ha="center", va="center")
        axes[0, 0].set(
            xlabel="Galactic x toward centre [pc]",
            ylabel="Galactic y [pc]",
            title="Heliocentric confirmed detections (top view)",
        )
        axes[0, 0].set_aspect("equal", adjustable="datalim")

        if positions.size:
            axes[0, 1].scatter(
                positions[:, 0],
                positions[:, 2],
                c=years,
                s=8,
                alpha=0.58,
                linewidths=0.0,
                cmap="viridis",
            )
            axes[0, 1].scatter(0.0, 0.0, color=ORANGE, marker="*", s=120)
        else:
            axes[0, 1].text(0.5, 0.5, "No complete 3D positions", ha="center", va="center")
        axes[0, 1].set(
            xlabel="Galactic x toward centre [pc]",
            ylabel="Galactic z north [pc]",
            title="Heliocentric confirmed detections (side view)",
        )

        discovered_years = [planet.discovery_year for planet in selection if planet.discovery_year]
        if discovered_years:
            first_year = min(discovered_years)
            last_year = max(discovered_years)
            bins = np.arange(first_year, last_year + 2) - 0.5
            axes[1, 0].hist(discovered_years, bins=bins, color=BLUE, edgecolor="white")
        axes[1, 0].set(
            xlabel="Discovery year",
            ylabel="Confirmed planets in selection",
            title="Discovery history (not occurrence rate)",
        )

        field_counts = {
            "3D position": sum(planet.has_3d_position for planet in selection),
            "Period": sum(planet.orbital_period_days is not None for planet in selection),
            "Mass": sum(planet.mass_earth is not None for planet in selection),
            "Radius": sum(planet.radius_earth is not None for planet in selection),
        }
        labels = tuple(field_counts)
        completeness = np.array(tuple(field_counts.values()), dtype=np.float64)
        if selection:
            completeness *= 100.0 / len(selection)
        axes[1, 1].bar(labels, completeness, color=(GREEN, BLUE, ORANGE, GREY))
        axes[1, 1].set_ylim(0.0, 105.0)
        axes[1, 1].set(
            ylabel="Rows with reported value [%]",
            title="Selected-field completeness",
        )
        for index, value in enumerate(completeness):
            axes[1, 1].text(index, value + 1.5, f"{value:.1f}%", ha="center", fontsize=8)

        method_counts = Counter(planet.discovery_method for planet in selection)
        leading_methods = ", ".join(
            f"{method}: {count}" for method, count in method_counts.most_common(3)
        )
        axes[1, 0].text(
            0.02,
            0.96,
            leading_methods,
            transform=axes[1, 0].transAxes,
            va="top",
            color=NAVY,
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": GREY},
        )
        figure.suptitle(
            "Milky Way context — NASA confirmed-exoplanet snapshot",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "milky_way_confirmed_exoplanet_catalog.png"
        figure.savefig(path, dpi=220, pad_inches=0.15)
        plt.close(figure)
    return path
