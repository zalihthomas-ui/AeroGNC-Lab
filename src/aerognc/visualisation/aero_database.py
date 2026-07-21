"""Aerodynamic database coefficient and derivative visualization."""

import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.vehicle.aero_database import (
    COEFFICIENT_NAMES,
    AerodynamicCondition,
    TabulatedAerodynamicDatabase,
)
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, engineering_style


def plot_aerodynamic_database(
    database: TabulatedAerodynamicDatabase,
    output_directory: str | Path,
    nominal_condition: AerodynamicCondition | None = None,
) -> Path:
    """Plot coefficient slices and the local table-derived Jacobian."""
    if "mach" not in database.axis_names or "alpha_rad" not in database.axis_names:
        raise ValueError("aerodynamic database plot requires mach and alpha_rad axes")
    nominal = nominal_condition or AerodynamicCondition(mach=0.8)
    axes_by_name = dict(zip(database.axis_names, database.axes, strict=True))
    mach_values = axes_by_name["mach"]
    selected_mach = np.unique(
        np.array([mach_values[0], mach_values[mach_values.size // 2], mach_values[-1]])
    )
    alpha_values = np.linspace(axes_by_name["alpha_rad"][0], axes_by_name["alpha_rad"][-1], 121)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with engineering_style():
        figure, plot_axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
        coefficient_panels = (
            ("drag", plot_axes[0, 0], NAVY),
            ("normal", plot_axes[0, 1], BLUE),
            ("pitch", plot_axes[1, 0], GREEN),
        )
        for coefficient_name, axis, color in coefficient_panels:
            for index, mach in enumerate(selected_mach):
                values = [
                    getattr(
                        database.evaluate(
                            replace(nominal, mach=float(mach), alpha_rad=float(alpha))
                        ),
                        coefficient_name,
                    )
                    for alpha in alpha_values
                ]
                axis.plot(
                    np.rad2deg(alpha_values),
                    values,
                    color=color if index == 0 else ORANGE if index == 1 else BLUE,
                    label=f"Mach {mach:.2g}",
                )
            axis.axhline(0.0, color="#66717E", linewidth=0.7)
            axis.set(
                title=coefficient_name.title() + " coefficient",
                xlabel="Angle of attack [deg]",
                ylabel=f"C{coefficient_name[0]} [-]",
            )
            axis.legend(ncol=min(3, selected_mach.size))

        axis_names, jacobian = database.coefficient_jacobian(nominal)
        image = plot_axes[1, 1].imshow(jacobian, aspect="auto", cmap="coolwarm")
        plot_axes[1, 1].set_xticks(np.arange(len(axis_names)), axis_names, rotation=30, ha="right")
        plot_axes[1, 1].set_yticks(np.arange(len(COEFFICIENT_NAMES)), COEFFICIENT_NAMES)
        plot_axes[1, 1].set_title("Local coefficient Jacobian")
        figure.colorbar(image, ax=plot_axes[1, 1], label="Partial derivative")
        figure.suptitle(
            "Synthetic aerodynamic database and table-derived stability derivatives",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "aerodynamic_database_analysis.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
