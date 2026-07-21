"""Monte Carlo distributions, ECDF, sensitivity, and requirement figures."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.configuration.monte_carlo_loader import MonteCarloRequirements
from aerognc.simulation.monte_carlo import MonteCarloSummary
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style


def plot_monte_carlo_summary(
    summary: MonteCarloSummary,
    requirements: MonteCarloRequirements,
    output_directory: str | Path,
) -> Path:
    """Plot a compact ensemble distribution and requirement dashboard."""
    successful = [run for run in summary.runs if run.success]
    if not successful:
        raise ValueError("cannot plot a Monte Carlo ensemble with no successful runs")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    apogee = np.array([run.metrics["apogee_m"] for run in successful])
    dynamic_pressure_kpa = np.array(
        [run.metrics["maximum_dynamic_pressure_pa"] / 1_000.0 for run in successful]
    )
    mass_scale = np.array([run.sample["vehicle_mass_scale"] for run in successful])
    wind_scale = np.array([run.sample["wind_scale"] for run in successful])
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), constrained_layout=True)
        bins = max(5, min(12, len(successful) // 2 + 2))
        axes[0, 0].hist(apogee, bins=bins, color=BLUE, alpha=0.82, edgecolor="white")
        axes[0, 0].axvline(
            requirements.minimum_apogee_m,
            color=RED,
            linestyle="--",
            label="Requirement",
        )
        axes[0, 0].set(xlabel="Apogee [m]", ylabel="Run count", title="Apogee distribution")
        axes[0, 0].legend()

        sorted_pressure = np.sort(dynamic_pressure_kpa)
        probability = np.arange(1, sorted_pressure.size + 1) / sorted_pressure.size
        axes[0, 1].step(sorted_pressure, probability, where="post", color=NAVY)
        axes[0, 1].axvline(
            requirements.maximum_dynamic_pressure_pa / 1_000.0,
            color=RED,
            linestyle="--",
            label="Maximum",
        )
        axes[0, 1].set(
            xlabel="Maximum dynamic pressure [kPa]",
            ylabel="Empirical cumulative probability",
            title="Dynamic-pressure ECDF",
        )
        axes[0, 1].set_ylim(0.0, 1.02)
        axes[0, 1].legend()

        scatter = axes[1, 0].scatter(
            mass_scale,
            apogee,
            c=wind_scale,
            cmap="viridis",
            s=34,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[1, 0].axhline(requirements.minimum_apogee_m, color=RED, linestyle="--")
        axes[1, 0].set(
            xlabel="Vehicle mass scale [-]",
            ylabel="Apogee [m]",
            title="Dispersion interaction",
        )
        colorbar = figure.colorbar(scatter, ax=axes[1, 0], pad=0.02)
        colorbar.set_label("Wind scale [-]")

        pass_rates = summary.requirement_pass_rates
        names = [name for name in pass_rates if name != "overall"]
        labels = [
            name.replace("minimum_", "min ").replace("maximum_", "max ").replace("_", " ")
            for name in names
        ]
        rates = np.array([pass_rates[name] * 100.0 for name in names])
        colors = [GREEN if rate >= 95.0 else ORANGE if rate >= 80.0 else RED for rate in rates]
        axes[1, 1].barh(labels, rates, color=colors)
        axes[1, 1].axvline(95.0, color=NAVY, linestyle="--", linewidth=0.9)
        axes[1, 1].set(
            xlabel="Passing runs [%]",
            title="Requirement pass assessment",
            xlim=(0.0, 102.0),
        )
        figure.suptitle(
            f"{summary.name} — seeded coupled Monte Carlo (n={len(summary.runs)})",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "monte_carlo_summary.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path


def plot_monte_carlo_sensitivity(
    summary: MonteCarloSummary,
    output_directory: str | Path,
) -> Path:
    """Plot Pearson linear correlations for core output metrics."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    successful = [run for run in summary.runs if run.success]
    if len(successful) < 2:
        raise ValueError("at least two successful runs are needed for sensitivity")
    parameter_names = list(successful[0].sample)
    metric_names = [
        "apogee_m",
        "maximum_dynamic_pressure_pa",
        "navigation_altitude_rms_m",
        "control_settling_time_s",
    ]
    correlation_matrix = np.full((len(parameter_names), len(metric_names)), np.nan)
    for row, parameter in enumerate(parameter_names):
        for column, metric in enumerate(metric_names):
            key = f"{parameter}__{metric}"
            if key in summary.correlations:
                correlation_matrix[row, column] = summary.correlations[key]
    parameter_labels = [
        name.replace("_offset", " offset")
        .replace("_scale", " scale")
        .replace("_deg", "")
        .replace("_mps", "")
        .replace("_", " ")
        for name in parameter_names
    ]
    metric_labels = ["Apogee", "Maximum q", "Navigation RMS", "Control settling"]
    with engineering_style():
        figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
        image = axis.imshow(
            correlation_matrix,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
        )
        axis.set_xticks(np.arange(len(metric_names)), labels=metric_labels)
        axis.set_yticks(np.arange(len(parameter_names)), labels=parameter_labels)
        for row in range(correlation_matrix.shape[0]):
            for column in range(correlation_matrix.shape[1]):
                value = correlation_matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if abs(value) > 0.55 else NAVY,
                    )
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label("Pearson correlation [-]")
        axis.set_title(
            "Linear sensitivity screening (synthetic ensemble)",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "monte_carlo_sensitivity.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
