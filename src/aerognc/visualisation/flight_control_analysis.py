"""Flight-control linear-analysis summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aerognc.verification.flight_control_analysis import FlightControlAnalysisResult
from aerognc.visualisation.style import engineering_style


def plot_flight_control_analysis(
    result: FlightControlAnalysisResult, output_directory: str | Path
) -> Path:
    """Plot closed-loop modes, open-loop response, and scheduled feedback gains."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
        pole_axis, magnitude_axis, phase_axis, schedule_axis = axes.ravel()
        eigenvalues = result.lqr.closed_loop_eigenvalues
        pole_axis.scatter(eigenvalues.real, eigenvalues.imag, marker="x", s=70, color="#C44E52")
        pole_axis.axvline(0.0, color="black", linewidth=0.8)
        pole_axis.set(title="Closed-loop modes", xlabel="Real [rad/s]", ylabel="Imaginary [rad/s]")

        frequency = result.angular_frequency_radps
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(result.open_loop_response), 1.0e-16))
        phase_deg = np.unwrap(np.angle(result.open_loop_response)) * 180.0 / np.pi
        magnitude_axis.semilogx(frequency, magnitude_db, color="#2878B5")
        magnitude_axis.axhline(0.0, color="black", linewidth=0.8)
        magnitude_axis.set(title="Open-loop magnitude", ylabel="Magnitude [dB]")
        phase_axis.semilogx(frequency, phase_deg, color="#E18727")
        phase_axis.axhline(-180.0, color="black", linewidth=0.8)
        phase_axis.set(
            title=f"Phase margin {result.margins.phase_margin_deg:.1f}°",
            xlabel="Frequency [rad/s]",
            ylabel="Phase [deg]",
        )
        for state_index, label in enumerate(("Angle gain", "Rate gain")):
            schedule_axis.plot(
                result.scheduling_points,
                result.scheduled_gains[:, state_index],
                marker="o",
                label=label,
            )
        schedule_axis.set(
            title="LQR gain schedule",
            xlabel="Synthetic scheduling variable [-]",
            ylabel="Feedback gain",
        )
        schedule_axis.legend()
        for axis in axes.ravel():
            axis.grid(True, alpha=0.28)
        figure.suptitle("Fictional civilian pitch-channel linear analysis", fontweight="bold")
        path = output / "flight_control_analysis.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
    return path
