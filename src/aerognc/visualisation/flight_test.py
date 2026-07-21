"""Synthetic flight-test reconstruction and event comparison figure."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.verification.flight_test import FlightTestWorkflowResult
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, RED, engineering_style


def plot_flight_test_summary(
    workflow: FlightTestWorkflowResult,
    output_directory: str | Path,
) -> Path:
    """Plot expected/reconstructed states, event errors, and measurement availability."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    analysis = workflow.analysis
    truth = workflow.truth
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True)
        axes[0, 0].plot(truth.time_s, truth.columns["altitude_m"], color=NAVY, label="Expected")
        axes[0, 0].scatter(
            analysis.time_s,
            analysis.measured_barometric_altitude_m,
            color=GREY,
            s=5,
            alpha=0.4,
            label="Measured",
        )
        axes[0, 0].plot(
            analysis.time_s,
            analysis.reconstructed_altitude_m,
            color=ORANGE,
            label="Reconstructed",
        )
        axes[0, 0].set(ylabel="Altitude [m]", title="Altitude reconstruction")
        axes[0, 0].legend(ncol=3, fontsize=8)
        axes[0, 1].plot(
            truth.time_s,
            truth.columns["vertical_velocity_up_mps"],
            color=NAVY,
            label="Expected",
        )
        axes[0, 1].plot(
            analysis.time_s,
            analysis.reconstructed_vertical_velocity_up_mps,
            color=BLUE,
            label="Reconstructed",
        )
        axes[0, 1].set(ylabel="Upward velocity [m/s]", title="Vertical velocity")
        axes[0, 1].legend(ncol=2)
        event_names = ["burnout", "apogee", "ground impact"]
        event_errors = [
            workflow.event_time_errors_s["burnout"],
            workflow.event_time_errors_s["apogee"],
            workflow.event_time_errors_s["ground_impact"],
        ]
        axes[1, 0].bar(event_names, event_errors, color=[BLUE, ORANGE, GREEN])
        axes[1, 0].axhline(0.0, color=NAVY, linewidth=0.8)
        axes[1, 0].set(
            xlabel="Detected event", ylabel="Timing error [s]", title="Event reconstruction error"
        )
        available = [
            np.mean(np.isfinite(analysis.measured_acceleration_up_mps2)) * 100.0,
            np.mean(np.isfinite(analysis.measured_barometric_altitude_m)) * 100.0,
            np.mean(np.isfinite(analysis.measured_gnss_altitude_m)) * 100.0,
        ]
        axes[1, 1].barh(["Accelerometer", "Barometer", "GNSS-like"], available, color=RED)
        axes[1, 1].set(
            xlabel="Available rows [%]",
            title="Measurement availability at common 50 Hz log rate",
            xlim=(0.0, 105.0),
        )
        figure.suptitle(
            "Synthetic flight-test reconstruction and event evaluation",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "flight_test_summary.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return path
