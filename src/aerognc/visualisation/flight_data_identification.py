"""Publication-style asynchronous flight-data identification diagnostics."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.verification.flight_data_identification import FlightDataIdentificationResult
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, RED, engineering_style


def plot_flight_data_identification(
    result: FlightDataIdentificationResult,
    output_directory: str | Path,
) -> Path:
    """Plot clock recovery, cleaning, parameters, validation, and residuals."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    configuration = result.configuration
    truth = {
        "inertia": configuration.plant.inertia_kgm2,
        "damping": configuration.plant.damping_nms_per_rad,
        "stiffness": configuration.plant.stiffness_nm_per_rad,
        "disturbance_moment": configuration.plant.disturbance_moment_nm,
    }
    alignment = result.clock_alignment
    corrected_centres = (alignment.sensor_marker_times_s - alignment.offset_s) / alignment.scale
    marker_residual_ms = 1.0e3 * (corrected_centres - alignment.reference_marker_times_s)
    with engineering_style():
        figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), constrained_layout=True)
        axes[0, 0].axhline(0.0, color=GREY, lw=0.9)
        axes[0, 0].scatter(
            alignment.reference_marker_times_s,
            marker_residual_ms,
            color=BLUE,
            s=30,
        )
        axes[0, 0].set(
            xlabel="Reference marker time [s]",
            ylabel="Aligned marker residual [ms]",
            title=(f"Clock recovery: {alignment.offset_s:.4f} s, {alignment.drift_ppm:.1f} ppm"),
        )

        axes[0, 1].plot(
            result.time_s,
            np.rad2deg(result.resampled_pitch_rad),
            color=GREY,
            alpha=0.55,
            label="Aligned raw",
        )
        axes[0, 1].plot(
            result.time_s,
            np.rad2deg(result.cleaned_pitch_rad),
            color=NAVY,
            label="Robust clean",
        )
        flagged = result.pitch_outlier_mask | result.rate_outlier_mask
        axes[0, 1].scatter(
            result.time_s[flagged],
            np.rad2deg(result.resampled_pitch_rad[flagged]),
            color=RED,
            marker="x",
            s=28,
            label="Flagged sample",
        )
        axes[0, 1].set(
            xlabel="Aligned time [s]",
            ylabel="Pitch angle [deg]",
            title="Gap preservation and outlier treatment",
        )
        axes[0, 1].legend()

        display_names = {
            "inertia": "Inertia",
            "damping": "Damping",
            "stiffness": "Stiffness",
            "disturbance_moment": "Disturbance moment",
        }
        parameter_names = [display_names[estimate.name] for estimate in result.parameter_estimates]
        relative_estimate = np.array(
            [
                100.0 * (estimate.estimate - truth[estimate.name]) / truth[estimate.name]
                for estimate in result.parameter_estimates
            ]
        )
        relative_lower = np.array(
            [
                100.0 * (estimate.estimate - estimate.lower_95) / truth[estimate.name]
                for estimate in result.parameter_estimates
            ]
        )
        relative_upper = np.array(
            [
                100.0 * (estimate.upper_95 - estimate.estimate) / truth[estimate.name]
                for estimate in result.parameter_estimates
            ]
        )
        positions = np.arange(len(parameter_names))
        axes[0, 2].axvline(0.0, color=GREY, lw=0.9, label="Synthetic truth")
        axes[0, 2].errorbar(
            relative_estimate,
            positions,
            xerr=np.vstack((relative_lower, relative_upper)),
            fmt="o",
            color=GREEN,
            ecolor=BLUE,
            capsize=3,
            label="Estimate and 95% CI",
        )
        axes[0, 2].set_yticks(positions, parameter_names)
        axes[0, 2].set(
            xlabel="Error relative to truth [%]",
            title="Physical parameter estimates",
        )
        axes[0, 2].legend(fontsize=8)

        validation = result.time_s >= result.validation_start_time_s
        axes[1, 0].plot(
            result.time_s[validation],
            np.rad2deg(result.smoothed_pitch_rad[validation]),
            color=NAVY,
            label="Measured/cleaned",
        )
        axes[1, 0].plot(
            result.time_s[validation],
            np.rad2deg(result.validation_pitch_prediction_rad[validation]),
            color=ORANGE,
            ls="--",
            label="Identified model",
        )
        axes[1, 0].set(
            xlabel="Time [s]",
            ylabel="Pitch angle [deg]",
            title=f"Held-out validation: RMS {result.validation_pitch_rms_deg:.3f} deg",
        )
        axes[1, 0].legend()

        axes[1, 1].plot(
            result.identification_time_s,
            result.identification_residual_radps2,
            color=BLUE,
            alpha=0.72,
        )
        axes[1, 1].axhline(0.0, color=GREY, lw=0.8)
        axes[1, 1].set(
            xlabel="Identification time [s]",
            ylabel="Rate-derivative residual [rad/s^2]",
            title=f"Robust fit residuals: R² = {result.identification_r_squared:.4f}",
        )

        autocorrelation = result.residual_diagnostics.autocorrelation
        lags = np.arange(1, autocorrelation.size)
        axes[1, 2].stem(
            lags,
            autocorrelation[1:],
            linefmt=BLUE,
            markerfmt="o",
            basefmt=GREY,
        )
        approximate_limit = 1.96 / np.sqrt(
            max(
                result.identification_residual_radps2.size
                / configuration.analysis.derivative_window,
                1,
            )
        )
        axes[1, 2].axhspan(
            -approximate_limit,
            approximate_limit,
            color=BLUE,
            alpha=0.12,
            label="Approx. 95% white-noise band",
        )
        axes[1, 2].set(
            xlabel="Non-overlapping-window lag [-]",
            ylabel="Residual autocorrelation [-]",
            title=f"Residual whiteness: p = {result.residual_diagnostics.ljung_box_p_value:.3f}",
        )
        axes[1, 2].legend(fontsize=8)
        figure.suptitle(
            "Synthetic flight-data alignment, robust identification, and validation",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "flight_data_identification.png"
        figure.savefig(path, dpi=220, pad_inches=0.14)
        plt.close(figure)
    return path
