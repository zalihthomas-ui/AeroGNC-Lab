"""Publication-style rotating-navigation and integrity diagnostics."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.simulation.advanced_navigation import (
    AdvancedNavigationResult,
    NavigationConsistencyResult,
)
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style


def plot_advanced_navigation(
    result: AdvancedNavigationResult,
    consistency: NavigationConsistencyResult,
    output_directory: str | Path,
) -> Path:
    """Plot estimation errors, covariance, NIS/NEES, and delayed-fault decisions."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    time_s = result.time_s
    position_error_norm = np.linalg.norm(result.position_error_m, axis=1)
    position_two_sigma = 2.0 * np.linalg.norm(result.position_sigma_m, axis=1)
    velocity_error_norm = np.linalg.norm(result.velocity_error_mps, axis=1)
    velocity_two_sigma = 2.0 * np.linalg.norm(result.velocity_sigma_mps, axis=1)
    attitude_error_norm_deg = np.rad2deg(np.linalg.norm(result.attitude_error_rad, axis=1))
    attitude_two_sigma_deg = 2.0 * np.rad2deg(np.linalg.norm(result.attitude_sigma_rad, axis=1))
    with engineering_style():
        figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), constrained_layout=True)
        axes[0, 0].plot(time_s, position_error_norm, color=NAVY, label="Error norm")
        axes[0, 0].plot(time_s, position_two_sigma, color=BLUE, ls="--", label="2-sigma norm")
        axes[0, 0].set(ylabel="Position error [m]", title="Position consistency")
        axes[0, 0].legend()

        axes[0, 1].plot(time_s, velocity_error_norm, color=ORANGE, label="Error norm")
        axes[0, 1].plot(
            time_s,
            velocity_two_sigma,
            color=BLUE,
            ls="--",
            label="2-sigma norm",
        )
        axes[0, 1].set(ylabel="Velocity error [m/s]", title="Velocity consistency")
        axes[0, 1].legend()

        axes[0, 2].plot(time_s, attitude_error_norm_deg, color=GREEN, label="Error norm")
        axes[0, 2].plot(
            time_s,
            attitude_two_sigma_deg,
            color=BLUE,
            ls="--",
            label="2-sigma norm",
        )
        axes[0, 2].set(ylabel="Attitude error [deg]", title="Attitude consistency")
        axes[0, 2].legend()

        for sensor_name, color, marker in (
            ("gnss", BLUE, "o"),
            ("barometer", ORANGE, "s"),
        ):
            updates = [
                update for update in result.aiding_updates if update.sensor_name == sensor_name
            ]
            accepted = [update for update in updates if update.accepted]
            rejected = [update for update in updates if not update.accepted]
            axes[1, 0].scatter(
                [update.processing_time_s for update in accepted],
                [update.nis for update in accepted],
                color=color,
                marker=marker,
                s=11,
                alpha=0.55,
                label=f"{sensor_name} accepted",
            )
            axes[1, 0].scatter(
                [update.processing_time_s for update in rejected],
                [update.nis for update in rejected],
                color=RED,
                marker="x",
                s=30,
                label=f"{sensor_name} rejected",
            )
            if updates:
                axes[1, 0].axhline(
                    updates[0].threshold,
                    color=color,
                    ls=":" if sensor_name == "gnss" else "--",
                    alpha=0.8,
                )
        axes[1, 0].set(
            xlabel="Processing time [s]",
            ylabel="NIS [-]",
            title="Innovation gating and injected faults",
            yscale="log",
        )
        axes[1, 0].legend(fontsize=7, ncol=2)

        axes[1, 1].plot(
            consistency.time_s,
            consistency.mean_nees_15,
            color=NAVY,
            label=f"Mean NEES ({consistency.run_count} runs)",
        )
        axes[1, 1].fill_between(
            consistency.time_s,
            consistency.nees_lower,
            consistency.nees_upper,
            color=BLUE,
            alpha=0.16,
            label=f"{100.0 * consistency.confidence:.0f}% bounds",
        )
        axes[1, 1].set(
            xlabel="Time [s]",
            ylabel="15-state NEES [-]",
            title="Seeded ensemble consistency",
        )
        axes[1, 1].legend()

        height_true_m = -result.true_position_ned_m[:, 2]
        height_estimated_m = -result.estimated_position_ned_m[:, 2]
        axes[1, 2].plot(time_s, height_true_m, color=NAVY, label="Truth")
        axes[1, 2].plot(time_s, height_estimated_m, color=ORANGE, label="Estimate")
        rejected_times = [
            update.processing_time_s for update in result.aiding_updates if not update.accepted
        ]
        for index, rejected_time_s in enumerate(rejected_times):
            axes[1, 2].axvline(
                rejected_time_s,
                color=RED,
                alpha=0.18,
                lw=0.8,
                label="Rejected aiding" if index == 0 else None,
            )
        axes[1, 2].set(
            xlabel="Time [s]",
            ylabel="Height above launch [m]",
            title="Delayed-aiding reconstruction",
        )
        axes[1, 2].legend()
        for axis in axes[0, :]:
            axis.set_xlabel("Time [s]")
        figure.suptitle(
            "Rotating strapdown INS, delayed ESKF, integrity, and consistency",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "advanced_navigation.png"
        figure.savefig(path, dpi=220, pad_inches=0.16)
        plt.close(figure)
    return path
