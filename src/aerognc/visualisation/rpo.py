"""Visualisation for rendezvous / proximity-operations relative trajectories."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.astrodynamics.relative_motion import RelativeTrajectory
from aerognc.visualisation.style import BLUE, GREEN, ORANGE, RED, engineering_style


def plot_rendezvous(trajectory: RelativeTrajectory, output_path: str | Path) -> Path:
    """Render the LVLH relative track and separation-vs-time to a PNG."""
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    states = np.asarray(trajectory.states_lvlh)
    along_track = states[:, 1]
    radial = states[:, 0]
    separation = np.linalg.norm(states[:, :3], axis=1)

    with engineering_style():
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        figure.suptitle(
            f"Rendezvous - closest approach {trajectory.closest_approach_m:.1f} m, "
            f"total dV {trajectory.total_delta_v_mps:.2f} m/s",
            fontsize=11,
        )

        track = axes[0]
        track.plot(along_track, radial, color=BLUE, label="chaser")
        track.plot(0.0, 0.0, "*", color=ORANGE, markersize=14, label="target")
        track.plot(along_track[0], radial[0], "s", color=GREEN, label="start")
        track.set_xlabel("along-track y [m]  (V-bar)")
        track.set_ylabel("radial x [m]  (R-bar)")
        track.set_title("LVLH relative trajectory")
        track.axhline(0.0, color="#B8C4CE", linewidth=0.6)
        track.axvline(0.0, color="#B8C4CE", linewidth=0.6)
        track.legend(loc="best", fontsize=8)
        track.set_aspect("equal", adjustable="datalim")

        sep = axes[1]
        sep.plot(trajectory.time_s / 60.0, separation, color=BLUE)
        sep.axhline(
            trajectory.closest_approach_m,
            color=RED,
            linestyle="--",
            label=f"min {trajectory.closest_approach_m:.1f} m",
        )
        sep.set_xlabel("time [min]")
        sep.set_ylabel("separation [m]")
        sep.set_title("Range to target")
        sep.legend(loc="best", fontsize=8)

        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(file_path, dpi=110)
        plt.close(figure)
    return file_path
