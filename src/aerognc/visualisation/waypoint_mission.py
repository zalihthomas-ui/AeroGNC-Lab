"""Post-run visualisation for waypoint fixed-wing missions.

Renders an engineering dashboard from a
:class:`~aerognc.simulation.waypoint_mission.WaypointMissionResult`: a **3D**
planned-vs-actual trajectory (East / North / altitude), altitude and airspeed vs
time with cross-track error, and the four actuator channels. Uses the project's
shared ``engineering_style`` and Matplotlib's non-interactive Agg path so it works
headless (CI, servers).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

# matplotlib lazily registers the "3d" projection on first use, so no explicit
# mpl_toolkits.mplot3d import is required for add_subplot(projection="3d").
from aerognc.simulation.waypoint_mission import WaypointMissionResult
from aerognc.visualisation.style import (
    BLUE,
    GREEN,
    NAVY,
    ORANGE,
    RED,
    engineering_style,
)


def plot_waypoint_mission(result: WaypointMissionResult, output_path: str | Path) -> Path:
    """Render the mission dashboard (with a 3D trajectory) to a PNG."""
    if not result.samples:
        raise ValueError("cannot plot a mission with no samples")
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    time_s = np.array([s.time_s for s in result.samples])
    north = np.array([s.north_m for s in result.samples])
    east = np.array([s.east_m for s in result.samples])
    altitude = np.array([s.altitude_m for s in result.samples])
    altitude_cmd = np.array([s.altitude_command_m for s in result.samples])
    airspeed = np.array([s.airspeed_mps for s in result.samples])
    airspeed_cmd = np.array([s.airspeed_command_mps for s in result.samples])
    cross_track = np.array([s.cross_track_error_m for s in result.samples])
    aileron = np.array([s.aileron for s in result.samples])
    elevator = np.array([s.elevator for s in result.samples])
    rudder = np.array([s.rudder for s in result.samples])
    throttle = np.array([s.throttle for s in result.samples])
    planned = np.asarray(result.planned_path_ned_m)
    planned_alt = -planned[:, 2]

    with engineering_style():
        figure = plt.figure(figsize=(13, 8))
        figure.suptitle(
            f"Waypoint mission - {result.outcome} ({result.final_state.value})", fontsize=12
        )
        grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))

        # --- 3D trajectory (spans the left column) ---
        track = figure.add_subplot(grid[:, 0], projection="3d")
        track.plot(planned[:, 1], planned[:, 0], planned_alt, "--o", color=ORANGE, label="planned")
        track.plot(east, north, altitude, color=BLUE, linewidth=1.4, label="actual")
        track.scatter(east[0], north[0], altitude[0], color=GREEN, s=40, label="start")
        track.scatter([0.0], [0.0], [0.0], color=NAVY, s=40, marker="^", label="home")
        track.set_xlabel("East [m]")
        track.set_ylabel("North [m]")
        track.set_zlabel("Altitude [m]")
        track.set_title("3D trajectory")
        track.legend(loc="upper left", fontsize=8)

        # --- altitude + airspeed + cross-track vs time ---
        alt_ax = figure.add_subplot(grid[0, 1])
        alt_ax.plot(time_s, altitude_cmd, "--", color=ORANGE, label="alt cmd")
        alt_ax.plot(time_s, altitude, color=BLUE, label="altitude")
        alt_ax.set_ylabel("Altitude [m]")
        alt_ax.set_title("Altitude / airspeed / cross-track")
        speed_ax = alt_ax.twinx()
        speed_ax.plot(time_s, airspeed, color=GREEN, linewidth=1.0, label="airspeed")
        speed_ax.plot(time_s, airspeed_cmd, ":", color=GREEN, linewidth=0.9)
        speed_ax.plot(time_s, cross_track, color=RED, linewidth=0.9, label="cross-track")
        speed_ax.set_ylabel("m/s  /  cross-track m")
        alt_ax.legend(loc="upper left", fontsize=7)
        speed_ax.legend(loc="lower right", fontsize=7)

        # --- actuators ---
        act_ax = figure.add_subplot(grid[1, 1])
        act_ax.plot(time_s, np.rad2deg(aileron), color=BLUE, label="aileron [deg]")
        act_ax.plot(time_s, np.rad2deg(elevator), color=ORANGE, label="elevator [deg]")
        act_ax.plot(time_s, np.rad2deg(rudder), color=GREEN, label="rudder [deg]")
        act_ax.plot(time_s, 100.0 * throttle, color=RED, label="throttle [%]")
        act_ax.set_xlabel("Time [s]")
        act_ax.set_ylabel("deflection [deg] / throttle [%]")
        act_ax.set_title("Actuators")
        act_ax.legend(loc="best", fontsize=7)

        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(file_path, dpi=110)
        plt.close(figure)
    return file_path
