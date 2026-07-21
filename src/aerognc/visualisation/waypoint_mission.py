"""Post-run visualisation for waypoint fixed-wing missions.

Renders a compact engineering dashboard from a
:class:`~aerognc.simulation.waypoint_mission.WaypointMissionResult`: the planned
vs actual ground track, altitude and airspeed vs time, lateral cross-track error,
and the four actuator channels. Uses the project's shared ``engineering_style``
and Matplotlib's non-interactive Agg path so it works headless (CI, servers).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

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
    """Render the mission dashboard to a PNG and return the path."""
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

    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        figure.suptitle(
            f"Waypoint mission - {result.outcome} ({result.final_state.value})", fontsize=12
        )

        track = axes[0, 0]
        track.plot(planned[:, 1], planned[:, 0], "--", color=ORANGE, label="planned")
        track.plot(east, north, color=BLUE, label="actual")
        track.plot(planned[:, 1], planned[:, 0], "o", color=NAVY, markersize=4)
        track.plot(east[0], north[0], "s", color=GREEN, label="start")
        track.set_xlabel("East [m]")
        track.set_ylabel("North [m]")
        track.set_title("Ground track")
        track.set_aspect("equal", adjustable="datalim")
        track.legend(loc="best", fontsize=8)

        alt_ax = axes[0, 1]
        alt_ax.plot(time_s, altitude_cmd, "--", color=ORANGE, label="command")
        alt_ax.plot(time_s, altitude, color=BLUE, label="actual")
        alt_ax.set_xlabel("Time [s]")
        alt_ax.set_ylabel("Altitude [m]")
        alt_ax.set_title("Altitude")
        alt_ax.legend(loc="best", fontsize=8)

        spd_ax = axes[1, 0]
        spd_ax.plot(time_s, airspeed_cmd, "--", color=ORANGE, label="airspeed cmd")
        spd_ax.plot(time_s, airspeed, color=BLUE, label="airspeed")
        spd_ax.plot(time_s, cross_track, color=RED, label="cross-track [m]")
        spd_ax.set_xlabel("Time [s]")
        spd_ax.set_ylabel("m/s  and  m")
        spd_ax.set_title("Airspeed and cross-track error")
        spd_ax.legend(loc="best", fontsize=8)

        act_ax = axes[1, 1]
        act_ax.plot(time_s, np.rad2deg(aileron), color=BLUE, label="aileron [deg]")
        act_ax.plot(time_s, np.rad2deg(elevator), color=ORANGE, label="elevator [deg]")
        act_ax.plot(time_s, np.rad2deg(rudder), color=GREEN, label="rudder [deg]")
        act_ax.plot(time_s, 100.0 * throttle, color=RED, label="throttle [%]")
        act_ax.set_xlabel("Time [s]")
        act_ax.set_ylabel("deflection [deg] / throttle [%]")
        act_ax.set_title("Actuators")
        act_ax.legend(loc="best", fontsize=8)

        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(file_path, dpi=110)
        plt.close(figure)
    return file_path
