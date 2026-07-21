"""Interactive 3D mission-control UI for interplanetary trajectory results."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

from aerognc.configuration.interplanetary_loader import SECONDS_PER_DAY
from aerognc.simulation.interplanetary import (
    InterplanetaryMission,
    body_column_prefix,
)

ASTRONOMICAL_UNIT_M = 149_597_870_700.0
MINIMUM_DAYS_PER_SECOND = 0.25
MAXIMUM_DAYS_PER_SECOND = 1_000.0

BACKGROUND = "#07111F"
PANEL = "#0D1B2A"
PANEL_EDGE = "#20384E"
TEXT = "#D9E8F2"
MUTED = "#7F96A8"
GRID = "#28435D"
CYAN = "#39C6E8"
AMBER = "#F2A541"
SUCCESS = "#5FD19A"
ALERT = "#F06464"

MissionCamera = Literal["system", "spacecraft", "assist", "destination", "top", "free"]
MISSION_CAMERAS: tuple[MissionCamera, ...] = (
    "system",
    "spacecraft",
    "assist",
    "destination",
    "top",
    "free",
)


@dataclass(frozen=True, slots=True)
class MissionPlaybackConfiguration:
    """Time and export settings for the interplanetary mission player."""

    frames_per_second: int = 24
    playback_days_per_second: float = 40.0
    repeat: bool = False
    export_dpi: int = 105

    def __post_init__(self) -> None:
        if not 5 <= self.frames_per_second <= 60:
            raise ValueError("frames_per_second must be between 5 and 60")
        if not np.isfinite(self.playback_days_per_second) or not (
            MINIMUM_DAYS_PER_SECOND <= self.playback_days_per_second <= MAXIMUM_DAYS_PER_SECOND
        ):
            raise ValueError(
                "playback_days_per_second must be in "
                f"[{MINIMUM_DAYS_PER_SECOND}, {MAXIMUM_DAYS_PER_SECOND}]"
            )
        if not 60 <= self.export_dpi <= 180:
            raise ValueError("export_dpi must be between 60 and 180")


def mission_phase(mission: InterplanetaryMission, time_days: float) -> str:
    """Return a display phase from the mission event sequence."""
    if not np.isfinite(time_days):
        raise ValueError("time_days must be finite")
    event_days = {event.name: event.time_s / SECONDS_PER_DAY for event in mission.result.events}
    if "assist_orbit_capture" in event_days:
        capture = event_days["assist_orbit_capture"]
        departure = event_days["assist_periapsis_departure"]
        destination = event_days["destination_orbit_capture"]
        if time_days <= min(1.0, mission.result.time_s[-1] / SECONDS_PER_DAY):
            return "DEPARTURE PARKING-ORBIT INJECTION"
        if time_days < capture:
            return "TRANSFER TO ORBIT CAPTURE"
        if time_days < departure:
            assist_name = mission.configuration.body_with_role("assist").name.upper()
            return f"{assist_name} CAPTURED PARKING ORBIT"
        if time_days < destination:
            return "POWERED DEPARTURE / OUTBOUND TRANSFER"
        return "DESTINATION ORBIT CAPTURE"
    if time_days <= min(10.0, mission.result.time_s[-1] / SECONDS_PER_DAY):
        return "DEPARTURE INJECTION"
    assist_entry = event_days.get("assist_entry", np.inf)
    assist_exit = event_days.get("assist_exit", np.inf)
    destination_arrival = event_days.get("destination_arrival", np.inf)
    if "assist_entry" not in event_days:
        if time_days < destination_arrival:
            return "DIRECT INTERPLANETARY TRANSFER"
        if time_days < mission.result.time_s[-1] / SECONDS_PER_DAY:
            return "DESTINATION ENCOUNTER"
        return "MISSION COMPLETE"
    if time_days < assist_entry:
        return "CRUISE TO GRAVITY ASSIST"
    if time_days <= assist_exit:
        assist_name = mission.configuration.body_with_role("assist").name.upper()
        return f"{assist_name} GRAVITY ASSIST"
    if time_days < destination_arrival:
        return "OUTBOUND TRANSFER"
    if time_days < mission.result.time_s[-1] / SECONDS_PER_DAY:
        return "DESTINATION ENCOUNTER"
    return "MISSION COMPLETE"


class InterplanetaryMissionControl:
    """Seekable dark-theme 3D mission player with encounter and energy diagnostics."""

    def __init__(
        self,
        mission: InterplanetaryMission,
        configuration: MissionPlaybackConfiguration | None = None,
        *,
        camera_mode: MissionCamera = "system",
    ) -> None:
        if configuration is None:
            configuration = MissionPlaybackConfiguration()
        if camera_mode not in MISSION_CAMERAS:
            raise ValueError(f"camera_mode must be one of {MISSION_CAMERAS}")
        self.mission = mission
        self.result = mission.result
        self.configuration = configuration
        self.camera_mode: MissionCamera = camera_mode
        self.current_day = 0.0
        self.playback_days_per_second = configuration.playback_days_per_second
        self.is_playing = True
        self._updating_slider = False
        self._interactive_animation: FuncAnimation | None = None
        self._days = self.result.time_s / SECONDS_PER_DAY
        self._spacecraft_au = (
            np.column_stack([self.result.columns[f"position_{axis}_m"] for axis in "xyz"])
            / ASTRONOMICAL_UNIT_M
        )
        self._event_days = {
            event.name: event.time_s / SECONDS_PER_DAY for event in self.result.events
        }
        self._is_orbit_tour = "assist_orbit_capture" in self._event_days
        self._assist = mission.configuration.body_with_role("assist")
        self._destination = mission.configuration.body_with_role("destination")
        self._departure = mission.configuration.body_with_role("departure")
        maximum_orbit_au = (
            max(body.semi_major_axis_m for body in mission.configuration.bodies)
            / ASTRONOMICAL_UNIT_M
        )
        maximum_spacecraft_au = float(np.max(np.linalg.norm(self._spacecraft_au, axis=1)))
        self._system_limit_au = 1.12 * max(maximum_orbit_au, maximum_spacecraft_au)

        style = {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": PANEL_EDGE,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "text.color": TEXT,
            "legend.frameon": False,
        }
        with mpl.rc_context(style):  # type: ignore[arg-type]
            self.figure = plt.figure(figsize=(14.2, 8.0), facecolor=BACKGROUND)
            grid = self.figure.add_gridspec(
                2,
                4,
                left=0.035,
                right=0.975,
                top=0.89,
                bottom=0.27,
                height_ratios=(2.65, 1.0),
                width_ratios=(1.0, 1.0, 1.0, 0.92),
                hspace=0.28,
                wspace=0.23,
            )
            self.scene_axis = cast(Axes3D, self.figure.add_subplot(grid[0, :3], projection="3d"))
            self.telemetry_axis = self.figure.add_subplot(grid[0, 3])
            self.history_axis = self.figure.add_subplot(grid[1, :])
            self.energy_axis = self.history_axis.twinx()
            self._build_header()
            self._build_scene()
            self._build_telemetry()
            self._build_history()
            self._build_controls()

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._update_artists(update_slider=False)

    @property
    def end_day(self) -> float:
        """Final mission elapsed time in days."""
        return float(self._days[-1])

    def _build_header(self) -> None:
        self.figure.text(
            0.035,
            0.952,
            "AEROGNC // DEEP-SPACE MISSION CONTROL",
            color=CYAN,
            fontsize=9.5,
            fontweight="bold",
            family="monospace",
        )
        self.figure.text(
            0.035,
            0.914,
            (
                f"{self.mission.configuration.spacecraft.name.upper()} — "
                + (
                    "CAPTURE / PARK / DEPART TOUR"
                    if self._is_orbit_tour
                    else "GRAVITY-ASSIST TRANSFER"
                )
            ),
            color=TEXT,
            fontsize=16,
            fontweight="bold",
        )
        self.figure.text(
            0.965,
            0.942,
            (
                "PATCHED CONICS  •  IDEAL BURNS  •  SYNTHETIC SYSTEM"
                if self._is_orbit_tour
                else "RESTRICTED N-BODY  •  SI CORE  •  SYNTHETIC SYSTEM"
            ),
            ha="right",
            color=MUTED,
            fontsize=8.2,
            family="monospace",
        )

    def _orbit_points(self, body_name: str) -> np.ndarray:
        body = next(body for body in self.mission.configuration.bodies if body.name == body_name)
        period_s = body.orbital_period_s(
            self.mission.configuration.primary.gravitational_parameter_m3_s2
        )
        points = np.empty((361, 3), dtype=np.float64)
        for index, current_time_s in enumerate(np.linspace(0.0, period_s, points.shape[0])):
            points[index], _velocity = body.state_at_time(
                float(current_time_s),
                self.mission.configuration.primary.gravitational_parameter_m3_s2,
            )
        return points / ASTRONOMICAL_UNIT_M

    def _build_scene(self) -> None:
        axis = self.scene_axis
        axis.set_facecolor(BACKGROUND)
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor(PANEL)
            pane.set_edgecolor(PANEL_EDGE)
            pane.set_alpha(0.72)
        axis.tick_params(colors=MUTED, labelsize=7.5)
        axis.grid(True, color=GRID, alpha=0.35)
        axis.set(
            xlabel="Ecliptic X [AU]",
            ylabel="Ecliptic Y [AU]",
            zlabel="Ecliptic Z [AU]",
            title="PRIMARY-CENTRED TRAJECTORY / LIVE EPHEMERIDES",
        )
        axis.set_box_aspect((1.0, 1.0, 0.36))

        axis.plot(
            self._spacecraft_au[:, 0],
            self._spacecraft_au[:, 1],
            self._spacecraft_au[:, 2],
            color="#33536A",
            linewidth=0.9,
            linestyle="--",
            alpha=0.75,
        )
        (self.completed_path,) = axis.plot([], [], [], color=CYAN, linewidth=2.0)
        (self.spacecraft_halo,) = axis.plot(
            [], [], [], marker="o", markersize=10, color=CYAN, alpha=0.22, linestyle="None"
        )
        (self.spacecraft_marker,) = axis.plot(
            [],
            [],
            [],
            marker="D",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=CYAN,
            markeredgewidth=1.3,
            linestyle="None",
        )
        (self.velocity_vector,) = axis.plot([], [], [], color=AMBER, linewidth=1.5)

        axis.plot(
            [0.0],
            [0.0],
            [0.0],
            marker="o",
            markersize=18,
            color=self.mission.configuration.primary.color,
            alpha=0.10,
            linestyle="None",
        )
        axis.plot(
            [0.0],
            [0.0],
            [0.0],
            marker="o",
            markersize=7,
            color=self.mission.configuration.primary.color,
            linestyle="None",
        )
        axis.text(0.0, 0.0, 0.0, "  HELIOS", color=AMBER, fontsize=7.5, fontweight="bold")

        self.body_markers: dict[str, object] = {}
        for body_index, body in enumerate(self.mission.configuration.bodies):
            orbit = self._orbit_points(body.name)
            axis.plot(
                orbit[:, 0],
                orbit[:, 1],
                orbit[:, 2],
                color=body.color,
                linewidth=0.7,
                alpha=0.34,
                linestyle=":" if body.role == "background" else "-",
            )
            marker_size = 5.0 if body.role == "departure" else 6.2
            (marker,) = axis.plot(
                [],
                [],
                [],
                marker="o",
                markersize=marker_size,
                color=body.color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                linestyle="None",
            )
            self.body_markers[body.name] = marker
            axis.text2D(
                0.015,
                0.965 - 0.052 * body_index,
                f"●  {body.name.upper():<10} / {body.role.upper()}",
                transform=axis.transAxes,
                color=body.color,
                fontsize=7.1,
                fontweight="bold",
                family="monospace",
            )

        highlighted_events = (
            (
                ("assist_orbit_capture", AMBER),
                ("assist_periapsis_departure", ALERT),
                ("destination_orbit_capture", SUCCESS),
            )
            if self._is_orbit_tour
            else (
                ("assist_closest_approach", AMBER),
                ("destination_arrival", SUCCESS),
            )
        )
        for event_name, color in highlighted_events:
            event = next((item for item in self.result.events if item.name == event_name), None)
            if event is None:
                continue
            location_au = event.state[:3] / ASTRONOMICAL_UNIT_M
            axis.plot(
                [location_au[0]],
                [location_au[1]],
                [location_au[2]],
                marker="o",
                markersize=7,
                markerfacecolor=BACKGROUND,
                markeredgecolor=color,
                markeredgewidth=1.3,
                linestyle="None",
            )

    def _build_telemetry(self) -> None:
        axis = self.telemetry_axis
        axis.set_facecolor(PANEL)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(PANEL_EDGE)
        axis.set_title("MISSION TELEMETRY", loc="left", fontsize=9.0, fontweight="bold")
        self.phase_text = axis.text(
            0.04,
            0.93,
            "",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=BACKGROUND,
            fontsize=8.4,
            fontweight="bold",
            family="monospace",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": CYAN, "edgecolor": "none"},
        )
        self.telemetry_text = axis.text(
            0.04,
            0.79,
            "",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=TEXT,
            fontsize=8.0,
            family="monospace",
            linespacing=1.32,
        )
        self.event_text = axis.text(
            0.04,
            0.39,
            "",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=MUTED,
            fontsize=7.0,
            family="monospace",
            linespacing=1.15,
        )
        if self._is_orbit_tour:
            total_delta_v_mps = float(self.result.maximum_summary["total_ideal_delta_v"]["value"])
            final_mass_kg = float(self.result.maximum_summary["final_mass"]["value"])
            outcome = (
                "ORBIT-TOUR BUDGET\n"
                f"total ideal delta-v  {total_delta_v_mps / 1_000.0:6.2f} km/s\n"
                f"final mass           {final_mass_kg / 1_000.0:6.2f} t"
            )
            outcome_color = SUCCESS
        else:
            flyby_gain = float(
                self.result.maximum_summary["assist_heliocentric_speed_gain"]["value"]
            )
            closest_radius = float(self.result.maximum_summary["assist_closest_approach"]["value"])
            altitude_km = (closest_radius - self._assist.radius_m) / 1_000.0
            outcome = (
                "FLYBY RESULT\n"
                f"delta heliocentric speed  {flyby_gain / 1_000.0:+6.2f} km/s\n"
                f"closest altitude          {altitude_km:7.0f} km"
            )
            outcome_color = SUCCESS if flyby_gain > 0.0 else ALERT
        self.outcome_text = axis.text(
            0.04,
            0.045,
            outcome,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color=outcome_color,
            fontsize=7.3,
            family="monospace",
            fontweight="bold",
        )

    def _build_history(self) -> None:
        speed_kmps = self.result.columns["heliocentric_speed_mps"] / 1_000.0
        energy_mjkg = self.result.columns["central_specific_energy_jkg"] / 1.0e6
        self.history_axis.set_facecolor(PANEL)
        self.history_axis.plot(self._days, speed_kmps, color="#385668", linewidth=1.0)
        self.energy_axis.plot(self._days, energy_mjkg, color="#5D4930", linewidth=1.0)
        (self.speed_history,) = self.history_axis.plot([], [], color=CYAN, label="Speed")
        (self.energy_history,) = self.energy_axis.plot([], [], color=AMBER, label="Energy")
        assist_entry = self._event_days.get(
            "assist_orbit_capture" if self._is_orbit_tour else "assist_entry"
        )
        assist_exit = self._event_days.get(
            "assist_periapsis_departure" if self._is_orbit_tour else "assist_exit"
        )
        if assist_entry is not None and assist_exit is not None:
            self.history_axis.axvspan(assist_entry, assist_exit, color=AMBER, alpha=0.13)
        destination_day = self._event_days.get(
            "destination_orbit_capture" if self._is_orbit_tour else "destination_arrival"
        )
        if destination_day is not None:
            self.history_axis.axvline(destination_day, color=SUCCESS, alpha=0.55, linewidth=0.9)
        self.time_cursor = self.history_axis.axvline(0.0, color="white", linewidth=0.9)
        self.history_axis.set(
            xlim=(0.0, self.end_day),
            xlabel="Mission elapsed time [days]",
            ylabel="Heliocentric speed [km/s]",
            title=(
                "SPEED / ENERGY HISTORY — CAPTURE AND POWERED DEPARTURE"
                if self._is_orbit_tour
                else "ENERGY EXCHANGE / GRAVITY-ASSIST SIGNATURE"
            ),
        )
        self.energy_axis.set_ylabel("Primary-relative specific energy [MJ/kg]", color=AMBER)
        self.energy_axis.tick_params(axis="y", colors=AMBER)
        self.history_axis.grid(True, color=GRID, alpha=0.40)
        self.history_axis.legend(
            [self.speed_history, self.energy_history],
            ["Heliocentric speed", "Specific energy"],
            loc="upper right",
            labelcolor=TEXT,
        )

    def _button(self, bounds: tuple[float, float, float, float], label: str) -> Button:
        axis = self.figure.add_axes(bounds, facecolor=PANEL)
        button = Button(axis, label, color=PANEL, hovercolor="#18324A")
        button.label.set_color(TEXT)
        button.label.set_fontsize(8.2)
        for spine in axis.spines.values():
            spine.set_color(PANEL_EDGE)
        return button

    def _build_controls(self) -> None:
        slider_axis = self.figure.add_axes((0.155, 0.178, 0.65, 0.024), facecolor=PANEL)
        self.time_slider = Slider(
            slider_axis,
            "MET [days]",
            0.0,
            self.end_day,
            valinit=0.0,
            color=CYAN,
            track_color="#253C50",
        )
        self.time_slider.label.set_color(MUTED)
        self.time_slider.valtext.set_color(TEXT)
        self.time_slider.on_changed(self._on_slider_changed)

        self.restart_button = self._button((0.19, 0.09, 0.085, 0.045), "RESTART")
        self.slower_button = self._button((0.29, 0.09, 0.065, 0.045), "½ SPEED")
        self.pause_button = self._button((0.37, 0.09, 0.085, 0.045), "PAUSE")
        self.faster_button = self._button((0.47, 0.09, 0.065, 0.045), "2X SPEED")
        self.event_button = self._button((0.55, 0.09, 0.10, 0.045), "NEXT EVENT")
        self.camera_button = self._button((0.665, 0.09, 0.145, 0.045), "")
        self.restart_button.on_clicked(self._restart_from_button)
        self.slower_button.on_clicked(self._slower_from_button)
        self.pause_button.on_clicked(self._toggle_from_button)
        self.faster_button.on_clicked(self._faster_from_button)
        self.event_button.on_clicked(self._event_from_button)
        self.camera_button.on_clicked(self._camera_from_button)
        self.speed_text = self.figure.text(
            0.825,
            0.112,
            "",
            ha="left",
            va="center",
            color=CYAN,
            fontsize=9.0,
            family="monospace",
            fontweight="bold",
        )
        self.figure.text(
            0.5,
            0.032,
            "SPACE play/pause   R restart   ←/→ seek 10 d   ↑/↓ speed   "
            "N next event   C camera   FREE: drag mouse",
            ha="center",
            color=MUTED,
            fontsize=7.8,
            family="monospace",
        )

    def _sample(self, column: str) -> float:
        return float(np.interp(self.current_day, self._days, self.result.columns[column]))

    def _current_spacecraft_position(self) -> np.ndarray:
        return np.array(
            [
                np.interp(self.current_day, self._days, self._spacecraft_au[:, index])
                for index in range(3)
            ]
        )

    def _current_body_position(self, body_name: str) -> np.ndarray:
        prefix = body_column_prefix(body_name)
        return np.array(
            [self._sample(f"{prefix}_position_{axis}_m") / ASTRONOMICAL_UNIT_M for axis in "xyz"]
        )

    def _set_camera(self, spacecraft_position: np.ndarray) -> None:
        if self.camera_mode == "free":
            return
        axis = self.scene_axis
        if self.camera_mode in {"system", "top"}:
            limit = self._system_limit_au
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
            axis.set_zlim(-0.22 * limit, 0.22 * limit)
            if self.camera_mode == "top":
                axis.view_init(elev=89.0, azim=-90.0)
            else:
                fraction = self.current_day / max(self.end_day, 1.0e-12)
                axis.view_init(elev=27.0, azim=-58.0 + 20.0 * fraction)
            return
        if self.camera_mode == "spacecraft":
            centre = spacecraft_position
            half_span = 0.32
        elif self.camera_mode == "assist":
            centre = self._current_body_position(self._assist.name)
            half_span = max(
                0.035,
                1.6 * self.mission.configuration.assist_encounter_radius_m / ASTRONOMICAL_UNIT_M,
            )
        else:
            centre = self._current_body_position(self._destination.name)
            half_span = max(
                0.045,
                2.0 * self.mission.configuration.destination_arrival_radius_m / ASTRONOMICAL_UNIT_M,
            )
        axis.set_xlim(centre[0] - half_span, centre[0] + half_span)
        axis.set_ylim(centre[1] - half_span, centre[1] + half_span)
        axis.set_zlim(centre[2] - 0.55 * half_span, centre[2] + 0.55 * half_span)
        axis.view_init(elev=24.0, azim=-52.0)

    def _event_status(self, event_name: str, label: str) -> str:
        day = self._event_days.get(event_name)
        if day is None:
            return f"[--] {label:<18} unavailable"
        marker = "OK" if self.current_day >= day else "  "
        return f"[{marker}] {label:<18} {day:7.1f} d"

    def _update_artists(self, *, update_slider: bool) -> tuple[Artist, ...]:
        index = int(np.searchsorted(self._days, self.current_day, side="right"))
        index = min(max(index, 1), self._days.size)
        position = self._current_spacecraft_position()
        self.completed_path.set_data_3d(
            self._spacecraft_au[:index, 0],
            self._spacecraft_au[:index, 1],
            self._spacecraft_au[:index, 2],
        )
        self.spacecraft_halo.set_data_3d([position[0]], [position[1]], [position[2]])
        self.spacecraft_marker.set_data_3d([position[0]], [position[1]], [position[2]])
        velocity = np.array(
            [self._sample(f"velocity_{axis}_mps") for axis in "xyz"], dtype=np.float64
        )
        velocity_direction = velocity / max(float(np.linalg.norm(velocity)), 1.0)
        vector_scale = 0.06 * self._system_limit_au
        velocity_end = position + vector_scale * velocity_direction
        self.velocity_vector.set_data_3d(
            [position[0], velocity_end[0]],
            [position[1], velocity_end[1]],
            [position[2], velocity_end[2]],
        )

        body_artists: list[Artist] = []
        for body in self.mission.configuration.bodies:
            body_position = self._current_body_position(body.name)
            marker = self.body_markers[body.name]
            marker.set_data_3d(  # type: ignore[attr-defined]
                [body_position[0]], [body_position[1]], [body_position[2]]
            )
            body_artists.append(cast(Artist, marker))

        speed_kmps = self.result.columns["heliocentric_speed_mps"] / 1_000.0
        energy_mjkg = self.result.columns["central_specific_energy_jkg"] / 1.0e6
        self.speed_history.set_data(self._days[:index], speed_kmps[:index])
        self.energy_history.set_data(self._days[:index], energy_mjkg[:index])
        self.time_cursor.set_xdata([self.current_day, self.current_day])

        phase = mission_phase(self.mission, self.current_day)
        phase_color = AMBER if "ASSIST" in phase else (SUCCESS if "DESTINATION" in phase else CYAN)
        self.phase_text.set_text(phase)
        phase_box = self.phase_text.get_bbox_patch()
        if phase_box is not None:
            phase_box.set_facecolor(phase_color)
        assist_prefix = body_column_prefix(self._assist.name)
        destination_prefix = body_column_prefix(self._destination.name)
        heliocentric_distance_au = self._sample("heliocentric_distance_m") / ASTRONOMICAL_UNIT_M
        heliocentric_speed_kmps = self._sample("heliocentric_speed_mps") / 1_000.0
        specific_energy_mjkg = self._sample("central_specific_energy_jkg") / 1.0e6
        assist_distance_gm = self._sample(f"distance_to_{assist_prefix}_m") / 1.0e9
        assist_relative_speed_kmps = (
            self._sample(f"relative_speed_to_{assist_prefix}_mps") / 1_000.0
        )
        destination_distance_gm = self._sample(f"distance_to_{destination_prefix}_m") / 1.0e9
        self.telemetry_text.set_text(
            "\n".join(
                (
                    f"MET             {self.current_day:8.1f} days",
                    f"HELIO DIST      {heliocentric_distance_au:8.3f} AU",
                    f"HELIO SPEED     {heliocentric_speed_kmps:8.3f} km/s",
                    f"SPEC ENERGY     {specific_energy_mjkg:8.2f} MJ/kg",
                    "",
                    f"TO {self._assist.name.upper():<11}{assist_distance_gm:8.2f} Gm",
                    f"REL SPEED       {assist_relative_speed_kmps:8.3f} km/s",
                    f"TO {self._destination.name.upper():<11}{destination_distance_gm:8.2f} Gm",
                )
            )
        )
        event_rows = (
            (
                self._event_status("departure_periapsis_injection", "Injection"),
                self._event_status("assist_orbit_capture", "Orbit capture"),
                self._event_status("assist_orbit_alignment", "Plane alignment"),
                self._event_status("assist_periapsis_departure", "Powered departure"),
                self._event_status("destination_orbit_capture", "Destination capture"),
            )
            if self._is_orbit_tour
            else (
                self._event_status("departure_injection", "Injection"),
                self._event_status("assist_entry", "Assist boundary in"),
                self._event_status("assist_closest_approach", "Closest approach"),
                self._event_status("assist_exit", "Assist boundary out"),
                self._event_status("destination_arrival", "Destination arrival"),
            )
        )
        self.event_text.set_text("MISSION EVENTS\n" + "\n".join(event_rows))
        self.pause_button.label.set_text("PAUSE" if self.is_playing else "PLAY")
        self.camera_button.label.set_text(f"CAMERA: {self.camera_mode.upper()}")
        self.speed_text.set_text(f"{self.playback_days_per_second:g} days/s")
        self._set_camera(position)

        if update_slider:
            self._updating_slider = True
            self.time_slider.set_val(self.current_day)
            self._updating_slider = False
        return (
            self.completed_path,
            self.spacecraft_halo,
            self.spacecraft_marker,
            self.velocity_vector,
            self.speed_history,
            self.energy_history,
            self.time_cursor,
            self.phase_text,
            self.telemetry_text,
            self.event_text,
            self.speed_text,
            *body_artists,
        )

    def seek(self, time_days: float) -> None:
        """Seek to a clamped mission elapsed day."""
        if not np.isfinite(time_days):
            raise ValueError("time_days must be finite")
        self.current_day = float(np.clip(time_days, 0.0, self.end_day))
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def set_speed(self, days_per_second: float) -> None:
        """Set bounded mission playback speed in days per real second."""
        if not np.isfinite(days_per_second):
            raise ValueError("days_per_second must be finite")
        self.playback_days_per_second = float(
            np.clip(days_per_second, MINIMUM_DAYS_PER_SECOND, MAXIMUM_DAYS_PER_SECOND)
        )
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def set_camera_mode(self, mode: MissionCamera) -> None:
        """Select a system, tracking, encounter, top, or free camera."""
        if mode not in MISSION_CAMERAS:
            raise ValueError(f"camera mode must be one of {MISSION_CAMERAS}")
        self.camera_mode = mode
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def cycle_camera(self) -> None:
        """Advance to the next mission camera."""
        index = MISSION_CAMERAS.index(self.camera_mode)
        self.set_camera_mode(MISSION_CAMERAS[(index + 1) % len(MISSION_CAMERAS)])

    def seek_next_event(self) -> None:
        """Jump to the next event, wrapping to mission start after the final event."""
        future = sorted(day for day in self._event_days.values() if day > self.current_day + 1.0e-9)
        self.seek(future[0] if future else 0.0)

    def toggle_pause(self) -> None:
        """Pause/resume, restarting when a completed mission is resumed."""
        if self.current_day >= self.end_day and not self.is_playing:
            self.current_day = 0.0
        self.is_playing = not self.is_playing
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def restart(self) -> None:
        """Return to departure injection and resume."""
        self.current_day = 0.0
        self.is_playing = True
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def advance_one_frame(self) -> tuple[Artist, ...]:
        """Advance one deterministic display frame."""
        if not self.is_playing:
            return self._update_artists(update_slider=False)
        next_day = self.current_day + (
            self.playback_days_per_second / self.configuration.frames_per_second
        )
        if next_day >= self.end_day:
            if self.configuration.repeat:
                self.current_day = 0.0
            else:
                self.current_day = self.end_day
                self.is_playing = False
        else:
            self.current_day = next_day
        return self._update_artists(update_slider=True)

    def _animation_tick(self, _frame: int) -> tuple[Artist, ...]:
        return self.advance_one_frame()

    def _on_slider_changed(self, value: float) -> None:
        if self._updating_slider:
            return
        self.current_day = float(np.clip(value, 0.0, self.end_day))
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == " ":
            self.toggle_pause()
        elif key in {"r", "home"}:
            self.restart()
        elif key == "left":
            self.seek(self.current_day - 10.0)
        elif key == "right":
            self.seek(self.current_day + 10.0)
        elif key in {"up", "+", "="}:
            self.set_speed(self.playback_days_per_second * 2.0)
        elif key in {"down", "-", "_"}:
            self.set_speed(self.playback_days_per_second * 0.5)
        elif key == "c":
            self.cycle_camera()
        elif key == "n":
            self.seek_next_event()

    def _on_close(self, _event: object) -> None:
        if (
            self._interactive_animation is not None
            and self._interactive_animation.event_source is not None
        ):
            self._interactive_animation.event_source.stop()  # type: ignore[no-untyped-call]

    def _restart_from_button(self, _event: object) -> None:
        self.restart()

    def _slower_from_button(self, _event: object) -> None:
        self.set_speed(self.playback_days_per_second * 0.5)

    def _toggle_from_button(self, _event: object) -> None:
        self.toggle_pause()

    def _faster_from_button(self, _event: object) -> None:
        self.set_speed(self.playback_days_per_second * 2.0)

    def _event_from_button(self, _event: object) -> None:
        self.seek_next_event()

    def _camera_from_button(self, _event: object) -> None:
        self.cycle_camera()

    def save_snapshot(self, path: str | Path, *, time_days: float) -> Path:
        """Save a deterministic high-resolution mission-control frame."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        previous_day = self.current_day
        previous_playing = self.is_playing
        self.is_playing = False
        self.seek(time_days)
        self.figure.savefig(
            output_path,
            dpi=self.configuration.export_dpi,
            facecolor=self.figure.get_facecolor(),
        )
        self.current_day = previous_day
        self.is_playing = previous_playing
        self._update_artists(update_slider=True)
        return output_path

    def save_gif(self, path: str | Path) -> Path:
        """Export a deterministic mission-control GIF."""
        output_path = Path(path)
        if output_path.suffix.lower() != ".gif":
            raise ValueError("mission-control export path must end in .gif")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        previous_day = self.current_day
        previous_playing = self.is_playing
        frame_count = max(
            2,
            int(
                np.ceil(
                    self.end_day
                    * self.configuration.frames_per_second
                    / self.playback_days_per_second
                )
            )
            + 1,
        )
        export_days = np.linspace(0.0, self.end_day, frame_count)

        def render(day: float) -> tuple[Artist, ...]:
            self.current_day = float(day)
            self.is_playing = False
            return self._update_artists(update_slider=True)

        animation = FuncAnimation(
            self.figure,
            render,
            frames=export_days,
            interval=1_000.0 / self.configuration.frames_per_second,
            blit=False,
            cache_frame_data=False,
        )
        animation.save(
            output_path,
            writer=PillowWriter(
                fps=self.configuration.frames_per_second,
                metadata={
                    "title": "AeroGNC-Lab synthetic interplanetary mission control",
                    "artist": "AeroGNC-Lab",
                },
            ),
            dpi=self.configuration.export_dpi,
        )
        self.current_day = previous_day
        self.is_playing = previous_playing
        self._update_artists(update_slider=True)
        return output_path

    def show(self) -> None:
        """Open the interactive mission-control window until it is closed."""
        backend = str(matplotlib.get_backend()).lower()
        if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
            raise RuntimeError(
                f"Matplotlib backend {backend!r} cannot open mission control; "
                "run from a desktop or use --save-gif with --no-window"
            )
        self._interactive_animation = FuncAnimation(
            self.figure,
            self._animation_tick,
            interval=1_000.0 / self.configuration.frames_per_second,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    def close(self) -> None:
        """Close the mission-control figure."""
        plt.close(self.figure)


def play_interplanetary_mission(
    mission: InterplanetaryMission,
    *,
    playback_days_per_second: float = 40.0,
    frames_per_second: int = 24,
    repeat: bool = False,
    camera_mode: MissionCamera = "system",
    save_gif: str | Path | None = None,
    save_snapshot: str | Path | None = None,
    show_window: bool = True,
) -> tuple[Path | None, Path | None]:
    """Export and/or display an interplanetary mission-control player."""
    player = InterplanetaryMissionControl(
        mission,
        MissionPlaybackConfiguration(
            frames_per_second=frames_per_second,
            playback_days_per_second=playback_days_per_second,
            repeat=repeat,
        ),
        camera_mode=camera_mode,
    )
    snapshot_path = (
        player.save_snapshot(
            save_snapshot,
            time_days=mission.configuration.snapshot_time_s / SECONDS_PER_DAY,
        )
        if save_snapshot is not None
        else None
    )
    gif_path = player.save_gif(save_gif) if save_gif is not None else None
    if show_window:
        player.show()
    else:
        player.close()
    return snapshot_path, gif_path
