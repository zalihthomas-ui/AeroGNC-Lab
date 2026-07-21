"""Interactive three-dimensional playback of quaternion 6-DOF flight results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

from aerognc.mathematics.quaternion import normalize_quaternion, quaternion_to_dcm
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.logging import SimulationResult
from aerognc.visualisation.playback import (
    MAXIMUM_PLAYBACK_SPEED,
    MINIMUM_PLAYBACK_SPEED,
    PlaybackConfiguration,
)
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, RED, engineering_style

CameraMode = Literal["orbit", "chase", "top", "side", "free"]
CAMERA_MODES: tuple[CameraMode, ...] = ("orbit", "chase", "top", "side", "free")
REQUIRED_SIX_DOF_COLUMNS = {
    "north_m",
    "east_m",
    "altitude_m",
    "total_velocity_mps",
    "quaternion_q0",
    "quaternion_q1",
    "quaternion_q2",
    "quaternion_q3",
    "quaternion_norm",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "roll_rate_degps",
    "pitch_rate_degps",
    "yaw_rate_degps",
    "attitude_error_deg",
    "alpha_deg",
    "beta_deg",
    "mach",
    "dynamic_pressure_pa",
    "mass_kg",
    "thrust_n",
}


def _validate_result(result: SimulationResult) -> None:
    missing = REQUIRED_SIX_DOF_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"3D playback result is missing columns: {sorted(missing)}")
    if "burnout" not in {event.name for event in result.events}:
        raise ValueError("3D playback requires a burnout event")
    if np.max(np.abs(result.columns["quaternion_norm"] - 1.0)) > 1.0e-8:
        raise ValueError("3D playback quaternions are not unit-normalised")


def six_dof_flight_phase(result: SimulationResult, time_s: float) -> str:
    """Return the powered/coast/completed phase of the configured 6-DOF ascent."""
    if not np.isfinite(time_s):
        raise ValueError("time_s must be finite")
    burnout_time_s = next(
        (event.time_s for event in result.events if event.name == "burnout"), None
    )
    if burnout_time_s is None:
        raise ValueError("3D playback requires a burnout event")
    if time_s < burnout_time_s:
        return "POWERED ASCENT"
    if time_s < result.time_s[-1]:
        return "COAST ASCENT"
    return "SCENARIO COMPLETE"


def _ned_to_scene(vector_ned: FloatArray) -> FloatArray:
    """Map NED components to plotted East/North/Altitude coordinates."""
    return np.array([vector_ned[1], vector_ned[0], -vector_ned[2]], dtype=np.float64)


class SixDofPlayback3D:
    """Seekable 3D flight player with a quaternion-oriented vehicle attitude glyph."""

    def __init__(
        self,
        result: SimulationResult,
        configuration: PlaybackConfiguration | None = None,
        *,
        camera_mode: CameraMode = "orbit",
    ) -> None:
        _validate_result(result)
        if configuration is None:
            configuration = PlaybackConfiguration()
        if camera_mode not in CAMERA_MODES:
            raise ValueError(f"camera_mode must be one of {CAMERA_MODES}")
        self.result = result
        self.configuration = configuration
        self.current_time_s = float(result.time_s[0])
        self.playback_speed = configuration.initial_speed
        self.is_playing = True
        self.camera_mode: CameraMode = camera_mode
        self._updating_slider = False
        self._interactive_animation: FuncAnimation | None = None
        self._east_m = result.columns["east_m"]
        self._north_m = result.columns["north_m"]
        self._altitude_m = result.columns["altitude_m"]
        self._burnout_time_s = next(
            event.time_s for event in result.events if event.name == "burnout"
        )
        self._full_limits = self._calculate_full_limits()
        altitude_span_m = max(float(np.ptp(self._altitude_m)), 100.0)
        self._vehicle_length_m = max(35.0, 0.045 * altitude_span_m)

        with engineering_style():
            self.figure = plt.figure(figsize=(12.4, 7.4))
            grid = self.figure.add_gridspec(
                2,
                4,
                left=0.05,
                right=0.96,
                top=0.90,
                bottom=0.23,
                height_ratios=(2.35, 1.0),
                hspace=0.34,
                wspace=0.34,
            )
            self.scene_axis = cast(Axes3D, self.figure.add_subplot(grid[0, :3], projection="3d"))
            self.telemetry_axis = self.figure.add_subplot(grid[0, 3])
            self.history_axis = self.figure.add_subplot(grid[1, :])
            self.error_axis = self.history_axis.twinx()
            self._build_static_scene()
            self._build_controls()

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._update_artists(update_slider=False)

    @property
    def end_time_s(self) -> float:
        """Final logged 6-DOF scenario time."""
        return float(self.result.time_s[-1])

    def _calculate_full_limits(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        east_min = float(np.min(self._east_m))
        east_max = float(np.max(self._east_m))
        north_min = float(np.min(self._north_m))
        north_max = float(np.max(self._north_m))
        horizontal_span = max(east_max - east_min, north_max - north_min, 40.0)
        margin = max(25.0, 0.35 * horizontal_span)
        altitude_max = max(float(np.max(self._altitude_m)), 100.0)
        return (
            (east_min - margin, east_max + margin),
            (north_min - margin, north_max + margin),
            (0.0, altitude_max * 1.08),
        )

    def _build_static_scene(self) -> None:
        self.figure.suptitle(
            "Asteria-SR1 quaternion 6-DOF simulation playback",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        x_limits, y_limits, _z_limits = self._full_limits
        ground_x, ground_y = np.meshgrid(np.linspace(*x_limits, 7), np.linspace(*y_limits, 7))
        self.scene_axis.plot_surface(
            ground_x,
            ground_y,
            np.zeros_like(ground_x),
            color="#D8D2C4",
            alpha=0.28,
            linewidth=0.3,
            edgecolor="#B8B0A2",
            shade=False,
        )
        self.scene_axis.plot(
            self._east_m,
            self._north_m,
            self._altitude_m,
            color="#BFC8D0",
            linewidth=1.1,
            linestyle="--",
        )
        self.scene_axis.plot(
            self._east_m,
            self._north_m,
            np.zeros_like(self._altitude_m),
            color="#8D969E",
            linewidth=0.8,
            linestyle=":",
        )
        burnout_east = float(np.interp(self._burnout_time_s, self.result.time_s, self._east_m))
        burnout_north = float(np.interp(self._burnout_time_s, self.result.time_s, self._north_m))
        burnout_altitude = float(
            np.interp(self._burnout_time_s, self.result.time_s, self._altitude_m)
        )
        self.scene_axis.scatter(
            [burnout_east],
            [burnout_north],
            [burnout_altitude],
            s=34,
            facecolors="white",
            edgecolors=ORANGE,
            linewidths=1.1,
        )
        self.scene_axis.text(
            burnout_east,
            burnout_north,
            burnout_altitude,
            "  Burnout",
            fontsize=8,
            color=GREY,
        )
        self.scene_axis.scatter(
            [self._east_m[0]],
            [self._north_m[0]],
            [0.0],
            marker="s",
            s=32,
            color=GREY,
        )
        (self.trail_line,) = self.scene_axis.plot([], [], [], color=NAVY, linewidth=2.5)
        (self.ground_track,) = self.scene_axis.plot(
            [], [], [], color=BLUE, linewidth=1.2, alpha=0.85
        )
        (self.altitude_line,) = self.scene_axis.plot(
            [], [], [], color=GREY, linewidth=0.8, linestyle="--"
        )
        (self.vehicle_body,) = self.scene_axis.plot(
            [], [], [], color=ORANGE, linewidth=5.2, solid_capstyle="round"
        )
        (self.vehicle_fin_right,) = self.scene_axis.plot([], [], [], color=NAVY, linewidth=2.4)
        (self.vehicle_fin_down,) = self.scene_axis.plot([], [], [], color=NAVY, linewidth=2.4)
        (self.vehicle_nose,) = self.scene_axis.plot(
            [], [], [], marker="o", markersize=5.2, color=RED, linestyle="None"
        )
        (self.body_x_axis,) = self.scene_axis.plot([], [], [], color=RED, linewidth=1.2)
        (self.body_y_axis,) = self.scene_axis.plot([], [], [], color=GREEN, linewidth=1.2)
        (self.body_z_axis,) = self.scene_axis.plot([], [], [], color=BLUE, linewidth=1.2)
        self.scene_axis.set(
            xlabel="East [m]",
            ylabel="North [m]",
            zlabel="Altitude [m]",
            title="3D trajectory and body attitude (vehicle glyph not to scale)",
        )
        self.scene_axis.set_box_aspect((1.0, 1.0, 1.75))
        self.scene_axis.grid(True, alpha=0.35)

        self.telemetry_axis.axis("off")
        self.phase_text = self.telemetry_axis.text(
            0.5,
            0.97,
            "",
            transform=self.telemetry_axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": BLUE, "edgecolor": "none"},
        )
        self.telemetry_text = self.telemetry_axis.text(
            0.01,
            0.82,
            "",
            transform=self.telemetry_axis.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=8.2,
            linespacing=1.23,
            color=NAVY,
        )
        self.event_text = self.telemetry_axis.text(
            0.01,
            0.12,
            "",
            transform=self.telemetry_axis.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=8.0,
            color=GREY,
        )

        self.history_axis.plot(
            self.result.time_s,
            self._altitude_m,
            color="#C5CBD1",
            linewidth=1.0,
        )
        self.error_axis.plot(
            self.result.time_s,
            self.result.columns["attitude_error_deg"],
            color="#EFC1C1",
            linewidth=1.0,
        )
        (self.altitude_history,) = self.history_axis.plot([], [], color=BLUE, label="Altitude")
        (self.error_history,) = self.error_axis.plot([], [], color=RED, label="Attitude error")
        self.time_cursor = self.history_axis.axvline(0.0, color=ORANGE, linewidth=1.0)
        self.history_axis.set(
            xlabel="Simulation time [s]",
            ylabel="Altitude [m]",
            title="Ascent and quaternion-attitude tracking history",
            xlim=(0.0, self.end_time_s),
        )
        self.error_axis.set_ylabel("Attitude error [deg]", color=RED)
        self.error_axis.tick_params(axis="y", colors=RED)
        self.history_axis.legend(
            [self.altitude_history, self.error_history],
            ["Altitude", "Attitude error"],
            loc="upper left",
        )

    def _build_controls(self) -> None:
        slider_axis = self.figure.add_axes((0.17, 0.145, 0.66, 0.026))
        self.time_slider = Slider(
            slider_axis,
            "Simulation time [s]",
            0.0,
            self.end_time_s,
            valinit=0.0,
            color=BLUE,
        )
        self.time_slider.on_changed(self._on_slider_changed)

        restart_axis = self.figure.add_axes((0.24, 0.065, 0.085, 0.045))
        slower_axis = self.figure.add_axes((0.34, 0.065, 0.065, 0.045))
        pause_axis = self.figure.add_axes((0.42, 0.065, 0.09, 0.045))
        faster_axis = self.figure.add_axes((0.53, 0.065, 0.065, 0.045))
        camera_axis = self.figure.add_axes((0.61, 0.065, 0.15, 0.045))
        self.restart_button = Button(restart_axis, "Restart", color="#EEF1F4")
        self.slower_button = Button(slower_axis, "0.5x", color="#EEF1F4")
        self.pause_button = Button(pause_axis, "Pause", color="#EEF1F4")
        self.faster_button = Button(faster_axis, "2x", color="#EEF1F4")
        self.camera_button = Button(camera_axis, "", color="#EEF1F4")
        self.restart_button.on_clicked(self._restart_from_button)
        self.slower_button.on_clicked(self._slower_from_button)
        self.pause_button.on_clicked(self._toggle_from_button)
        self.faster_button.on_clicked(self._faster_from_button)
        self.camera_button.on_clicked(self._camera_from_button)
        self.speed_text = self.figure.text(
            0.78,
            0.087,
            "",
            ha="left",
            va="center",
            color=NAVY,
            fontweight="bold",
        )
        self.figure.text(
            0.5,
            0.025,
            "Space: play/pause   R: restart   Left/Right: seek   Up/Down: speed   "
            "C: camera   Free camera: drag mouse",
            ha="center",
            va="center",
            fontsize=8.0,
            color=GREY,
        )

    def _sample(self, column: str) -> float:
        return float(
            np.interp(self.current_time_s, self.result.time_s, self.result.columns[column])
        )

    def _quaternion_at_current_time(self) -> FloatArray:
        quaternion = np.array(
            [
                self._sample("quaternion_q0"),
                self._sample("quaternion_q1"),
                self._sample("quaternion_q2"),
                self._sample("quaternion_q3"),
            ],
            dtype=np.float64,
        )
        return normalize_quaternion(quaternion)

    def _set_full_limits(self) -> None:
        x_limits, y_limits, z_limits = self._full_limits
        self.scene_axis.set_xlim(*x_limits)
        self.scene_axis.set_ylim(*y_limits)
        self.scene_axis.set_zlim(*z_limits)

    def _update_camera(self, centre: FloatArray) -> None:
        if self.camera_mode == "free":
            return
        if self.camera_mode == "chase":
            horizontal_half_span = 55.0
            vertical_half_span = 180.0
            self.scene_axis.set_xlim(
                centre[0] - horizontal_half_span, centre[0] + horizontal_half_span
            )
            self.scene_axis.set_ylim(
                centre[1] - horizontal_half_span, centre[1] + horizontal_half_span
            )
            self.scene_axis.set_zlim(
                max(0.0, centre[2] - vertical_half_span),
                max(2.0 * vertical_half_span, centre[2] + vertical_half_span),
            )
            self.scene_axis.view_init(elev=18.0, azim=-52.0)
            return
        self._set_full_limits()
        if self.camera_mode == "orbit":
            orbit_fraction = self.current_time_s / max(self.end_time_s, 1.0e-12)
            self.scene_axis.view_init(elev=23.0, azim=-58.0 + 32.0 * orbit_fraction)
        elif self.camera_mode == "top":
            self.scene_axis.view_init(elev=89.0, azim=-90.0)
        elif self.camera_mode == "side":
            self.scene_axis.view_init(elev=8.0, azim=-90.0)

    def _update_artists(self, *, update_slider: bool) -> tuple[Artist, ...]:
        index = int(np.searchsorted(self.result.time_s, self.current_time_s, side="right"))
        index = min(max(index, 1), self.result.time_s.size)
        centre = np.array(
            [
                self._sample("east_m"),
                self._sample("north_m"),
                self._sample("altitude_m"),
            ]
        )
        dcm_nb = quaternion_to_dcm(self._quaternion_at_current_time())
        forward = _ned_to_scene(dcm_nb[:, 0])
        right = _ned_to_scene(dcm_nb[:, 1])
        down = _ned_to_scene(dcm_nb[:, 2])
        tail = centre - 0.40 * self._vehicle_length_m * forward
        nose = centre + 0.60 * self._vehicle_length_m * forward
        fin_centre = centre - 0.32 * self._vehicle_length_m * forward
        fin_half_span = 0.18 * self._vehicle_length_m
        right_fin_start = fin_centre - fin_half_span * right
        right_fin_end = fin_centre + fin_half_span * right
        down_fin_start = fin_centre - fin_half_span * down
        down_fin_end = fin_centre + fin_half_span * down
        axis_length = 0.45 * self._vehicle_length_m

        self.trail_line.set_data_3d(
            self._east_m[:index], self._north_m[:index], self._altitude_m[:index]
        )
        self.ground_track.set_data_3d(self._east_m[:index], self._north_m[:index], np.zeros(index))
        self.altitude_line.set_data_3d(
            [centre[0], centre[0]], [centre[1], centre[1]], [0.0, centre[2]]
        )
        self.vehicle_body.set_data_3d([tail[0], nose[0]], [tail[1], nose[1]], [tail[2], nose[2]])
        self.vehicle_fin_right.set_data_3d(
            [right_fin_start[0], right_fin_end[0]],
            [right_fin_start[1], right_fin_end[1]],
            [right_fin_start[2], right_fin_end[2]],
        )
        self.vehicle_fin_down.set_data_3d(
            [down_fin_start[0], down_fin_end[0]],
            [down_fin_start[1], down_fin_end[1]],
            [down_fin_start[2], down_fin_end[2]],
        )
        self.vehicle_nose.set_data_3d([nose[0]], [nose[1]], [nose[2]])
        for axis_line, direction in (
            (self.body_x_axis, forward),
            (self.body_y_axis, right),
            (self.body_z_axis, down),
        ):
            endpoint = centre + axis_length * direction
            axis_line.set_data_3d(
                [centre[0], endpoint[0]],
                [centre[1], endpoint[1]],
                [centre[2], endpoint[2]],
            )

        self.altitude_history.set_data(self.result.time_s[:index], self._altitude_m[:index])
        self.error_history.set_data(
            self.result.time_s[:index], self.result.columns["attitude_error_deg"][:index]
        )
        self.time_cursor.set_xdata([self.current_time_s, self.current_time_s])
        phase = six_dof_flight_phase(self.result, self.current_time_s)
        phase_color = {
            "POWERED ASCENT": ORANGE,
            "COAST ASCENT": BLUE,
            "SCENARIO COMPLETE": GREY,
        }[phase]
        self.phase_text.set_text(phase)
        phase_box = self.phase_text.get_bbox_patch()
        if phase_box is not None:
            phase_box.set_facecolor(phase_color)
        self.telemetry_text.set_text(
            "\n".join(
                (
                    f"TIME       {self.current_time_s:7.2f} s",
                    f"ALTITUDE   {self._sample('altitude_m'):7.1f} m",
                    f"NORTH/EAST {self._sample('north_m'):5.1f} / {self._sample('east_m'):5.1f} m",
                    f"SPEED/MACH {self._sample('total_velocity_mps'):5.1f} m/s / "
                    f"{self._sample('mach'):.3f}",
                    f"DYN PRESS  {self._sample('dynamic_pressure_pa') / 1_000.0:7.2f} kPa",
                    f"MASS/THR   {self._sample('mass_kg'):5.2f} kg / "
                    f"{self._sample('thrust_n') / 1_000.0:.2f} kN",
                    "",
                    f"ROLL/PITCH {self._sample('roll_deg'):5.2f} / "
                    f"{self._sample('pitch_deg'):5.2f} deg",
                    f"YAW        {self._sample('yaw_deg'):7.2f} deg",
                    f"p / q / r  {self._sample('roll_rate_degps'):5.1f} / "
                    f"{self._sample('pitch_rate_degps'):5.1f} / "
                    f"{self._sample('yaw_rate_degps'):5.1f}",
                    f"ALPHA/BETA {self._sample('alpha_deg'):5.2f} / "
                    f"{self._sample('beta_deg'):5.2f} deg",
                    f"ATT ERROR  {self._sample('attitude_error_deg'):7.2f} deg",
                )
            )
        )
        event_status = "OK" if self.current_time_s >= self._burnout_time_s else "  "
        self.event_text.set_text(
            f"EVENT\n[{event_status}] Burnout  {self._burnout_time_s:5.2f} s\n"
            f"Camera: {self.camera_mode.title()}"
        )
        self.speed_text.set_text(f"{self.playback_speed:g}x")
        self.pause_button.label.set_text("Pause" if self.is_playing else "Play")
        self.camera_button.label.set_text(f"Camera: {self.camera_mode.title()}")
        self._update_camera(centre)

        if update_slider:
            self._updating_slider = True
            self.time_slider.set_val(self.current_time_s)
            self._updating_slider = False
        return (
            self.trail_line,
            self.ground_track,
            self.altitude_line,
            self.vehicle_body,
            self.vehicle_fin_right,
            self.vehicle_fin_down,
            self.vehicle_nose,
            self.body_x_axis,
            self.body_y_axis,
            self.body_z_axis,
            self.altitude_history,
            self.error_history,
            self.time_cursor,
            self.phase_text,
            self.telemetry_text,
            self.event_text,
            self.speed_text,
        )

    def seek(self, time_s: float) -> None:
        """Seek to a clamped logged simulation time."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        self.current_time_s = float(np.clip(time_s, 0.0, self.end_time_s))
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def set_speed(self, speed: float) -> None:
        """Set playback speed using the shared bounded playback configuration."""
        if not np.isfinite(speed):
            raise ValueError("speed must be finite")
        self.playback_speed = float(np.clip(speed, MINIMUM_PLAYBACK_SPEED, MAXIMUM_PLAYBACK_SPEED))
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def set_camera_mode(self, mode: CameraMode) -> None:
        """Select orbit, chase, top, side, or mouse-controlled free camera."""
        if mode not in CAMERA_MODES:
            raise ValueError(f"camera mode must be one of {CAMERA_MODES}")
        self.camera_mode = mode
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def cycle_camera(self) -> None:
        """Advance to the next predefined camera mode."""
        index = CAMERA_MODES.index(self.camera_mode)
        self.set_camera_mode(CAMERA_MODES[(index + 1) % len(CAMERA_MODES)])

    def toggle_pause(self) -> None:
        """Toggle play/pause, restarting a completed scenario when resumed."""
        if self.current_time_s >= self.end_time_s and not self.is_playing:
            self.current_time_s = 0.0
        self.is_playing = not self.is_playing
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def restart(self) -> None:
        """Return to rail exit and resume playback."""
        self.current_time_s = 0.0
        self.is_playing = True
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def advance_one_frame(self) -> tuple[Artist, ...]:
        """Advance one deterministic display frame."""
        if not self.is_playing:
            return self._update_artists(update_slider=False)
        next_time_s = self.current_time_s + (
            self.playback_speed / self.configuration.frames_per_second
        )
        if next_time_s >= self.end_time_s:
            if self.configuration.repeat:
                self.current_time_s = 0.0
            else:
                self.current_time_s = self.end_time_s
                self.is_playing = False
        else:
            self.current_time_s = next_time_s
        return self._update_artists(update_slider=True)

    def _animation_tick(self, _frame: int) -> tuple[Artist, ...]:
        return self.advance_one_frame()

    def _on_slider_changed(self, value: float) -> None:
        if self._updating_slider:
            return
        self.current_time_s = float(np.clip(value, 0.0, self.end_time_s))
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == " ":
            self.toggle_pause()
        elif key in {"r", "home"}:
            self.restart()
        elif key == "left":
            self.seek(self.current_time_s - 0.5)
        elif key == "right":
            self.seek(self.current_time_s + 0.5)
        elif key in {"up", "+", "="}:
            self.set_speed(self.playback_speed * 2.0)
        elif key in {"down", "-", "_"}:
            self.set_speed(self.playback_speed * 0.5)
        elif key == "c":
            self.cycle_camera()

    def _on_close(self, _event: object) -> None:
        if (
            self._interactive_animation is not None
            and self._interactive_animation.event_source is not None
        ):
            self._interactive_animation.event_source.stop()  # type: ignore[no-untyped-call]

    def _restart_from_button(self, _event: object) -> None:
        self.restart()

    def _slower_from_button(self, _event: object) -> None:
        self.set_speed(self.playback_speed * 0.5)

    def _toggle_from_button(self, _event: object) -> None:
        self.toggle_pause()

    def _faster_from_button(self, _event: object) -> None:
        self.set_speed(self.playback_speed * 2.0)

    def _camera_from_button(self, _event: object) -> None:
        self.cycle_camera()

    def save_snapshot(self, path: str | Path, *, time_s: float) -> Path:
        """Save a deterministic 3D dashboard frame."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        previous_time_s = self.current_time_s
        previous_playing = self.is_playing
        self.is_playing = False
        self.seek(time_s)
        self.figure.savefig(output_path, dpi=self.configuration.export_dpi)
        self.current_time_s = previous_time_s
        self.is_playing = previous_playing
        self._update_artists(update_slider=True)
        return output_path

    def save_gif(self, path: str | Path) -> Path:
        """Export the 3D camera animation as a deterministic GIF."""
        output_path = Path(path)
        if output_path.suffix.lower() != ".gif":
            raise ValueError("3D playback export path must end in .gif")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        previous_time_s = self.current_time_s
        previous_playing = self.is_playing
        frame_count = max(
            2,
            int(
                np.ceil(
                    self.end_time_s * self.configuration.frames_per_second / self.playback_speed
                )
            )
            + 1,
        )
        export_times = np.linspace(0.0, self.end_time_s, frame_count)

        def render(time_s: float) -> tuple[Artist, ...]:
            self.current_time_s = float(time_s)
            self.is_playing = False
            return self._update_artists(update_slider=True)

        animation = FuncAnimation(
            self.figure,
            render,
            frames=export_times,
            interval=1_000.0 / self.configuration.frames_per_second,
            blit=False,
            cache_frame_data=False,
        )
        animation.save(
            output_path,
            writer=PillowWriter(
                fps=self.configuration.frames_per_second,
                metadata={
                    "title": "AeroGNC-Lab quaternion 6-DOF 3D playback",
                    "artist": "AeroGNC-Lab",
                },
            ),
            dpi=self.configuration.export_dpi,
        )
        self.current_time_s = previous_time_s
        self.is_playing = previous_playing
        self._update_artists(update_slider=True)
        return output_path

    def show(self) -> None:
        """Open the interactive 3D window until the user closes it."""
        backend = str(matplotlib.get_backend()).lower()
        if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
            raise RuntimeError(
                f"Matplotlib backend {backend!r} cannot open a 3D playback window; "
                "run from a desktop session or use --save-gif with --no-window"
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
        """Close the 3D playback figure."""
        plt.close(self.figure)


def play_six_dof_3d(
    result: SimulationResult,
    *,
    playback_speed: float = 2.0,
    frames_per_second: int = 30,
    repeat: bool = False,
    camera_mode: CameraMode = "orbit",
    save_gif: str | Path | None = None,
    show_window: bool = True,
) -> Path | None:
    """Create an optional GIF and/or open the interactive quaternion 3D player."""
    player = SixDofPlayback3D(
        result,
        PlaybackConfiguration(
            frames_per_second=frames_per_second,
            initial_speed=playback_speed,
            repeat=repeat,
        ),
        camera_mode=camera_mode,
    )
    saved_path = player.save_gif(save_gif) if save_gif is not None else None
    if show_window:
        player.show()
    else:
        player.close()
    return saved_path
