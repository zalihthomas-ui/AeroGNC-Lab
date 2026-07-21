"""Interactive, seekable playback of a completed point-mass flight simulation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.widgets import Button, Slider

from aerognc.simulation.logging import SimulationResult
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, RED, engineering_style

MINIMUM_PLAYBACK_SPEED = 0.25
MAXIMUM_PLAYBACK_SPEED = 64.0
REQUIRED_COLUMNS = {
    "altitude_m",
    "ground_range_m",
    "vertical_velocity_up_mps",
    "total_velocity_mps",
    "mach",
    "dynamic_pressure_pa",
    "mass_kg",
    "thrust_n",
    "drag_n",
    "flight_path_angle_deg",
}


@dataclass(frozen=True, slots=True)
class PlaybackConfiguration:
    """Interactive rendering settings; speed is simulation seconds per real second."""

    frames_per_second: int = 30
    initial_speed: float = 4.0
    repeat: bool = False
    export_dpi: int = 105

    def __post_init__(self) -> None:
        if not 5 <= self.frames_per_second <= 120:
            raise ValueError("frames_per_second must be between 5 and 120")
        if not np.isfinite(self.initial_speed) or not (
            MINIMUM_PLAYBACK_SPEED <= self.initial_speed <= MAXIMUM_PLAYBACK_SPEED
        ):
            raise ValueError(
                f"initial_speed must be in [{MINIMUM_PLAYBACK_SPEED}, {MAXIMUM_PLAYBACK_SPEED}]"
            )
        if not 60 <= self.export_dpi <= 220:
            raise ValueError("export_dpi must be between 60 and 220")


def _event_times(result: SimulationResult) -> dict[str, float]:
    return {event.name: event.time_s for event in result.events}


def flight_phase(result: SimulationResult, time_s: float) -> str:
    """Return the research-flight phase at an arbitrary playback time."""
    if not np.isfinite(time_s):
        raise ValueError("time_s must be finite")
    event_times = _event_times(result)
    missing = {"burnout", "apogee", "ground_impact"} - set(event_times)
    if missing:
        raise ValueError(f"playback result is missing events: {sorted(missing)}")
    if time_s < event_times["burnout"]:
        return "POWERED ASCENT"
    if time_s < event_times["apogee"]:
        return "COAST ASCENT"
    if time_s < event_times["ground_impact"]:
        return "DESCENT"
    return "COMPLETE"


def _validate_result(result: SimulationResult) -> None:
    missing_columns = REQUIRED_COLUMNS - set(result.columns)
    if missing_columns:
        raise ValueError(f"playback result is missing columns: {sorted(missing_columns)}")
    flight_phase(result, float(result.time_s[0]))


class ThreeDofPlayback:
    """Matplotlib flight player with immutable source data and interactive controls."""

    def __init__(
        self,
        result: SimulationResult,
        configuration: PlaybackConfiguration | None = None,
    ) -> None:
        _validate_result(result)
        if configuration is None:
            configuration = PlaybackConfiguration()
        self.result = result
        self.configuration = configuration
        self.current_time_s = float(result.time_s[0])
        self.playback_speed = configuration.initial_speed
        self.is_playing = True
        self._updating_slider = False
        self._interactive_animation: FuncAnimation | None = None

        self._range_km = result.columns["ground_range_m"] / 1_000.0
        self._altitude_km = result.columns["altitude_m"] / 1_000.0
        self._speed_mps = result.columns["total_velocity_mps"]
        self._event_times = _event_times(result)

        with engineering_style():
            self.figure = plt.figure(figsize=(12.0, 7.2))
            grid = self.figure.add_gridspec(
                2,
                4,
                left=0.07,
                right=0.96,
                top=0.90,
                bottom=0.23,
                height_ratios=(2.25, 1.0),
                hspace=0.34,
                wspace=0.35,
            )
            self.trajectory_axis = self.figure.add_subplot(grid[0, :3])
            self.telemetry_axis = self.figure.add_subplot(grid[0, 3])
            self.history_axis = self.figure.add_subplot(grid[1, :])
            self.speed_axis = self.history_axis.twinx()
            self._build_static_scene()
            self._build_controls()

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._update_artists(update_slider=False)

    @property
    def end_time_s(self) -> float:
        """Final simulation time available for playback."""
        return float(self.result.time_s[-1])

    def _build_static_scene(self) -> None:
        self.figure.suptitle(
            "Asteria-SR1 fictional research-flight playback",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        self.trajectory_axis.plot(
            self._range_km,
            self._altitude_km,
            color="#C5CBD1",
            linewidth=1.0,
            linestyle="--",
        )
        self.trajectory_axis.axhline(0.0, color="#4C5B45", linewidth=1.3)
        self.trajectory_axis.fill_between(
            [float(np.min(self._range_km)) - 0.05, float(np.max(self._range_km)) + 0.05],
            -0.06,
            0.0,
            color="#D8D2C4",
            alpha=0.75,
        )
        self.trajectory_axis.scatter([0.0], [0.0], marker="s", s=32, color=GREY, zorder=4)
        for event in self.result.events:
            horizontal_km = float(np.interp(event.time_s, self.result.time_s, self._range_km))
            altitude_km = float(np.interp(event.time_s, self.result.time_s, self._altitude_km))
            self.trajectory_axis.scatter(
                [horizontal_km],
                [altitude_km],
                s=32,
                facecolors="white",
                edgecolors=ORANGE,
                linewidths=1.1,
                zorder=4,
            )
            self.trajectory_axis.annotate(
                event.name.replace("_", " ").title(),
                (horizontal_km, altitude_km),
                xytext=(-5, 6) if event.name == "ground_impact" else (5, 6),
                textcoords="offset points",
                fontsize=8,
                color=GREY,
                ha="right" if event.name == "ground_impact" else "left",
            )
        (self.trail_line,) = self.trajectory_axis.plot([], [], color=NAVY, linewidth=2.5)
        (self.vehicle_body,) = self.trajectory_axis.plot(
            [], [], color=ORANGE, linewidth=5.0, solid_capstyle="round", zorder=6
        )
        self.vehicle_nose = self.trajectory_axis.scatter(
            [], [], s=24, color=RED, edgecolor="white", linewidth=0.5, zorder=7
        )
        x_span = max(float(np.ptp(self._range_km)), 0.25)
        y_span = max(float(np.ptp(self._altitude_km)), 0.25)
        self._vehicle_length_km = 0.045 * max(x_span, y_span)
        self.trajectory_axis.set(
            xlabel="Ground range [km]",
            ylabel="Altitude [km]",
            title="NED trajectory (dashed: complete logged path)",
            xlim=(
                min(-0.03, float(np.min(self._range_km)) - 0.03),
                float(np.max(self._range_km)) + 0.06,
            ),
            ylim=(-0.06, float(np.max(self._altitude_km)) + 0.10),
        )
        self.trajectory_axis.set_aspect("equal", adjustable="box")

        self.telemetry_axis.axis("off")
        self.phase_text = self.telemetry_axis.text(
            0.5,
            0.96,
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
            0.02,
            0.80,
            "",
            transform=self.telemetry_axis.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=8.4,
            linespacing=1.25,
            color=NAVY,
        )
        self.event_text = self.telemetry_axis.text(
            0.02,
            0.22,
            "",
            transform=self.telemetry_axis.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=7.8,
            linespacing=1.30,
            color=GREY,
        )

        self.history_axis.plot(
            self.result.time_s,
            self.result.columns["altitude_m"],
            color="#C5CBD1",
            linewidth=1.0,
        )
        self.speed_axis.plot(
            self.result.time_s,
            self._speed_mps,
            color="#F1C49A",
            linewidth=1.0,
        )
        (self.altitude_history,) = self.history_axis.plot([], [], color=BLUE, label="Altitude")
        (self.speed_history,) = self.speed_axis.plot([], [], color=ORANGE, label="Speed")
        self.time_cursor = self.history_axis.axvline(0.0, color=RED, linewidth=1.0)
        self.history_axis.set(
            xlabel="Simulation time [s]",
            ylabel="Altitude [m]",
            title="Flight history and playback cursor",
            xlim=(0.0, self.end_time_s),
        )
        self.speed_axis.set_ylabel("Speed [m/s]", color=ORANGE)
        self.speed_axis.tick_params(axis="y", colors=ORANGE)
        lines = [self.altitude_history, self.speed_history]
        self.history_axis.legend(lines, ["Altitude", "Speed"], loc="upper right")

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

        restart_axis = self.figure.add_axes((0.29, 0.065, 0.09, 0.045))
        slower_axis = self.figure.add_axes((0.40, 0.065, 0.07, 0.045))
        pause_axis = self.figure.add_axes((0.49, 0.065, 0.10, 0.045))
        faster_axis = self.figure.add_axes((0.61, 0.065, 0.07, 0.045))
        self.restart_button = Button(restart_axis, "Restart", color="#EEF1F4", hovercolor="#DCE5EC")
        self.slower_button = Button(slower_axis, "0.5x", color="#EEF1F4", hovercolor="#DCE5EC")
        self.pause_button = Button(pause_axis, "Pause", color="#EEF1F4", hovercolor="#DCE5EC")
        self.faster_button = Button(faster_axis, "2x", color="#EEF1F4", hovercolor="#DCE5EC")
        self.restart_button.on_clicked(self._restart_from_button)
        self.slower_button.on_clicked(self._slower_from_button)
        self.pause_button.on_clicked(self._toggle_from_button)
        self.faster_button.on_clicked(self._faster_from_button)
        self.speed_text = self.figure.text(
            0.72,
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
            "Space: play/pause   R: restart   Left/Right: seek 1 s   Up/Down: speed",
            ha="center",
            va="center",
            fontsize=8.2,
            color=GREY,
        )

    def _sample(self, column: str) -> float:
        return float(
            np.interp(self.current_time_s, self.result.time_s, self.result.columns[column])
        )

    def _update_artists(self, *, update_slider: bool) -> tuple[Artist, ...]:
        index = int(np.searchsorted(self.result.time_s, self.current_time_s, side="right"))
        index = min(max(index, 1), self.result.time_s.size)
        current_range_km = float(np.interp(self.current_time_s, self.result.time_s, self._range_km))
        current_altitude_km = float(
            np.interp(self.current_time_s, self.result.time_s, self._altitude_km)
        )
        flight_path_angle_rad = np.deg2rad(self._sample("flight_path_angle_deg"))
        direction = np.array([np.cos(flight_path_angle_rad), np.sin(flight_path_angle_rad)])
        centre = np.array([current_range_km, current_altitude_km])
        tail = centre - 0.45 * self._vehicle_length_km * direction
        nose = centre + 0.55 * self._vehicle_length_km * direction

        self.trail_line.set_data(self._range_km[:index], self._altitude_km[:index])
        self.vehicle_body.set_data([tail[0], nose[0]], [tail[1], nose[1]])
        self.vehicle_nose.set_offsets(nose[None, :])
        self.altitude_history.set_data(
            self.result.time_s[:index], self.result.columns["altitude_m"][:index]
        )
        self.speed_history.set_data(self.result.time_s[:index], self._speed_mps[:index])
        self.time_cursor.set_xdata([self.current_time_s, self.current_time_s])

        phase = flight_phase(self.result, self.current_time_s)
        phase_color = {
            "POWERED ASCENT": ORANGE,
            "COAST ASCENT": BLUE,
            "DESCENT": GREEN,
            "COMPLETE": GREY,
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
                    f"RANGE      {self._sample('ground_range_m'):7.1f} m",
                    f"VERT VEL   {self._sample('vertical_velocity_up_mps'):7.1f} m/s",
                    f"SPEED      {self._sample('total_velocity_mps'):7.1f} m/s",
                    f"MACH       {self._sample('mach'):7.3f}",
                    f"DYN PRESS  {self._sample('dynamic_pressure_pa') / 1_000.0:7.2f} kPa",
                    f"MASS       {self._sample('mass_kg'):7.2f} kg",
                    f"THRUST     {self._sample('thrust_n') / 1_000.0:7.2f} kN",
                    f"PATH ANG   {self._sample('flight_path_angle_deg'):7.2f} deg",
                )
            )
        )
        event_lines = ["EVENTS"]
        for name in ("burnout", "apogee", "ground_impact"):
            status = "OK" if self.current_time_s >= self._event_times[name] else "  "
            label = name.replace("_", " ").title()
            event_lines.append(f"[{status}] {label:<13} {self._event_times[name]:6.2f} s")
        self.event_text.set_text("\n".join(event_lines))
        self.speed_text.set_text(f"Playback: {self.playback_speed:g}x")
        self.pause_button.label.set_text("Pause" if self.is_playing else "Play")

        if update_slider:
            self._updating_slider = True
            self.time_slider.set_val(self.current_time_s)
            self._updating_slider = False
        return (
            self.trail_line,
            self.vehicle_body,
            self.vehicle_nose,
            self.altitude_history,
            self.speed_history,
            self.time_cursor,
            self.phase_text,
            self.telemetry_text,
            self.event_text,
            self.speed_text,
        )

    def seek(self, time_s: float) -> None:
        """Seek to a clamped simulation time without changing play/pause state."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        self.current_time_s = float(np.clip(time_s, 0.0, self.end_time_s))
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def set_speed(self, speed: float) -> None:
        """Set playback speed in simulation-seconds per real-second."""
        if not np.isfinite(speed):
            raise ValueError("speed must be finite")
        self.playback_speed = float(np.clip(speed, MINIMUM_PLAYBACK_SPEED, MAXIMUM_PLAYBACK_SPEED))
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def toggle_pause(self) -> None:
        """Toggle play/pause, restarting from zero if playback has completed."""
        if self.current_time_s >= self.end_time_s and not self.is_playing:
            self.current_time_s = 0.0
        self.is_playing = not self.is_playing
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def restart(self) -> None:
        """Return to launch and resume playback."""
        self.current_time_s = 0.0
        self.is_playing = True
        self._update_artists(update_slider=True)
        self.figure.canvas.draw_idle()

    def advance_one_frame(self) -> tuple[Artist, ...]:
        """Advance by one configured display frame; exposed for deterministic tests."""
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
            self.seek(self.current_time_s - 1.0)
        elif key == "right":
            self.seek(self.current_time_s + 1.0)
        elif key in {"up", "+", "="}:
            self.set_speed(self.playback_speed * 2.0)
        elif key in {"down", "-", "_"}:
            self.set_speed(self.playback_speed * 0.5)

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

    def save_snapshot(self, path: str | Path, *, time_s: float) -> Path:
        """Save one deterministic playback frame for documentation."""
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
        """Export a deterministic GIF at the configured playback speed and frame rate."""
        output_path = Path(path)
        if output_path.suffix.lower() != ".gif":
            raise ValueError("playback export path must end in .gif")
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
                    "title": "AeroGNC-Lab fictional research-flight playback",
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
        """Open the interactive player and block until its window is closed."""
        backend = str(matplotlib.get_backend()).lower()
        if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
            raise RuntimeError(
                f"Matplotlib backend {backend!r} cannot open a playback window; "
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
        """Close the playback figure, primarily for embedding and automated tests."""
        plt.close(self.figure)


def play_three_dof(
    result: SimulationResult,
    *,
    playback_speed: float = 4.0,
    frames_per_second: int = 30,
    repeat: bool = False,
    save_gif: str | Path | None = None,
    show_window: bool = True,
) -> Path | None:
    """Create an optional GIF and/or open an interactive point-mass flight player."""
    player = ThreeDofPlayback(
        result,
        PlaybackConfiguration(
            frames_per_second=frames_per_second,
            initial_speed=playback_speed,
            repeat=repeat,
        ),
    )
    saved_path = player.save_gif(save_gif) if save_gif is not None else None
    if show_window:
        player.show()
    else:
        player.close()
    return saved_path
