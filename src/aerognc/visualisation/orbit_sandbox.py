"""Publication plots and interactive 3D playback for the near-planet orbit sandbox."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

from aerognc.simulation.orbit_sandbox import (
    ORBIT_MODEL_DESCRIPTIONS,
    OrbitSandboxSimulation,
)
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style


def _equal_3d_limits(axis: Axes3D, positions_km: np.ndarray, margin: float = 0.12) -> None:
    minimum = np.min(positions_km, axis=0)
    maximum = np.max(positions_km, axis=0)
    center = 0.5 * (minimum + maximum)
    half_span = max(0.5 * float(np.max(maximum - minimum)), 1.0) * (1.0 + margin)
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)


def plot_orbit_sandbox(
    simulation: OrbitSandboxSimulation, output_directory: str | Path
) -> tuple[Path, Path]:
    """Save a 3D orbit view and time-history diagnostics at publication resolution."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = simulation.result
    columns = result.columns
    position_km = np.column_stack((columns["x_m"], columns["y_m"], columns["z_m"])) / 1_000.0
    radius_km = simulation.configuration.primary.radius_m / 1_000.0
    with engineering_style():
        figure = plt.figure(figsize=(8.8, 7.0))
        axis = cast(Axes3D, figure.add_subplot(111, projection="3d"))
        longitude = np.linspace(0.0, 2.0 * np.pi, 48)
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 24)
        x = radius_km * np.outer(np.cos(latitude), np.cos(longitude))
        y = radius_km * np.outer(np.cos(latitude), np.sin(longitude))
        z = radius_km * np.outer(np.sin(latitude), np.ones_like(longitude))
        axis.plot_surface(x, y, z, color="#4C8FD8", alpha=0.40, linewidth=0.0)
        axis.plot(
            position_km[:, 0],
            position_km[:, 1],
            position_km[:, 2],
            color=ORANGE,
            label=simulation.configuration.satellite.name,
        )
        axis.scatter(*position_km[0], color=GREEN, s=34, label="Start")
        axis.scatter(*position_km[-1], color=RED, s=34, label="End")
        axis.set_title(f"{simulation.configuration.model.replace('_', ' ').title()} trajectory")
        axis.set_xlabel("Planet-centred X [km]")
        axis.set_ylabel("Planet-centred Y [km]")
        axis.set_zlabel("Planet-centred Z [km]")
        axis.legend(loc="upper left")
        _equal_3d_limits(axis, np.vstack((position_km, np.array([[radius_km, 0.0, 0.0]]))))
        trajectory_path = output / "orbit_trajectory_3d.png"
        figure.savefig(trajectory_path, dpi=220)
        plt.close(figure)

        figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.2), sharex=True)
        days = result.time_s / 86_400.0
        axes[0].plot(days, columns["altitude_m"] / 1_000.0, color=BLUE, label="Altitude")
        axes[0].plot(
            days,
            columns["perigee_altitude_m"] / 1_000.0,
            color=ORANGE,
            linestyle="--",
            label="Osculating perigee",
        )
        axes[0].plot(
            days,
            columns["apogee_altitude_m"] / 1_000.0,
            color=GREEN,
            linestyle=":",
            label="Osculating apogee",
        )
        axes[0].axhline(
            simulation.configuration.reentry_altitude_m / 1_000.0,
            color=RED,
            linewidth=1.0,
            label="Reentry threshold",
        )
        axes[0].set_ylabel("Altitude [km]")
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].plot(days, columns["speed_mps"] / 1_000.0, color=BLUE)
        axes[1].set_ylabel("Speed [km/s]")
        axes[1].twinx().plot(
            days,
            columns["eccentricity"],
            color=ORANGE,
            alpha=0.8,
        )
        axes[1].set_title("Speed (blue) and osculating eccentricity (orange)", fontsize=9)
        axes[2].semilogy(
            days,
            np.maximum(columns["atmospheric_density_kgpm3"], 1.0e-30),
            color=GREEN,
            label="Reference density",
        )
        drag_axis = axes[2].twinx()
        drag_axis.semilogy(
            days,
            np.maximum(columns["drag_acceleration_mps2"], 1.0e-30),
            color=RED,
            label="Drag acceleration",
        )
        axes[2].set_ylabel("Density [kg/m3]")
        drag_axis.set_ylabel("Drag acceleration [m/s2]")
        axes[2].set_xlabel("Modeled time [days]")
        figure.suptitle(simulation.survival_statement, fontsize=10, fontweight="bold")
        diagnostics_path = output / "orbit_decay_diagnostics.png"
        figure.savefig(diagnostics_path, dpi=220)
        plt.close(figure)
    return trajectory_path, diagnostics_path


class OrbitSandboxPlayback:
    """Seekable 3D playback with satellite/system focus and explicit lifetime scope."""

    def __init__(self, simulation: OrbitSandboxSimulation, frames_per_second: int = 30) -> None:
        if isinstance(frames_per_second, bool) or not 10 <= frames_per_second <= 60:
            raise ValueError("frames_per_second must lie in [10, 60]")
        self.simulation = simulation
        self.result = simulation.result
        self.configuration = simulation.configuration
        self.frames_per_second = frames_per_second
        self.current_time_s = 0.0
        self.is_playing = True
        self.system_focus = False
        self.playback_rate = max(
            self.configuration.output_step_s,
            float(self.result.time_s[-1]) / 25.0,
        )
        self._updating_slider = False
        self._animation: FuncAnimation | None = None
        self._positions_km = (
            np.column_stack(
                (
                    self.result.columns["x_m"],
                    self.result.columns["y_m"],
                    self.result.columns["z_m"],
                )
            )
            / 1_000.0
        )
        with engineering_style():
            self.figure = plt.figure(figsize=(12.2, 7.4))
            grid = self.figure.add_gridspec(
                1, 4, left=0.05, right=0.97, top=0.90, bottom=0.19, wspace=0.28
            )
            self.axis = cast(Axes3D, self.figure.add_subplot(grid[0, :3], projection="3d"))
            self.telemetry_axis = self.figure.add_subplot(grid[0, 3])
            self._build_scene()
            self._build_controls()
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._update_artists(update_slider=False)

    def _build_scene(self) -> None:
        model_label = self.configuration.model.replace("_", " ").title()
        self.figure.suptitle(
            f"Satellite Orbit Sandbox | {model_label}",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        radius_km = self.configuration.primary.radius_m / 1_000.0
        longitude = np.linspace(0.0, 2.0 * np.pi, 42)
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 22)
        self.axis.plot_surface(
            radius_km * np.outer(np.cos(latitude), np.cos(longitude)),
            radius_km * np.outer(np.cos(latitude), np.sin(longitude)),
            radius_km * np.outer(np.sin(latitude), np.ones_like(longitude)),
            color=self.configuration.primary.color,
            alpha=0.38,
            linewidth=0.0,
            shade=True,
        )
        (self.full_path,) = self.axis.plot(
            self._positions_km[:, 0],
            self._positions_km[:, 1],
            self._positions_km[:, 2],
            color="#BCC5CC",
            linewidth=0.9,
            linestyle="--",
        )
        (self.trail,) = self.axis.plot([], [], [], color=ORANGE, linewidth=2.0)
        self.satellite_marker = self.axis.scatter([], [], [], color=RED, s=38, depthshade=False)
        self.secondary_markers = []
        for secondary in self.configuration.secondaries:
            marker = self.axis.scatter(
                [], [], [], color=secondary.color, s=28, label=secondary.name, depthshade=False
            )
            self.secondary_markers.append(marker)
        self.axis.set_xlabel("Planet-centred X [km]")
        self.axis.set_ylabel("Planet-centred Y [km]")
        self.axis.set_zlabel("Planet-centred Z [km]")
        self.axis.view_init(elev=27.0, azim=-55.0)
        self.telemetry_axis.axis("off")
        self.telemetry_text = self.telemetry_axis.text(
            0.0,
            1.0,
            "",
            va="top",
            transform=self.telemetry_axis.transAxes,
            family="monospace",
            fontsize=8.6,
        )
        self.scope_text = self.telemetry_axis.text(
            0.0,
            0.04,
            self.simulation.survival_statement,
            va="bottom",
            transform=self.telemetry_axis.transAxes,
            wrap=True,
            color=GREEN if not self.simulation.reentered else RED,
            fontsize=8.4,
            fontweight="bold",
        )

    def _build_controls(self) -> None:
        slider_axis = self.figure.add_axes((0.16, 0.09, 0.62, 0.035))
        self.time_slider = Slider(
            slider_axis,
            "Modeled time [days]",
            0.0,
            float(self.result.time_s[-1]) / 86_400.0,
            valinit=0.0,
        )
        self.time_slider.on_changed(self._on_slider)
        play_axis = self.figure.add_axes((0.05, 0.075, 0.08, 0.06))
        self.play_button = Button(play_axis, "Pause")
        self.play_button.on_clicked(self._toggle_play)
        focus_axis = self.figure.add_axes((0.81, 0.075, 0.13, 0.06))
        self.focus_button = Button(focus_axis, "Show system")
        self.focus_button.on_clicked(self._toggle_focus)
        self.figure.text(
            0.16,
            0.035,
            "Space play/pause | Home restart | +/- speed | C satellite/system focus | "
            "drag slider to seek",
            fontsize=8,
            color="#5D6873",
        )

    def _index(self) -> int:
        return int(
            np.clip(
                np.searchsorted(self.result.time_s, self.current_time_s),
                0,
                self.result.time_s.size - 1,
            )
        )

    def _set_limits(self, index: int) -> None:
        radius_km = self.configuration.primary.radius_m / 1_000.0
        if self.system_focus:
            points = [self._positions_km]
            for secondary in self.configuration.secondaries:
                key = secondary.name.casefold().replace(" ", "_")
                points.append(
                    np.column_stack(
                        (
                            self.result.columns[f"{key}_x_m"],
                            self.result.columns[f"{key}_y_m"],
                            self.result.columns[f"{key}_z_m"],
                        )
                    )
                    / 1_000.0
                )
            _equal_3d_limits(self.axis, np.vstack(points), margin=0.08)
        else:
            local_span = max(
                radius_km * 1.12,
                float(np.linalg.norm(self._positions_km[index])) * 1.08,
            )
            self.axis.set_xlim(-local_span, local_span)
            self.axis.set_ylim(-local_span, local_span)
            self.axis.set_zlim(-local_span, local_span)

    def _update_artists(self, *, update_slider: bool = True) -> tuple[Artist, ...]:
        index = self._index()
        position = self._positions_km[index]
        trail_start = max(0, index - 500)
        trail = self._positions_km[trail_start : index + 1]
        self.trail.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        self.satellite_marker._offsets3d = ([position[0]], [position[1]], [position[2]])
        for secondary, marker in zip(
            self.configuration.secondaries, self.secondary_markers, strict=True
        ):
            key = secondary.name.casefold().replace(" ", "_")
            marker._offsets3d = (
                [self.result.columns[f"{key}_x_m"][index] / 1_000.0],
                [self.result.columns[f"{key}_y_m"][index] / 1_000.0],
                [self.result.columns[f"{key}_z_m"][index] / 1_000.0],
            )
            marker.set_visible(self.system_focus)
        self._set_limits(index)
        columns = self.result.columns
        self.telemetry_text.set_text(
            "ORBIT TELEMETRY\n"
            f"Model        {self.configuration.model}\n"
            f"Time         {self.result.time_s[index] / 86_400.0:9.4f} days\n"
            f"Altitude     {columns['altitude_m'][index] / 1_000.0:9.2f} km\n"
            f"Speed        {columns['speed_mps'][index] / 1_000.0:9.4f} km/s\n"
            f"Perigee      {columns['perigee_altitude_m'][index] / 1_000.0:9.2f} km\n"
            f"Apogee       {columns['apogee_altitude_m'][index] / 1_000.0:9.2f} km\n"
            f"Eccentricity {columns['eccentricity'][index]:9.6f}\n"
            f"Revolutions  {columns['revolutions_completed'][index]:9.3f}\n"
            f"Mass         {columns['mass_kg'][index]:9.2f} kg\n"
            f"Drag accel   {columns['drag_acceleration_mps2'][index]:9.3e} m/s2\n"
            f"Corrections  {len(self.simulation.correction_burns):9d}\n"
            f"Playback     {self.playback_rate:9.1f} sim s/real s\n\n"
            "WHAT THIS MODE MEANS\n"
            f"{ORBIT_MODEL_DESCRIPTIONS[self.configuration.model]}"
        )
        if update_slider:
            self._updating_slider = True
            self.time_slider.set_val(self.current_time_s / 86_400.0)
            self._updating_slider = False
        return (self.trail, self.satellite_marker, self.telemetry_text, self.scope_text)

    def _animation_frame(self, _frame: int) -> tuple[Artist, ...]:
        if self.is_playing:
            self.current_time_s += self.playback_rate / self.frames_per_second
            if self.current_time_s >= self.result.time_s[-1]:
                self.current_time_s = float(self.result.time_s[-1])
                self.is_playing = False
                self.play_button.label.set_text("Play")
        return self._update_artists()

    def _on_slider(self, value_days: float) -> None:
        if self._updating_slider:
            return
        self.current_time_s = float(value_days) * 86_400.0
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def _toggle_play(self, _event: object) -> None:
        if self.current_time_s >= self.result.time_s[-1]:
            self.current_time_s = 0.0
        self.is_playing = not self.is_playing
        self.play_button.label.set_text("Pause" if self.is_playing else "Play")

    def _toggle_focus(self, _event: object) -> None:
        self.system_focus = not self.system_focus
        self.focus_button.label.set_text("Show satellite" if self.system_focus else "Show system")
        self._update_artists(update_slider=False)
        self.figure.canvas.draw_idle()

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key in {" ", "space"}:
            self._toggle_play(event)
        elif key == "home":
            self.current_time_s = 0.0
            self.is_playing = True
        elif key in {"+", "="}:
            self.playback_rate *= 1.5
        elif key in {"-", "_"}:
            self.playback_rate = max(self.configuration.output_step_s, self.playback_rate / 1.5)
        elif key in {"c", "C"}:
            self._toggle_focus(event)

    def _on_close(self, _event: object) -> None:
        self._animation = None
        self.is_playing = False

    def show(self) -> None:
        """Open the seekable 3D orbit player."""
        self._animation = FuncAnimation(
            self.figure,
            self._animation_frame,
            interval=1_000.0 / self.frames_per_second,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def play_orbit_sandbox(simulation: OrbitSandboxSimulation, frames_per_second: int = 30) -> None:
    """Open interactive orbit playback for an already propagated simulation."""
    OrbitSandboxPlayback(simulation, frames_per_second).show()
