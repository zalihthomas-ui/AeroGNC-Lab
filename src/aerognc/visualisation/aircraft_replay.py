"""Seekable playback of recorded aircraft states without re-running dynamics."""

from __future__ import annotations

import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]
from PIL import Image

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.mathematics.geodesy import dcm_inertial_to_ecef
from aerognc.mathematics.quaternion import quaternion_to_dcm
from aerognc.mathematics.vectors import FloatArray
from aerognc.vehicle.fixed_wing import (
    AircraftState,
    initial_tangent_displacement_ned_m,
    local_ned_dcm_inertial,
)
from aerognc.visualisation.aircraft_experience import RecordedFlight, load_recorded_flight
from aerognc.visualisation.mesh import TriangleMesh, load_triangle_mesh
from aerognc.visualisation.style import BLUE, NAVY, ORANGE, engineering_style


class AircraftReplayPlayer:
    """Replay recorded states/telemetry with a seekable timeline and no plant evaluation."""

    def __init__(
        self,
        configuration: AircraftSandboxConfiguration,
        mesh: TriangleMesh,
        recording: RecordedFlight,
        *,
        playback_factor: float = 1.0,
    ) -> None:
        if not np.isfinite(playback_factor) or not 0.1 <= playback_factor <= 20.0:
            raise ValueError("aircraft replay factor must lie in [0.1, 20]")
        self.configuration = configuration
        normalized_mesh = (
            mesh.scaled_to_length(configuration.geometry.wingspan_m)
            if mesh.origin_is_explicit
            else mesh.centered_and_scaled(configuration.geometry.wingspan_m)
        )
        self.mesh = normalized_mesh.decimated()
        self.recording = recording
        self.playback_factor = float(playback_factor)
        self.time_s = float(recording.time_s[0])
        self.state, self.command = recording.sample(self.time_s)
        self.telemetry = recording.sample_telemetry(self.time_s)
        self.is_playing = False
        self._last_wall_time_s = time.perf_counter()
        self._animation: FuncAnimation | None = None
        self._updating_slider = False
        self._initial_position = recording.state[0, :3].copy()
        self._initial_ned = local_ned_dcm_inertial(self._initial_position)
        self._dcm_display_ned = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        self._recorded_display_position = self._build_display_positions()
        with engineering_style():
            self.figure = plt.figure(figsize=(13.8, 7.8))
            grid = self.figure.add_gridspec(
                1,
                4,
                left=0.045,
                right=0.98,
                top=0.90,
                bottom=0.16,
                width_ratios=(1.4, 1.4, 1.4, 1.0),
                wspace=0.28,
            )
            self.scene_axis = cast(Axes3D, self.figure.add_subplot(grid[0, :3], projection="3d"))
            self.telemetry_axis = self.figure.add_subplot(grid[0, 3])
            self._build_scene()
            slider_axis = self.figure.add_axes((0.12, 0.065, 0.70, 0.035))
            self.timeline = Slider(
                slider_axis,
                "Recorded time [s]",
                float(recording.time_s[0]),
                float(recording.time_s[-1]),
                valinit=self.time_s,
            )
            self.timeline.on_changed(self._on_slider)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._update_artists()

    def _display_position(self, position_inertial_m: FloatArray, time_s: float) -> FloatArray:
        ned = initial_tangent_displacement_ned_m(
            position_inertial_m,
            time_s,
            self._initial_position,
            self.configuration.planet.rotation_rate_radps,
        )
        return np.asarray(
            [ned[1], ned[0], self.configuration.initial.altitude_m - ned[2]],
            dtype=np.float64,
        )

    def _dcm_display_inertial(self, time_s: float) -> FloatArray:
        return np.asarray(
            self._dcm_display_ned
            @ self._initial_ned.T
            @ dcm_inertial_to_ecef(self.configuration.planet.rotation_rate_radps * time_s),
            dtype=np.float64,
        )

    def _build_display_positions(self) -> FloatArray:
        sample_count = self.recording.time_s.size
        stride = max(1, int(np.ceil(sample_count / 8_000)))
        indices = np.arange(0, sample_count, stride, dtype=np.int64)
        if indices[-1] != sample_count - 1:
            indices = np.append(indices, sample_count - 1)
        return np.vstack(
            [
                self._display_position(
                    self.recording.state[index, :3], float(self.recording.time_s[index])
                )
                for index in indices
            ]
        )

    def _mesh_triangles(self, view_span_m: float) -> FloatArray:
        typed = AircraftState.from_array(self.state, normalize=True)
        dcm_display_body = self._dcm_display_inertial(self.time_s) @ quaternion_to_dcm(
            typed.quaternion_ib
        )
        position = self._display_position(typed.position_inertial_m, self.time_s)
        base_length = max(float(np.ptp(self.mesh.vertices_body[:, 0])), 1.0e-9)
        scale = max(1.0, 0.032 * view_span_m / base_length)
        return np.asarray(
            self.mesh.triangles_body * scale @ dcm_display_body.T + position,
            dtype=np.float64,
        )

    def _build_scene(self) -> None:
        self.figure.suptitle(
            "Aquila-X1 Recorded State + Telemetry Replay | no plant evaluation",
            color=NAVY,
            fontsize=14,
            fontweight="bold",
        )
        ground_x, ground_y = np.meshgrid(
            np.linspace(-100_000.0, 100_000.0, 11),
            np.linspace(-100_000.0, 100_000.0, 11),
        )
        self.scene_axis.plot_wireframe(
            ground_x,
            ground_y,
            np.zeros_like(ground_x),
            color="#A9A28F",
            alpha=0.16,
            linewidth=0.45,
        )
        (self.track_line,) = self.scene_axis.plot([], [], [], color=BLUE, linewidth=1.8)
        self.aircraft_collection = Poly3DCollection(
            self._mesh_triangles(500.0),
            facecolors="#38A6D8",
            edgecolors="#14384D",
            linewidths=0.4,
            alpha=0.98,
        )
        self.scene_axis.add_collection3d(self.aircraft_collection)
        self.scene_axis.set_xlabel("East from start [m]")
        self.scene_axis.set_ylabel("North from start [m]")
        self.scene_axis.set_zlabel("Altitude [m]")
        self.scene_axis.view_init(elev=25.0, azim=-55.0)
        self.telemetry_axis.axis("off")
        self.telemetry_text = self.telemetry_axis.text(
            0.0,
            1.0,
            "",
            va="top",
            transform=self.telemetry_axis.transAxes,
            family="monospace",
            fontsize=9.0,
        )
        self.mode_text = self.figure.text(
            0.84,
            0.07,
            "PAUSED",
            color=ORANGE,
            fontsize=10,
            fontweight="bold",
        )

    def _recorded_path_to_current(self) -> FloatArray:
        fraction = (self.time_s - self.recording.time_s[0]) / max(
            self.recording.time_s[-1] - self.recording.time_s[0], 1.0e-12
        )
        count = max(1, int(np.ceil(fraction * (self._recorded_display_position.shape[0] - 1))) + 1)
        return self._recorded_display_position[:count]

    def _update_artists(self) -> tuple[Artist, ...]:
        telemetry = self.telemetry
        path = self._recorded_path_to_current()
        self.track_line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
        span = max(500.0, float(np.max(np.ptp(self._recorded_display_position, axis=0))))
        self.aircraft_collection.set_verts(self._mesh_triangles(span))
        margin = 0.12 * span
        minimum = np.min(self._recorded_display_position, axis=0) - margin
        maximum = np.max(self._recorded_display_position, axis=0) + margin
        self.scene_axis.set_xlim(minimum[0], maximum[0])
        self.scene_axis.set_ylim(minimum[1], maximum[1])
        self.scene_axis.set_zlim(min(-20.0, minimum[2]), max(100.0, maximum[2]))
        self.telemetry_text.set_text(
            "RECORDED STATE\n"
            f"Time       {self.time_s:9.2f} s\n"
            f"Altitude   {telemetry.altitude_m:9.1f} m\n"
            f"TAS        {telemetry.true_airspeed_mps:9.1f} m/s\n"
            f"Mach       {telemetry.mach:9.3f}\n"
            f"Heading    {telemetry.heading_deg:9.1f} deg\n"
            f"Bank       {telemetry.roll_deg:+9.1f} deg\n"
            f"Pitch      {telemetry.pitch_deg:+9.1f} deg\n"
            f"AoA        {telemetry.angle_of_attack_deg:+9.2f} deg\n"
            f"normal nz  {telemetry.normal_load_g:+9.2f} g\n"
            f"Throttle   {100.0 * telemetry.throttle:9.1f} %\n\n"
            "Space play/pause\nLeft/Right seek 1 s\nHome restart   Esc close\n\n"
            "State + telemetry are interpolated\nfrom CSV; no plant evaluation."
        )
        self.mode_text.set_text("PLAYING" if self.is_playing else "PAUSED")
        return self.track_line, self.aircraft_collection, self.telemetry_text, self.mode_text

    def set_time(self, time_s: float) -> None:
        """Seek to a recorded time and update the exact-state interpolation."""
        self.time_s = float(np.clip(time_s, self.recording.time_s[0], self.recording.time_s[-1]))
        self.state, self.command = self.recording.sample(self.time_s)
        self.telemetry = self.recording.sample_telemetry(self.time_s)
        if not self._updating_slider and not np.isclose(self.timeline.val, self.time_s):
            self._updating_slider = True
            self.timeline.set_val(self.time_s)
            self._updating_slider = False
        self._update_artists()

    def _on_slider(self, value: float) -> None:
        if not self._updating_slider:
            self.set_time(float(value))

    def _animation_frame(self, _frame: int) -> tuple[Artist, ...]:
        now_s = time.perf_counter()
        wall_delta_s = max(0.0, now_s - self._last_wall_time_s)
        self._last_wall_time_s = now_s
        if self.is_playing:
            next_time = self.time_s + wall_delta_s * self.playback_factor
            if next_time >= self.recording.time_s[-1]:
                next_time = float(self.recording.time_s[-1])
                self.is_playing = False
            self.set_time(next_time)
        return self._update_artists()

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "key", None)
        if not isinstance(key, str):
            return
        key = key.casefold()
        if key in {" ", "space", "p"}:
            self.is_playing = not self.is_playing
            self._last_wall_time_s = time.perf_counter()
        elif key == "left":
            self.set_time(self.time_s - 1.0)
        elif key == "right":
            self.set_time(self.time_s + 1.0)
        elif key == "home":
            self.is_playing = False
            self.set_time(float(self.recording.time_s[0]))
        elif key == "escape":
            plt.close(self.figure)

    def _on_close(self, _event: object) -> None:
        self.is_playing = False
        if self._animation is not None and self._animation.event_source is not None:
            self._animation.event_source.stop()  # type: ignore[no-untyped-call]
        self._animation = None

    def show(self, *, block: bool = True) -> None:
        """Open the replay window with a lightweight presentation timer."""
        self._last_wall_time_s = time.perf_counter()
        self._animation = FuncAnimation(
            self.figure,
            self._animation_frame,
            interval=33.0,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=block)

    def export_gif(
        self,
        path: str | Path,
        *,
        maximum_duration_s: float = 20.0,
        frames_per_second: int = 12,
        width_px: int = 960,
        height_px: int = 540,
    ) -> Path:
        """Export a bounded recorded-state GIF without advancing the simulator."""
        values = np.asarray([maximum_duration_s, width_px, height_px], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("GIF duration and dimensions must be positive and finite")
        if isinstance(frames_per_second, bool) or not 2 <= frames_per_second <= 20:
            raise ValueError("GIF frame rate must lie in [2, 20]")
        if width_px > 1_280 or height_px > 720:
            raise ValueError("GIF dimensions cannot exceed 1280 by 720")
        start_s = float(self.recording.time_s[0])
        end_s = min(float(self.recording.time_s[-1]), start_s + maximum_duration_s)
        frame_count = int(np.ceil((end_s - start_s) * frames_per_second)) + 1
        frame_count = int(np.clip(frame_count, 2, 400))
        frame_times = np.linspace(start_s, end_s, frame_count)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        original_size = self.figure.get_size_inches().copy()
        original_time_s = self.time_s
        self.figure.set_size_inches(width_px / 100.0, height_px / 100.0)
        images: list[Image.Image] = []
        try:
            for frame_time_s in frame_times:
                self.set_time(float(frame_time_s))
                buffer = BytesIO()
                self.figure.savefig(buffer, format="png", dpi=100)
                buffer.seek(0)
                with Image.open(buffer) as frame:
                    images.append(frame.convert("RGBA").copy())
            images[0].save(
                output,
                save_all=True,
                append_images=images[1:],
                duration=round(1_000.0 / frames_per_second),
                loop=0,
                optimize=False,
            )
        finally:
            self.figure.set_size_inches((float(original_size[0]), float(original_size[1])))
            self.set_time(original_time_s)
            for image in images:
                image.close()
        return output


def play_aircraft_recording(
    configuration: AircraftSandboxConfiguration,
    recording_path: str | Path,
    mesh_path: str | Path = "assets/models/aquila_x1.obj",
    *,
    playback_factor: float = 1.0,
) -> None:
    """Load one recorder CSV and open the seekable exact-state replay."""
    recording = load_recorded_flight(recording_path)
    mesh = load_triangle_mesh(mesh_path)
    AircraftReplayPlayer(
        configuration,
        mesh,
        recording,
        playback_factor=playback_factor,
    ).show()
