"""Live 3D keyboard/controller flight for the fictional Aquila-X1 plant."""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d.art3d import (  # type: ignore[import-untyped]
    Line3D,
    Line3DCollection,
    Poly3DCollection,
)
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.mathematics.geodesy import dcm_inertial_to_ecef
from aerognc.mathematics.integrators import rk4_step
from aerognc.mathematics.quaternion import quaternion_to_dcm
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.aircraft_telemetry import AircraftTelemetry, aircraft_telemetry
from aerognc.simulation.aircraft_training import (
    TrainingTask,
    evaluate_training_task,
    scripted_demo_command,
)
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    AircraftState,
    FixedWingFlightModel,
    aircraft_initial_state,
    initial_tangent_displacement_ned_m,
    local_ned_dcm_inertial,
    longitudinal_trim_command,
    project_aircraft_state,
)
from aerognc.visualisation.aircraft_controls import (
    AircraftControlMode,
    PilotControlProfile,
    VirtualPilotStick,
    apply_stability_assist,
    shape_pilot_command,
)
from aerognc.visualisation.aircraft_experience import (
    TRAIL_MODES,
    FlightRecorder,
    FlightTrailBuffer,
    OperatingEnvelopeLimits,
    RealtimeClockTick,
    RealtimeSimulationClock,
    TrailMode,
    TrailSettings,
    classify_touchdown,
    evaluate_flight_warnings,
    interpolate_ground_contact,
)
from aerognc.visualisation.mesh import TriangleMesh, load_triangle_mesh
from aerognc.visualisation.pilot_input import GamepadSnapshot, XInputGamepad
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style

LiveCameraMode = Literal["chase", "cockpit", "orbit", "top", "free"]
LIVE_CAMERA_MODES: tuple[LiveCameraMode, ...] = (
    "chase",
    "cockpit",
    "orbit",
    "top",
    "free",
)
MeshScaleMode = Literal["enlarged_marker", "true_scale"]


def advance_live_aircraft(
    model: FixedWingFlightModel,
    state: FloatArray,
    time_s: float,
    duration_s: float,
    command: AircraftControlCommand,
) -> FloatArray:
    """Advance a held-command live plant using its configured deterministic RK4 step."""
    if not np.isfinite([time_s, duration_s]).all() or time_s < 0.0 or duration_s <= 0.0:
        raise ValueError("live advance time/duration must be finite and positive")
    configuration = model.configuration
    remaining = duration_s
    current_time = time_s
    values = state.copy()

    def derivative(stage_time_s: float, stage_values: FloatArray) -> FloatArray:
        return model.derivative(stage_time_s, stage_values, command)

    while remaining > 1.0e-12:
        step_s = min(configuration.integration_step_s, remaining)
        values = project_aircraft_state(
            rk4_step(derivative, current_time, values, step_s), configuration
        )
        current_time += step_s
        remaining -= step_s
    return values


def research_ascent_assist_command(
    model: FixedWingFlightModel,
    state: AircraftState,
    time_s: float,
    start_time_s: float = 0.0,
) -> AircraftControlCommand:
    """Return a roll-level, pitch-ramp command for a civilian 100 km research attempt.

    This is an optional attitude aid for exploring the atmosphere-to-space transition. It
    commands no target, interception, terminal guidance, or orbital insertion maneuver.
    """
    if not np.isfinite([time_s, start_time_s]).all() or time_s < start_time_s:
        raise ValueError("research-ascent times must be finite and ordered")
    roll, pitch, _heading = model.local_attitude_rad(state.as_array())
    roll_rate, pitch_rate, _yaw_rate = state.angular_rate_body_radps
    elapsed = time_s - start_time_s
    initial_pitch = (
        model.configuration.initial.flight_path_angle_rad
        + model.configuration.initial.angle_of_attack_rad
    )
    target_pitch = min(np.deg2rad(78.0), initial_pitch + np.deg2rad(7.0) * elapsed)
    pitch_command = float(np.clip(1.8 * (target_pitch - pitch) - 0.55 * pitch_rate, -1.0, 1.0))
    roll_command = float(np.clip(-1.5 * roll - 0.4 * roll_rate, -1.0, 1.0))
    return AircraftControlCommand(
        roll=roll_command,
        pitch=pitch_command,
        yaw=0.0,
        throttle=1.0,
        rocket_assist=True,
    )


class AircraftLivePlayer:
    """Interactive 3D player where every pilot input enters the nonlinear plant."""

    def __init__(
        self,
        configuration: AircraftSandboxConfiguration,
        mesh: TriangleMesh,
        *,
        frames_per_second: int = 30,
        real_time_factor: float = 1.0,
        camera_mode: LiveCameraMode = "chase",
        enable_gamepad: bool = True,
        control_profile: PilotControlProfile | None = None,
        trail_settings: TrailSettings | None = None,
        envelope_limits: OperatingEnvelopeLimits | None = None,
        mesh_scale_mode: MeshScaleMode = "enlarged_marker",
        initially_paused: bool = True,
        recorder_directory: str | Path | None = None,
        simulation_clock: RealtimeSimulationClock | None = None,
        training_task: TrainingTask | None = None,
    ) -> None:
        if isinstance(frames_per_second, bool) or not 10 <= frames_per_second <= 120:
            raise ValueError("frames_per_second must lie in [10, 120]")
        if not np.isfinite(real_time_factor) or not 0.1 <= real_time_factor <= 10.0:
            raise ValueError("real_time_factor must lie in [0.1, 10]")
        if camera_mode not in LIVE_CAMERA_MODES:
            raise ValueError(f"camera_mode must be one of {LIVE_CAMERA_MODES}")
        if mesh_scale_mode not in ("enlarged_marker", "true_scale"):
            raise ValueError("mesh scale mode must be enlarged_marker or true_scale")
        self.configuration = configuration
        self.model = FixedWingFlightModel(configuration, wind_horizon_s=3_601.0)
        normalized_mesh = (
            mesh.scaled_to_length(configuration.geometry.wingspan_m)
            if mesh.origin_is_explicit
            else mesh.centered_and_scaled(configuration.geometry.wingspan_m)
        )
        self.mesh = normalized_mesh.decimated()
        self.frames_per_second = frames_per_second
        self.real_time_factor = real_time_factor
        self.camera_mode: LiveCameraMode = camera_mode
        self.mesh_scale_mode: MeshScaleMode = mesh_scale_mode
        self.gamepad = XInputGamepad() if enable_gamepad else None
        self.control_profile = control_profile or PilotControlProfile()
        self.virtual_stick = VirtualPilotStick()
        self._trim_command = longitudinal_trim_command(configuration).pitch
        self.virtual_stick.pitch_trim = self._trim_command
        self.trail_settings = trail_settings or TrailSettings()
        self.trail_mode: TrailMode = self.trail_settings.mode
        self.trail = FlightTrailBuffer(self.trail_settings)
        self.envelope_limits = envelope_limits or OperatingEnvelopeLimits(
            maximum_mach=configuration.propulsion.maximum_operating_mach,
            angle_of_attack_warning_deg=0.9
            * np.rad2deg(configuration.aerodynamics.stall_angle_rad),
        )
        self.recorder = FlightRecorder()
        self.recorder_directory = Path(
            recorder_directory or configuration.output_directory / "live_session"
        )
        self.pressed_keys: set[str] = set()
        self.keyboard_throttle = configuration.initial_throttle
        self.time_s = 0.0
        self.state = aircraft_initial_state(configuration)
        self.is_paused = initially_paused
        self.pause_reason = "ready"
        self.finished_reason = ""
        self.space_assist_start_s: float | None = None
        self.scripted_demo_enabled = False
        self.training_task = training_task
        self.show_help = False
        self.overlay_body_axes = False
        self.overlay_velocity = True
        self.overlay_wind = False
        self.overlay_prediction = True
        self._animation: FuncAnimation | None = None
        self._replay_players: list[object] = []
        self._initial_position = self.state[:3].copy()
        self._initial_ned = local_ned_dcm_inertial(self._initial_position)
        self._dcm_display_ned = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        )
        self._history_time: deque[float] = deque(maxlen=900)
        self._history_altitude: deque[float] = deque(maxlen=900)
        self._history_airspeed: deque[float] = deque(maxlen=900)
        self._history_alpha: deque[float] = deque(maxlen=900)
        self._next_log_time_s = 0.1
        self._camera_azimuth_deg = -55.0
        # In "free" mode the world limits are fixed once so the aircraft visibly
        # translates through a stationary scene and the user owns rotation/zoom.
        self._free_limits_set = False
        self._last_saved_message = ""
        self._active_warning_codes: set[str] = set()
        self._clock = simulation_clock or RealtimeSimulationClock(
            configuration.integration_step_s,
            real_time_factor,
            maximum_catch_up_s=max(0.5, 5.0 * configuration.integration_step_s),
        )
        self._last_clock_tick = RealtimeClockTick(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._clock.resynchronize()
        initial_command = self._current_command(GamepadSnapshot(False))
        self._record_sample(initial_command, force=True)

        with engineering_style():
            self.figure = plt.figure(figsize=(15.6, 8.8))
            grid = self.figure.add_gridspec(
                3,
                5,
                left=0.035,
                right=0.985,
                top=0.925,
                bottom=0.075,
                width_ratios=(1.4, 1.4, 1.4, 1.4, 1.05),
                height_ratios=(1.25, 0.78, 0.92),
                hspace=0.32,
                wspace=0.30,
            )
            self.scene_axis = cast(Axes3D, self.figure.add_subplot(grid[:, :4], projection="3d"))
            self.telemetry_axis = self.figure.add_subplot(grid[0, 4])
            self.map_axis = self.figure.add_subplot(grid[1, 4])
            self.history_axis = self.figure.add_subplot(grid[2, 4])
            self._build_scene()
            self._build_map()
            self._build_history()
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.figure.canvas.mpl_connect("figure_leave_event", self._on_focus_lost)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._update_artists(initial_command)

    def _display_position(self, position_inertial_m: FloatArray, time_s: float) -> FloatArray:
        displacement_ned = initial_tangent_displacement_ned_m(
            position_inertial_m,
            time_s,
            self._initial_position,
            self.configuration.planet.rotation_rate_radps,
        )
        return np.asarray(
            [
                displacement_ned[1],
                displacement_ned[0],
                self.configuration.initial.altitude_m - displacement_ned[2],
            ],
            dtype=np.float64,
        )

    def _dcm_display_inertial(self, time_s: float) -> FloatArray:
        return np.asarray(
            self._dcm_display_ned
            @ self._initial_ned.T
            @ dcm_inertial_to_ecef(self.configuration.planet.rotation_rate_radps * time_s),
            dtype=np.float64,
        )

    def _build_scene(self) -> None:
        self.figure.suptitle(
            "Aquila-X1 Flight Deck | nonlinear fictional civilian research aircraft",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        # Dynamic local ground grid: re-tiled each frame to the current view with a
        # stable "nice" line spacing. This replaces a single huge fixed wireframe
        # whose sparse gridlines popped in and out as the view moved (the glitch).
        self.ground_grid = Line3DCollection(
            [], colors="#A9A28F", linewidths=0.5, alpha=0.35
        )
        self.scene_axis.add_collection3d(self.ground_grid, autolim=False)
        self.scene_axis.plot(
            [-200_000.0, 200_000.0],
            [0.0, 0.0],
            [100_000.0, 100_000.0],
            color=ORANGE,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label="100 km reference (not orbit)",
        )
        self.trail_collection = Line3DCollection([], linewidths=2.0, alpha=0.95)
        self.scene_axis.add_collection3d(self.trail_collection, autolim=False)
        initial_triangles = self._mesh_triangles(500.0)
        self.aircraft_collection = Poly3DCollection(
            initial_triangles,
            facecolors="#38A6D8",
            edgecolors="#14384D",
            linewidths=0.42,
            alpha=0.98,
        )
        self.scene_axis.add_collection3d(self.aircraft_collection)
        current = self._display_position(self.state[:3], self.time_s)
        (self.ground_marker,) = self.scene_axis.plot(
            [current[0]], [current[1]], [0.0], marker="o", color="#30363D", markersize=4
        )
        (self.shadow_line,) = self.scene_axis.plot(
            [current[0], current[0]],
            [current[1], current[1]],
            [0.0, current[2]],
            color="#59636C",
            linestyle=":",
            linewidth=0.8,
            alpha=0.65,
        )
        self.body_axis_lines = tuple(
            self.scene_axis.plot([], [], [], linewidth=1.4, color=color)[0]
            for color in (RED, GREEN, BLUE)
        )
        (self.velocity_line,) = self.scene_axis.plot([], [], [], color=GREEN, linewidth=1.6)
        (self.wind_line,) = self.scene_axis.plot([], [], [], color=ORANGE, linewidth=1.5)
        (self.prediction_line,) = self.scene_axis.plot(
            [], [], [], color="#6574CD", linestyle="--", linewidth=1.2
        )
        self.scene_axis.set_xlabel("East from start [m]")
        self.scene_axis.set_ylabel("North from start [m]")
        self.scene_axis.set_zlabel("Altitude [m]")
        self.scene_axis.set_box_aspect((1.0, 1.0, 0.62))
        self.scene_axis.legend(loc="upper left", fontsize=7)
        self.telemetry_axis.axis("off")
        self.telemetry_text = self.telemetry_axis.text(
            0.0,
            1.0,
            "",
            transform=self.telemetry_axis.transAxes,
            va="top",
            family="monospace",
            fontsize=8.35,
        )
        self.warning_text = self.telemetry_axis.text(
            0.0,
            -0.02,
            "",
            transform=self.telemetry_axis.transAxes,
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            wrap=True,
        )
        self.start_text = self.scene_axis.text2D(
            0.5,
            0.53,
            "READY TO FLY\nClick this window, then press Space\nPress H for controls",
            transform=self.scene_axis.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#F5F7FA",
            bbox={"boxstyle": "round,pad=0.8", "facecolor": "#081621", "alpha": 0.90},
        )
        self.help_text = self.scene_axis.text2D(
            0.025,
            0.97,
            self._help_message(),
            transform=self.scene_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
            family="monospace",
            color="#F5F7FA",
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#081621", "alpha": 0.94},
        )
        self.help_text.set_visible(False)
        self.status_text = self.figure.text(
            0.04,
            0.025,
            "",
            color=NAVY,
            fontsize=8.5,
            family="monospace",
        )

    def _build_map(self) -> None:
        (self.map_track_line,) = self.map_axis.plot([], [], color=BLUE, linewidth=1.3)
        (self.map_aircraft_marker,) = self.map_axis.plot(
            [], [], marker="^", color=RED, markersize=6, linestyle="None"
        )
        self.map_axis.axhline(0.0, color="#9AA4AD", linewidth=0.55, alpha=0.5)
        self.map_axis.axvline(0.0, color="#9AA4AD", linewidth=0.55, alpha=0.5)
        self.map_axis.set_title("Ground track")
        self.map_axis.set_xlabel("East [m]")
        self.map_axis.set_ylabel("North [m]")
        self.map_axis.grid(True, alpha=0.22)

    def _build_history(self) -> None:
        (self.altitude_line,) = self.history_axis.plot([], [], color=BLUE, label="Alt / 100 m")
        (self.airspeed_line,) = self.history_axis.plot([], [], color=GREEN, label="TAS m/s")
        (self.alpha_line,) = self.history_axis.plot([], [], color=ORANGE, label="AoA deg")
        self.history_axis.axhline(
            np.rad2deg(self.configuration.aerodynamics.stall_angle_rad),
            color=RED,
            linewidth=0.85,
            linestyle="--",
            label="stall AoA",
        )
        self.history_axis.set_title("Last 60 simulation seconds")
        self.history_axis.set_xlabel("Time [s]")
        self.history_axis.grid(True, alpha=0.22)
        self.history_axis.legend(fontsize=6.2, loc="upper left", ncols=2)

    def _help_message(self) -> str:
        bindings = self.control_profile.bindings
        return (
            "FLIGHT CONTROLS\n"
            f"{bindings.roll_left}/{bindings.roll_right}: roll   "
            f"{bindings.pitch_up}/{bindings.pitch_down}: pitch\n"
            f"{bindings.yaw_left}/{bindings.yaw_right}: yaw    "
            f"{bindings.throttle_up}/{bindings.throttle_down}: throttle\n"
            f"{bindings.trim_nose_down}/{bindings.trim_nose_up}: pitch trim   "
            f"hold {bindings.wings_level}: wings level\n"
            f"hold {bindings.rocket_assist}: research rocket assist   T: ascent aid\n"
            "F6: reproducible civilian demo inputs\n\n"
            "FLIGHT DECK\n"
            "Space/P pause   Home reset   C camera   0 chase view\n"
            "V trail mode   X clear trail   M true/enlarged mesh\n"
            "1 body axes   2 velocity   3 wind   4 prediction\n"
            "Tab direct/assisted   F9 save   F10 replay   G GIF   F12 screenshot\n"
            "H close help   Esc close\n\n"
            "XInput: left stick roll/pitch, right-X yaw, RT throttle, A rocket"
        )

    def _raw_pilot_command(self, gamepad: GamepadSnapshot) -> AircraftControlCommand:
        roll, pitch, yaw = self.virtual_stick.roll, self.virtual_stick.pitch, self.virtual_stick.yaw
        if gamepad.connected:
            if abs(gamepad.roll) > abs(roll):
                roll = gamepad.roll
            if abs(gamepad.pitch) > abs(pitch):
                pitch = gamepad.pitch
            if abs(gamepad.yaw) > abs(yaw):
                yaw = gamepad.yaw
        throttle = self.keyboard_throttle if gamepad.throttle is None else gamepad.throttle
        return AircraftControlCommand(
            roll,
            pitch,
            yaw,
            throttle,
            self.control_profile.bindings.rocket_assist in self.pressed_keys
            or gamepad.rocket_assist,
        )

    def _current_command(self, gamepad: GamepadSnapshot) -> AircraftControlCommand:
        if self.space_assist_start_s is not None:
            return research_ascent_assist_command(
                self.model,
                AircraftState.from_array(self.state, normalize=True),
                self.time_s,
                self.space_assist_start_s,
            )
        if self.scripted_demo_enabled:
            return scripted_demo_command(
                self.time_s,
                AircraftState.from_array(self.state, normalize=True),
                self.configuration,
            )
        shaped = shape_pilot_command(self._raw_pilot_command(gamepad), self.control_profile)
        return apply_stability_assist(
            self.model,
            AircraftState.from_array(self.state, normalize=True),
            shaped,
            self.control_profile,
            pitch_trim=self.virtual_stick.pitch_trim,
            wings_level=self.control_profile.bindings.wings_level in self.pressed_keys,
        )

    def _mesh_triangles(self, view_span_m: float) -> FloatArray:
        state = AircraftState.from_array(self.state, normalize=True)
        dcm_display_body = (
            self._dcm_display_inertial(self.time_s) @ quaternion_to_dcm(state.quaternion_ib)
        )
        position = self._display_position(state.position_inertial_m, self.time_s)
        base_length = max(float(np.ptp(self.mesh.vertices_body[:, 0])), 1.0e-9)
        if self.mesh_scale_mode == "true_scale":
            scale = 1.0
        else:
            scale = max(1.0, 0.035 * view_span_m / base_length)
        return np.asarray(
            self.mesh.triangles_body * scale @ dcm_display_body.T + position,
            dtype=np.float64,
        )

    def _snapshot(self, command: AircraftControlCommand) -> AircraftTelemetry:
        return aircraft_telemetry(
            self.model,
            self.time_s,
            self.state,
            command,
            initial_position_inertial_m=self._initial_position,
        )

    def _full_trail_positions(self, current: FloatArray) -> FloatArray:
        view = self.trail.view(self.time_s, mode="full")
        if view.positions_display_m.shape[0] == 0:
            return current.reshape(1, 3)
        return view.positions_display_m

    @staticmethod
    def _smoothed_angle_deg(current_deg: float, target_deg: float, gain: float) -> float:
        difference = (target_deg - current_deg + 180.0) % 360.0 - 180.0
        return float(current_deg + gain * difference)

    def _camera(self, current: FloatArray, telemetry: AircraftTelemetry) -> float:
        positions = self._full_trail_positions(current)
        full_span = max(float(np.max(np.ptp(positions, axis=0))), 500.0)
        if self.camera_mode == "free":
            # Stationary world: fix the limits once, then leave rotation/zoom to
            # the user's mouse. The aircraft translates through the fixed scene.
            if not self._free_limits_set:
                self._apply_world_limits(positions, full_span)
                self._free_limits_set = True
            self._update_ground_grid()
            return full_span
        target_azimuth = 90.0 - telemetry.heading_deg
        self._camera_azimuth_deg = self._smoothed_angle_deg(
            self._camera_azimuth_deg, target_azimuth, 0.18
        )
        if self.camera_mode in {"chase", "cockpit"}:
            if self.camera_mode == "cockpit":
                span = max(180.0, min(3_000.0, 3.0 * telemetry.true_airspeed_mps))
                direction = np.array(
                    [
                        np.sin(np.deg2rad(telemetry.heading_deg)),
                        np.cos(np.deg2rad(telemetry.heading_deg)),
                    ]
                )
                centre_xy = current[:2] + 0.42 * span * direction
                self.scene_axis.set_xlim(centre_xy[0] - span, centre_xy[0] + span)
                self.scene_axis.set_ylim(centre_xy[1] - span, centre_xy[1] + span)
                self.scene_axis.set_zlim(
                    max(-50.0, current[2] - 0.35 * span), current[2] + 0.8 * span
                )
                self.scene_axis.view_init(
                    elev=float(np.clip(telemetry.pitch_deg, -30.0, 45.0)),
                    azim=self._camera_azimuth_deg,
                )
            else:
                span = max(350.0, min(8_000.0, 6.0 * telemetry.true_airspeed_mps))
                self.scene_axis.set_xlim(current[0] - span, current[0] + span)
                self.scene_axis.set_ylim(current[1] - span, current[1] + span)
                self.scene_axis.set_zlim(
                    max(-50.0, current[2] - 0.65 * span), current[2] + span
                )
                self.scene_axis.view_init(elev=22.0, azim=self._camera_azimuth_deg)
            self._update_ground_grid()
            return 2.0 * span
        self._apply_world_limits(positions, full_span)
        self.scene_axis.view_init(
            elev=90.0 if self.camera_mode == "top" else 28.0,
            azim=-90.0 if self.camera_mode == "top" else self._camera_azimuth_deg,
        )
        self._update_ground_grid()
        return full_span

    def _apply_world_limits(self, positions: FloatArray, full_span: float) -> None:
        """Fit the scene limits to the whole flight (stationary-world view)."""
        margin = 0.18 * full_span
        minimum = np.min(positions, axis=0) - margin
        maximum = np.max(positions, axis=0) + margin
        self.scene_axis.set_xlim(minimum[0], maximum[0])
        self.scene_axis.set_ylim(minimum[1], maximum[1])
        self.scene_axis.set_zlim(min(-20.0, minimum[2]), max(100.0, maximum[2]))

    def _update_ground_grid(self) -> None:
        """Re-tile the ground grid to the current view with a stable spacing.

        Regenerating a bounded set of evenly-spaced lines every frame (instead of
        clipping one enormous fixed wireframe) keeps the gridlines steady as the
        view pans/zooms, which removes the flicker/"glitch".
        """
        x_min, x_max = self.scene_axis.get_xlim3d()
        y_min, y_max = self.scene_axis.get_ylim3d()
        span = max(x_max - x_min, y_max - y_min)
        if not np.isfinite(span) or span <= 0.0:
            return
        magnitude = 10.0 ** np.floor(np.log10(span / 10.0))
        spacing = 10.0 * magnitude
        for nice in (1.0, 2.0, 5.0):
            candidate = nice * magnitude
            if span / candidate <= 14.0:
                spacing = candidate
                break
        x0 = np.floor(x_min / spacing) * spacing
        x1 = np.ceil(x_max / spacing) * spacing
        y0 = np.floor(y_min / spacing) * spacing
        y1 = np.ceil(y_max / spacing) * spacing
        segments = [
            [(float(x), y0, 0.0), (float(x), y1, 0.0)]
            for x in np.arange(x0, x1 + 0.5 * spacing, spacing)
        ]
        segments += [
            [(x0, float(y), 0.0), (x1, float(y), 0.0)]
            for y in np.arange(y0, y1 + 0.5 * spacing, spacing)
        ]
        self.ground_grid.set_segments(segments)

    def _update_trail_artist(self) -> FloatArray:
        view = self.trail.view(self.time_s, mode=self.trail_mode)
        positions = view.positions_display_m
        if positions.shape[0] < 2:
            self.trail_collection.set_segments([])
            return positions
        segments = np.stack((positions[:-1], positions[1:]), axis=1)
        self.trail_collection.set_segments(segments)
        if self.trail_settings.color_source == "constant":
            rgba = np.tile(np.asarray([0.10, 0.42, 0.72, 1.0]), (segments.shape[0], 1))
        else:
            values = view.color_values[1:]
            spread = float(np.ptp(values))
            normalized = (
                np.zeros_like(values)
                if spread <= 1.0e-12
                else (values - values.min()) / spread
            )
            rgba = np.asarray(plt.get_cmap("viridis")(normalized), dtype=np.float64)
        rgba[:, 3] = view.alpha[1:]
        self.trail_collection.set_color(rgba)
        return positions

    @staticmethod
    def _set_line_3d(line: Line3D, points: FloatArray, visible: bool) -> None:
        line.set_visible(visible)
        if visible:
            line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])

    def _update_overlays(self, current: FloatArray, view_span_m: float) -> None:
        typed = AircraftState.from_array(self.state, normalize=True)
        display_inertial = self._dcm_display_inertial(self.time_s)
        dcm_display_body = display_inertial @ quaternion_to_dcm(typed.quaternion_ib)
        axis_length = max(25.0, 0.06 * view_span_m)
        for index, line in enumerate(self.body_axis_lines):
            points = np.vstack((current, current + axis_length * dcm_display_body[:, index]))
            self._set_line_3d(line, points, self.overlay_body_axes)
        planet_rate = np.array([0.0, 0.0, self.configuration.planet.rotation_rate_radps])
        ground_velocity = typed.velocity_inertial_mps - np.cross(
            planet_rate, typed.position_inertial_m
        )
        velocity_display = display_inertial @ ground_velocity
        speed = max(float(np.linalg.norm(velocity_display)), 1.0e-12)
        velocity_tip = current + 0.12 * view_span_m * velocity_display / speed
        self._set_line_3d(
            self.velocity_line,
            np.vstack((current, velocity_tip)),
            self.overlay_velocity,
        )
        altitude_m = (
            float(np.linalg.norm(typed.position_inertial_m)) - self.configuration.planet.radius_m
        )
        wind_ned = self.model.wind_model.velocity_ned_mps(
            self.time_s, altitude_m
        ) + self.model.discrete_gust.velocity_ned_mps(self.time_s)
        wind_display = np.asarray([wind_ned[1], wind_ned[0], -wind_ned[2]])
        wind_speed = float(np.linalg.norm(wind_display))
        wind_tip = (
            current
            if wind_speed <= 1.0e-12
            else current + 0.10 * view_span_m * wind_display / wind_speed
        )
        self._set_line_3d(self.wind_line, np.vstack((current, wind_tip)), self.overlay_wind)
        prediction_time_s = min(12.0, max(2.0, 0.012 * view_span_m))
        predicted = current + velocity_display * prediction_time_s
        self._set_line_3d(
            self.prediction_line,
            np.vstack((current, predicted)),
            self.overlay_prediction,
        )

    def _update_history_and_map(self, current: FloatArray) -> None:
        cutoff = max(0.0, self.time_s - 60.0)
        history_time = np.asarray(self._history_time)
        first = int(np.searchsorted(history_time, cutoff))
        history_time = history_time[first:]
        altitude = np.asarray(self._history_altitude)[first:] / 100.0
        airspeed = np.asarray(self._history_airspeed)[first:]
        alpha = np.asarray(self._history_alpha)[first:]
        self.altitude_line.set_data(history_time, altitude)
        self.airspeed_line.set_data(history_time, airspeed)
        self.alpha_line.set_data(history_time, alpha)
        self.history_axis.set_xlim(cutoff, max(cutoff + 5.0, self.time_s))
        history_values = np.concatenate((altitude, airspeed, alpha))
        self.history_axis.set_ylim(
            min(-5.0, float(np.min(history_values)) - 5.0),
            float(np.max(history_values)) + 10.0,
        )
        positions = self._full_trail_positions(current)
        self.map_track_line.set_data(positions[:, 0], positions[:, 1])
        self.map_aircraft_marker.set_data([current[0]], [current[1]])
        east_span = max(250.0, float(np.ptp(positions[:, 0])) * 1.2)
        north_span = max(250.0, float(np.ptp(positions[:, 1])) * 1.2)
        self.map_axis.set_xlim(current[0] - 0.55 * east_span, current[0] + 0.55 * east_span)
        self.map_axis.set_ylim(current[1] - 0.55 * north_span, current[1] + 0.55 * north_span)

    def _update_artists(self, command: AircraftControlCommand) -> tuple[Artist, ...]:
        telemetry = self._snapshot(command)
        turn_rate_degps = 0.0
        if len(self.recorder.records) >= 2:
            previous_record = self.recorder.records[-2]
            latest_record = self.recorder.records[-1]
            heading_change_deg = (
                latest_record.telemetry.heading_deg
                - previous_record.telemetry.heading_deg
                + 180.0
            ) % 360.0 - 180.0
            turn_rate_degps = heading_change_deg / (
                latest_record.time_s - previous_record.time_s
            )
        current = self._display_position(self.state[:3], self.time_s)
        self._update_trail_artist()
        view_span = self._camera(current, telemetry)
        self.aircraft_collection.set_verts(self._mesh_triangles(view_span))
        self.aircraft_collection.set_visible(self.camera_mode != "cockpit")
        self.ground_marker.set_data_3d([current[0]], [current[1]], [0.0])
        self.shadow_line.set_data_3d(
            [current[0], current[0]], [current[1], current[1]], [0.0, current[2]]
        )
        self._update_overlays(current, view_span)
        typed = AircraftState.from_array(self.state, normalize=True)
        lagged = self._last_clock_tick.dropped_simulation_s > 1.0e-9
        warnings = evaluate_flight_warnings(
            telemetry, command, self.envelope_limits, numerical_lag=lagged
        )
        controller = (
            "XInput"
            if self.gamepad is not None and self.gamepad.available
            else "keyboard"
        )
        self.telemetry_text.set_text(
            "FLIGHT HUD  [SI]\n"
            f"T / RTF     {telemetry.time_s:7.1f}s  "
            f"{self._last_clock_tick.achieved_real_time_factor:4.2f}x\n"
            f"FPS / lag   {self._last_clock_tick.measured_fps:7.1f}  "
            f"{self._clock.total_dropped_simulation_s:5.2f}s\n"
            f"TAS / Mach  {telemetry.true_airspeed_mps:7.1f}  {telemetry.mach:5.3f}\n"
            f"ALT / VS    {telemetry.altitude_m:7.0f}  {telemetry.vertical_speed_mps:+6.1f}\n"
            f"HDG / FPA   {telemetry.heading_deg:7.1f}  {telemetry.flight_path_angle_deg:+6.1f}\n"
            f"BANK/PITCH  {telemetry.roll_deg:+7.1f}  {telemetry.pitch_deg:+6.1f}\n"
            f"turn rate   {turn_rate_degps:+7.2f} deg/s\n"
            f"AoA / beta  {telemetry.angle_of_attack_deg:+7.2f}  "
            f"{telemetry.sideslip_angle_deg:+6.2f}\n"
            f"q dynamic   {telemetry.dynamic_pressure_pa:7.0f} Pa\n"
            f"normal nz   {telemetry.normal_load_g:+7.2f} g\n"
            f"1-g Vs ref  {telemetry.stall_speed_1g_mps:7.1f} m/s\n"
            f"stall margin{telemetry.stall_margin_mps:+7.1f} m/s\n"
            f"fuel / mass {100.0 * telemetry.fuel_fraction:6.1f}%  {telemetry.mass_kg:6.0f}kg\n"
            f"THR actual  {100.0 * telemetry.throttle:7.1f}%\n"
            f"A/E/R deg   {np.rad2deg(typed.control_surface_rad[0]):+5.1f} "
            f"{np.rad2deg(typed.control_surface_rad[1]):+5.1f} "
            f"{np.rad2deg(typed.control_surface_rad[2]):+5.1f}\n"
            f"Mode {self.control_profile.control_mode.replace('_', ' ')}\n"
            f"{controller} | {self.camera_mode} | trail {self.trail_mode}"
            f" | demo {'ON' if self.scripted_demo_enabled else 'off'}\n"
            f"Task {self.training_task or 'free flight'}"
        )
        if self.finished_reason:
            annunciation, color = self.finished_reason, RED
        elif warnings:
            annunciation = " | ".join(warning.message for warning in warnings[:2])
            color = RED if any(warning.severity == "warning" for warning in warnings) else ORANGE
        elif self.is_paused:
            annunciation, color = f"PAUSED - {self.pause_reason}", ORANGE
        else:
            annunciation, color = "NORMAL SYNTHETIC MODEL OPERATION", GREEN
        self.warning_text.set_text(annunciation)
        self.warning_text.set_color(color)
        self.start_text.set_visible(
            self.is_paused and self.time_s <= 1.0e-12 and not self.show_help
        )
        self.help_text.set_visible(self.show_help)
        self.help_text.set_text(self._help_message())
        self.status_text.set_text(
            f"H controls | C camera | V trail | F9 save | mesh: {self.mesh_scale_mode}"
            + (f" | {self._last_saved_message}" if self._last_saved_message else "")
        )
        self._update_history_and_map(current)
        return (
            self.trail_collection,
            self.aircraft_collection,
            self.ground_marker,
            self.shadow_line,
            *self.body_axis_lines,
            self.velocity_line,
            self.wind_line,
            self.prediction_line,
            self.telemetry_text,
            self.warning_text,
            self.altitude_line,
            self.airspeed_line,
            self.alpha_line,
            self.map_track_line,
            self.map_aircraft_marker,
            self.start_text,
            self.help_text,
            self.status_text,
        )

    def _record_sample(self, command: AircraftControlCommand, *, force: bool = False) -> None:
        telemetry = self._snapshot(command)
        position = self._display_position(self.state[:3], self.time_s)
        self.trail.append(
            self.time_s,
            position,
            altitude_m=telemetry.altitude_m,
            airspeed_mps=telemetry.true_airspeed_mps,
            force=force,
        )
        self._history_time.append(self.time_s)
        self._history_altitude.append(telemetry.altitude_m)
        self._history_airspeed.append(telemetry.true_airspeed_mps)
        self._history_alpha.append(telemetry.angle_of_attack_deg)
        if not self.recorder.records or self.time_s > self.recorder.records[-1].time_s:
            warnings = evaluate_flight_warnings(telemetry, command, self.envelope_limits)
            warning_codes = {warning.code for warning in warnings}
            for warning in warnings:
                if warning.code not in self._active_warning_codes:
                    self.recorder.add_event(
                        f"warning_{warning.code}_onset",
                        self.time_s,
                        warning.message,
                    )
            for cleared_code in self._active_warning_codes - warning_codes:
                self.recorder.add_event(
                    f"warning_{cleared_code}_cleared",
                    self.time_s,
                )
            self._active_warning_codes = warning_codes
            self.recorder.append(self.state, command, telemetry, warnings)

    def _advance_one_step(
        self,
        command: AircraftControlCommand,
        duration_s: float,
    ) -> None:
        previous_time = self.time_s
        previous_state = self.state.copy()
        remaining_session_s = 3_600.0 - previous_time
        if remaining_session_s <= 1.0e-12:
            self.time_s = 3_600.0
            self.finished_reason = "ONE-HOUR LIVE LIMIT REACHED - Home resets"
            self.is_paused = True
            self.pause_reason = "one-hour safety limit"
            return
        # Cap the final integration interval so the state and timestamp both
        # represent exactly the one-hour boundary (rather than relabelling a
        # state that was propagated beyond it).
        duration_s = min(duration_s, remaining_session_s)
        next_state = advance_live_aircraft(
            self.model, previous_state, previous_time, duration_s, command
        )
        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError("aircraft integrator produced a non-finite state")
        next_time = previous_time + duration_s
        previous_altitude = (
            float(np.linalg.norm(previous_state[:3])) - self.configuration.planet.radius_m
        )
        next_altitude = float(np.linalg.norm(next_state[:3])) - self.configuration.planet.radius_m
        if previous_altitude >= 0.0 and next_altitude <= 0.0:
            self.time_s, self.state = interpolate_ground_contact(
                previous_time,
                previous_state,
                next_time,
                next_state,
                self.configuration,
            )
            contact_telemetry = self._snapshot(command)
            assessment = classify_touchdown(contact_telemetry)
            self.finished_reason = (
                f"SURFACE CONTACT: {assessment.classification.replace('_', ' ').upper()} "
                "(no gear/ground roll) - Home resets"
            )
            self.is_paused = True
            self.pause_reason = "surface contact"
            self.recorder.add_event(
                "surface_contact",
                self.time_s,
                f"{assessment.classification}; vertical_speed_mps="
                f"{assessment.vertical_speed_mps:.3f}",
            )
            self._record_sample(command, force=True)
            return
        self.state = next_state
        self.time_s = next_time
        if self.time_s >= 3_600.0 - 1.0e-10:
            self.time_s = 3_600.0
            self.finished_reason = "ONE-HOUR LIVE LIMIT REACHED - Home resets"
            self.is_paused = True
            self.pause_reason = "one-hour safety limit"
            self.recorder.add_event("session_limit", self.time_s, "one-hour live limit")

    def _update_pilot_inputs(self, duration_s: float) -> None:
        bounded_duration_s = min(max(duration_s, 0.0), 0.15)
        self.virtual_stick.update(self.pressed_keys, bounded_duration_s, self.control_profile)
        bindings = self.control_profile.bindings
        throttle_direction = float(
            (bindings.throttle_up in self.pressed_keys)
            - (bindings.throttle_down in self.pressed_keys)
        )
        self.keyboard_throttle = float(
            np.clip(
                self.keyboard_throttle
                + throttle_direction
                * self.control_profile.throttle_rate_per_s
                * bounded_duration_s,
                0.0,
                1.0,
            )
        )

    def _animation_frame(self, _frame: int) -> tuple[Artist, ...]:
        paused = self.is_paused or bool(self.finished_reason)
        tick = self._clock.tick(paused=paused)
        self._last_clock_tick = tick
        gamepad = self.gamepad.poll() if self.gamepad is not None else GamepadSnapshot(False)
        if not paused:
            self._update_pilot_inputs(tick.wall_delta_s)
        command = self._current_command(gamepad)
        if not paused and tick.dropped_simulation_s > 2.0:
            self._set_paused(True, "excessive numerical lag; press Space to resume")
        elif not paused:
            try:
                for _ in range(tick.physics_step_count):
                    command = self._current_command(gamepad)
                    self._advance_one_step(command, self.configuration.integration_step_s)
                    if self.finished_reason:
                        break
                    if self.time_s + 1.0e-10 >= self._next_log_time_s:
                        self._record_sample(command)
                        self._next_log_time_s += 0.1
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                self.finished_reason = f"NUMERICAL FAILURE: {error} - Home resets"
                self.is_paused = True
                self.pause_reason = "numerical failure"
                self.recorder.add_event("numerical_failure", self.time_s, str(error))
        return self._update_artists(command)

    def _set_paused(self, paused: bool, reason: str = "pilot command") -> None:
        self.is_paused = paused
        self.pause_reason = reason if paused else ""
        self.pressed_keys.clear()
        self.virtual_stick.clear()
        self._clock.resynchronize()

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "key", None)
        if not isinstance(key, str):
            return
        key = key.casefold()
        if key in {"p", " ", "space"}:
            if not self.finished_reason:
                self._set_paused(not self.is_paused)
        elif key == "h":
            self.show_help = not self.show_help
        elif key == "c":
            index = (LIVE_CAMERA_MODES.index(self.camera_mode) + 1) % len(LIVE_CAMERA_MODES)
            self.camera_mode = LIVE_CAMERA_MODES[index]
            self._free_limits_set = False  # re-fit the world if entering free
        elif key == "0":
            self.camera_mode = "chase"
            self._camera_azimuth_deg = -55.0
            self._free_limits_set = False
        elif key == "v":
            index = (TRAIL_MODES.index(self.trail_mode) + 1) % len(TRAIL_MODES)
            self.trail_mode = TRAIL_MODES[index]
        elif key == "x":
            self.trail.clear()
        elif key == "m":
            self.mesh_scale_mode = (
                "true_scale" if self.mesh_scale_mode == "enlarged_marker" else "enlarged_marker"
            )
        elif key == "1":
            self.overlay_body_axes = not self.overlay_body_axes
        elif key == "2":
            self.overlay_velocity = not self.overlay_velocity
        elif key == "3":
            self.overlay_wind = not self.overlay_wind
        elif key == "4":
            self.overlay_prediction = not self.overlay_prediction
        elif key == "tab":
            next_mode: AircraftControlMode = (
                "direct"
                if self.control_profile.control_mode == "stability_assisted"
                else "stability_assisted"
            )
            self.control_profile = replace(self.control_profile, control_mode=next_mode)
        elif key == "home":
            self.reset()
        elif key == "t":
            self.space_assist_start_s = self.time_s if self.space_assist_start_s is None else None
        elif key == "f6":
            self.scripted_demo_enabled = not self.scripted_demo_enabled
            self.space_assist_start_s = None
            self.recorder.add_event(
                "scripted_demo_toggled",
                self.time_s,
                "enabled" if self.scripted_demo_enabled else "disabled",
            )
        elif key == "f9":
            self.save_session()
        elif key == "f10":
            self.open_replay()
        elif key == "g":
            self.export_gif()
        elif key == "f12":
            self.save_screenshot()
        elif key == "escape":
            plt.close(self.figure)
        else:
            self.pressed_keys.add(key)

    def _on_key_release(self, event: object) -> None:
        key = getattr(event, "key", None)
        if isinstance(key, str):
            self.pressed_keys.discard(key.casefold())

    def _on_focus_lost(self, _event: object) -> None:
        self.pressed_keys.clear()
        self.virtual_stick.clear()
        self._clock.resynchronize()

    def _on_close(self, _event: object) -> None:
        self.is_paused = True
        self.pressed_keys.clear()
        self.virtual_stick.clear()
        if self._animation is not None and self._animation.event_source is not None:
            self._animation.event_source.stop()  # type: ignore[no-untyped-call]
        self._animation = None

    def reset(self) -> None:
        """Restore the configured initial condition and open in a safe paused state."""
        self.time_s = 0.0
        self.state = aircraft_initial_state(self.configuration)
        self.keyboard_throttle = self.configuration.initial_throttle
        self.is_paused = True
        self.pause_reason = "reset complete"
        self.finished_reason = ""
        self.space_assist_start_s = None
        self.scripted_demo_enabled = False
        self.pressed_keys.clear()
        self.virtual_stick.reset()
        self.virtual_stick.pitch_trim = self._trim_command
        self.trail = FlightTrailBuffer(self.trail_settings)
        self.recorder.clear()
        self._history_time.clear()
        self._history_altitude.clear()
        self._history_airspeed.clear()
        self._history_alpha.clear()
        self._next_log_time_s = 0.1
        self._camera_azimuth_deg = -55.0
        # In "free" mode the world limits are fixed once so the aircraft visibly
        # translates through a stationary scene and the user owns rotation/zoom.
        self._free_limits_set = False
        self._last_saved_message = ""
        self._active_warning_codes.clear()
        self._clock.resynchronize()
        self._record_sample(self._current_command(GamepadSnapshot(False)), force=True)

    def save_session(self, directory: str | Path | None = None) -> Path:
        """Save exact recorded states plus an automatic debrief on demand."""
        artifacts = self.recorder.write(
            directory or self.recorder_directory,
            termination_reason=self.finished_reason or "session saved by pilot",
        )
        if self.training_task is not None and len(self.recorder.records) >= 2:
            records = tuple(self.recorder.records)
            evaluation = evaluate_training_task(
                self.training_task,
                [record.telemetry for record in records],
                [record.command for record in records],
            )
            training_path = artifacts.csv_path.parent / "aircraft_training_evaluation.json"
            training_path.write_text(
                json.dumps(asdict(evaluation), indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        self._last_saved_message = f"saved {artifacts.csv_path.parent}"
        return artifacts.csv_path

    def save_screenshot(self, path: str | Path | None = None) -> Path:
        """Save a high-resolution photo-mode frame without changing simulation state."""
        output = Path(path or self.recorder_directory / "aircraft_flight_deck.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output, dpi=180, bbox_inches="tight")
        self._last_saved_message = f"screenshot {output.name}"
        return output

    def open_replay(self) -> object | None:
        """Save and open a seekable exact-state replay without re-running the plant."""
        from aerognc.visualisation.aircraft_experience import load_recorded_flight
        from aerognc.visualisation.aircraft_replay import AircraftReplayPlayer

        if len(self.recorder.records) < 2:
            self._last_saved_message = "fly briefly before opening replay"
            return None
        recording_path = self.save_session()
        player = AircraftReplayPlayer(
            self.configuration,
            self.mesh,
            load_recorded_flight(recording_path),
        )
        self._replay_players.append(player)

        def release_player(_event: object, replay_player: object = player) -> None:
            if replay_player in self._replay_players:
                self._replay_players.remove(replay_player)

        player.figure.canvas.mpl_connect("close_event", release_player)
        player.show(block=False)
        return player

    def export_gif(self, path: str | Path | None = None) -> Path | None:
        """Export a bounded GIF from exact recorded states, never a dynamics rerun."""
        from aerognc.visualisation.aircraft_experience import load_recorded_flight
        from aerognc.visualisation.aircraft_replay import AircraftReplayPlayer

        if len(self.recorder.records) < 2:
            self._last_saved_message = "fly briefly before exporting a GIF"
            return None
        recording_path = self.save_session()
        player = AircraftReplayPlayer(
            self.configuration,
            self.mesh,
            load_recorded_flight(recording_path),
        )
        try:
            output = player.export_gif(
                path or self.recorder_directory / "aircraft_flight_replay.gif"
            )
        finally:
            plt.close(player.figure)
        self._last_saved_message = f"GIF {output.name}"
        return output

    def show(self, *, block: bool = True) -> None:
        """Open the live window; physics remains fixed-step regardless of display FPS."""
        self._animation = FuncAnimation(
            self.figure,
            self._animation_frame,
            interval=1_000.0 / self.frames_per_second,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=block)


def play_aircraft_live(
    configuration: AircraftSandboxConfiguration,
    mesh_path: str | Path = "assets/models/aquila_x1.obj",
    *,
    axis_convention: Literal["body_frd", "x_forward_z_up", "y_forward_z_up"] = "body_frd",
    frames_per_second: int = 30,
    real_time_factor: float = 1.0,
    camera_mode: LiveCameraMode = "chase",
    enable_gamepad: bool = True,
    control_profile: PilotControlProfile | None = None,
    trail_settings: TrailSettings | None = None,
    mesh_scale_mode: MeshScaleMode = "enlarged_marker",
    recorder_directory: str | Path | None = None,
    scripted_demo: bool = False,
    training_task: TrainingTask | None = None,
) -> None:
    """Load a visual mesh and open the live coefficient-driven aircraft simulator."""
    mesh = load_triangle_mesh(mesh_path, axis_convention=axis_convention)
    player = AircraftLivePlayer(
        configuration,
        mesh,
        frames_per_second=frames_per_second,
        real_time_factor=real_time_factor,
        camera_mode=camera_mode,
        enable_gamepad=enable_gamepad,
        control_profile=control_profile,
        trail_settings=trail_settings,
        mesh_scale_mode=mesh_scale_mode,
        recorder_directory=recorder_directory,
        training_task=training_task,
    )
    player.scripted_demo_enabled = scripted_demo
    player.show()
