"""Interactive map-based waypoint mission planner (Tk).

Two layers, so the logic is testable without a display:

* :class:`PlannerModel` — a pure, Tk-free model: the home and waypoint list,
  geodetic<->pixel projection (via :class:`~aerognc.mathematics.local_frame.LocalTangentFrame`),
  add / move / delete / duplicate / reorder / clear edits, undo/redo, import/export,
  and building a validated :class:`~aerognc.mission.mission.Mission`.
* :class:`InteractiveMissionPlanner` — a Tk view binding a canvas (left-click add,
  drag to move, right-click menu, double-click set-home, wheel zoom) and a property
  panel to the model, drawing home, numbered waypoints, the planned route,
  acceptance-radius and loiter circles, a geofence, and — after a run — the actual
  trajectory. Fields use an explicit dark theme so text is never white-on-white.

Launch with ``python -m aerognc.cli mission-planner`` (optionally ``--mission``).
The planner runs missions on the internal simulator only; it commands no hardware.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aerognc.mathematics.geodesy import GeodeticPosition
from aerognc.mathematics.local_frame import WGS84, LocalTangentFrame
from aerognc.mission.mission import HomePosition, Mission, MissionDefaults, MissionLimits
from aerognc.mission.mission_io import load_mission, mission_to_dict, save_mission
from aerognc.mission.waypoint import AltitudeReference, LoiterDirection, Waypoint, WaypointAction

# Dark theme (shared with the workbench palette).
BACKGROUND = "#07111F"
PANEL = "#0D1B2A"
FIELD = "#0A1724"
TEXT = "#D9E8F2"
MUTED = "#8FA6B8"
CYAN = "#39C6E8"
GREEN = "#5FD19A"
AMBER = "#F2B84B"
RED = "#E06C6C"
ROUTE = "#4C86C6"


@dataclass
class PlannerWaypoint:
    """An editable waypoint held by the planner (geodetic, degrees)."""

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 120.0
    airspeed_mps: float | None = None
    acceptance_radius_m: float | None = None
    altitude_tolerance_m: float | None = None
    action: WaypointAction = WaypointAction.FLY_THROUGH
    loiter_radius_m: float | None = None
    loiter_duration_s: float | None = None
    loiter_direction: LoiterDirection = LoiterDirection.CLOCKWISE

    def to_waypoint(self, identifier: int) -> Waypoint:
        return Waypoint(
            id=identifier,
            name=self.name,
            latitude_deg=self.latitude_deg,
            longitude_deg=self.longitude_deg,
            altitude_m=self.altitude_m,
            altitude_reference=AltitudeReference.RELATIVE_HOME,
            airspeed_mps=self.airspeed_mps,
            acceptance_radius_m=self.acceptance_radius_m,
            altitude_tolerance_m=self.altitude_tolerance_m,
            action=self.action,
            loiter_radius_m=self.loiter_radius_m,
            loiter_duration_s=self.loiter_duration_s,
            loiter_direction=self.loiter_direction,
        )


@dataclass
class _Snapshot:
    home: HomePosition
    waypoints: list[PlannerWaypoint]


class PlannerModel:
    """Tk-free mission-planning model with projection and undo/redo."""

    def __init__(
        self,
        home: HomePosition | None = None,
        *,
        canvas_width_px: int = 900,
        canvas_height_px: int = 640,
        meters_per_pixel: float = 3.0,
    ) -> None:
        self.home = home or HomePosition(39.925, 32.8369, 0.0)
        self.waypoints: list[PlannerWaypoint] = []
        self.defaults = MissionDefaults()
        self.limits = MissionLimits()
        self.geofence_radius_m: float | None = None
        self.center_px = (canvas_width_px / 2.0, canvas_height_px / 2.0)
        self.meters_per_pixel = meters_per_pixel
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []

    # -- projection -----------------------------------------------------------
    def _frame(self) -> LocalTangentFrame:
        return LocalTangentFrame(origin=self.home.geodetic(), ellipsoid=WGS84)

    def geo_to_pixel(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
        """Project a geodetic point to canvas pixels (north up)."""
        ned = self._frame().geodetic_to_ned(
            GeodeticPosition(np.deg2rad(latitude_deg), np.deg2rad(longitude_deg), 0.0)
        )
        px = self.center_px[0] + ned[1] / self.meters_per_pixel
        py = self.center_px[1] - ned[0] / self.meters_per_pixel
        return float(px), float(py)

    def pixel_to_geo(self, px: float, py: float) -> tuple[float, float]:
        """Inverse-project canvas pixels to (latitude_deg, longitude_deg)."""
        north_m = (self.center_px[1] - py) * self.meters_per_pixel
        east_m = (px - self.center_px[0]) * self.meters_per_pixel
        geodetic = self._frame().ned_to_geodetic(np.array([north_m, east_m, 0.0]))
        return float(np.rad2deg(geodetic.latitude_rad)), float(np.rad2deg(geodetic.longitude_rad))

    def meters_to_pixels(self, meters: float) -> float:
        return meters / self.meters_per_pixel

    def zoom(self, factor: float, *, anchor_px: tuple[float, float] | None = None) -> None:
        """Zoom in/out, keeping the anchor pixel over the same ground point."""
        if factor <= 0.0:
            raise ValueError("zoom factor must be positive")
        if anchor_px is None:
            self.meters_per_pixel = float(np.clip(self.meters_per_pixel / factor, 0.05, 500.0))
            return
        lat, lon = self.pixel_to_geo(*anchor_px)
        self.meters_per_pixel = float(np.clip(self.meters_per_pixel / factor, 0.05, 500.0))
        new_px, new_py = self.geo_to_pixel(lat, lon)
        self.center_px = (
            self.center_px[0] + (anchor_px[0] - new_px),
            self.center_px[1] + (anchor_px[1] - new_py),
        )

    def pan_pixels(self, dx: float, dy: float) -> None:
        self.center_px = (self.center_px[0] + dx, self.center_px[1] + dy)

    # -- edits (each records undo) -------------------------------------------
    def _checkpoint(self) -> None:
        self._undo.append(_Snapshot(self.home, copy.deepcopy(self.waypoints)))
        self._redo.clear()

    def add_waypoint_geo(self, latitude_deg: float, longitude_deg: float) -> PlannerWaypoint:
        self._checkpoint()
        waypoint = PlannerWaypoint(
            name=f"WP{len(self.waypoints) + 1}",
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_m=self.defaults_altitude_m(),
        )
        self.waypoints.append(waypoint)
        return waypoint

    def defaults_altitude_m(self) -> float:
        return self.waypoints[-1].altitude_m if self.waypoints else 120.0

    def move_waypoint_geo(self, index: int, latitude_deg: float, longitude_deg: float) -> None:
        self._require_index(index)
        self._checkpoint()
        self.waypoints[index].latitude_deg = latitude_deg
        self.waypoints[index].longitude_deg = longitude_deg

    def delete_waypoint(self, index: int) -> None:
        self._require_index(index)
        self._checkpoint()
        del self.waypoints[index]

    def duplicate_waypoint(self, index: int) -> None:
        self._require_index(index)
        self._checkpoint()
        clone = copy.deepcopy(self.waypoints[index])
        clone.name = f"{clone.name}_copy"
        self.waypoints.insert(index + 1, clone)

    def reorder_waypoint(self, index: int, new_index: int) -> None:
        self._require_index(index)
        new_index = int(np.clip(new_index, 0, len(self.waypoints) - 1))
        if new_index == index:
            return
        self._checkpoint()
        self.waypoints.insert(new_index, self.waypoints.pop(index))

    def clear(self) -> None:
        if not self.waypoints:
            return
        self._checkpoint()
        self.waypoints.clear()

    def set_home_geo(self, latitude_deg: float, longitude_deg: float) -> None:
        self._checkpoint()
        self.home = HomePosition(latitude_deg, longitude_deg, self.home.altitude_m)

    def set_geofence_radius_m(self, radius_m: float | None) -> None:
        self.geofence_radius_m = radius_m

    def update_waypoint_fields(self, index: int, **fields: Any) -> None:
        """Update editable fields on a waypoint (validated on mission build)."""
        self._require_index(index)
        self._checkpoint()
        waypoint = self.waypoints[index]
        for key, value in fields.items():
            if not hasattr(waypoint, key):
                raise ValueError(f"unknown waypoint field: {key}")
            setattr(waypoint, key, value)

    # -- undo/redo ------------------------------------------------------------
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(_Snapshot(self.home, copy.deepcopy(self.waypoints)))
        snapshot = self._undo.pop()
        self.home = snapshot.home
        self.waypoints = snapshot.waypoints

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(_Snapshot(self.home, copy.deepcopy(self.waypoints)))
        snapshot = self._redo.pop()
        self.home = snapshot.home
        self.waypoints = snapshot.waypoints

    # -- mission build / IO ---------------------------------------------------
    def build_mission(self, name: str = "planner_mission") -> Mission:
        """Assemble a validated :class:`Mission` from the current plan."""
        waypoints = tuple(
            waypoint.to_waypoint(index + 1) for index, waypoint in enumerate(self.waypoints)
        )
        mission = Mission(
            name=name,
            home=self.home,
            waypoints=waypoints,
            defaults=self.defaults,
            limits=self.limits,
        )
        return mission

    def validation_issues(self) -> list[str]:
        try:
            return self.build_mission().validation_issues()
        except ValueError as error:
            return [str(error)]

    def export_mission(self, path: str, name: str = "planner_mission") -> None:
        save_mission(self.build_mission(name).validate(), path)

    def load_from_mission(self, mission: Mission) -> None:
        self._checkpoint()
        self.home = mission.home
        self.waypoints = [
            PlannerWaypoint(
                name=w.name,
                latitude_deg=w.latitude_deg,
                longitude_deg=w.longitude_deg,
                altitude_m=w.altitude_m,
                airspeed_mps=w.airspeed_mps,
                acceptance_radius_m=w.acceptance_radius_m,
                altitude_tolerance_m=w.altitude_tolerance_m,
                action=w.action,
                loiter_radius_m=w.loiter_radius_m,
                loiter_duration_s=w.loiter_duration_s,
                loiter_direction=w.loiter_direction,
            )
            for w in mission.waypoints
        ]
        self.defaults = mission.defaults
        self.limits = mission.limits

    def import_mission(self, path: str) -> None:
        self.load_from_mission(load_mission(path))

    def as_dict(self) -> dict[str, Any]:
        return mission_to_dict(self.build_mission())

    # -- helpers --------------------------------------------------------------
    def _require_index(self, index: int) -> None:
        if not 0 <= index < len(self.waypoints):
            raise IndexError(f"waypoint index {index} out of range")

    def nearest_waypoint(self, px: float, py: float, radius_px: float = 12.0) -> int | None:
        """Return the index of the waypoint nearest the pixel, if within radius."""
        best_index: int | None = None
        best_distance = radius_px
        for index, waypoint in enumerate(self.waypoints):
            wx, wy = self.geo_to_pixel(waypoint.latitude_deg, waypoint.longitude_deg)
            distance = float(np.hypot(wx - px, wy - py))
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index


# ---------------------------------------------------------------------------
# Tk view
# ---------------------------------------------------------------------------

_ACTIONS = [action.value for action in WaypointAction]


class PlaybackController:
    """Tk-free playback cursor over a flown mission's samples.

    Advances an index through the recorded samples at a frame multiplier so the
    view can animate the aircraft flying the mission. Pure and unit-testable.
    """

    def __init__(self, samples: Any, *, speed: float = 8.0) -> None:
        self._samples = list(samples)
        self.speed = max(1.0, float(speed))
        self.index = 0
        self.playing = False

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def finished(self) -> bool:
        return not self._samples or self.index >= len(self._samples) - 1

    def reset(self) -> None:
        self.index = 0
        self.playing = False

    def current(self) -> Any:
        if not self._samples:
            return None
        return self._samples[min(self.index, len(self._samples) - 1)]

    def advance(self, frames: int) -> Any:
        """Advance by ``frames`` samples (clamped to the end) and return the sample."""
        if self._samples:
            self.index = min(self.index + max(1, int(frames)), len(self._samples) - 1)
        return self.current()

    def progress(self) -> float:
        if len(self._samples) <= 1:
            return 1.0
        return self.index / (len(self._samples) - 1)


@dataclass
class _ViewState:
    selected_index: int | None = None
    dragging: bool = False
    actual_track_px: list[tuple[float, float]] = field(default_factory=list)
    status: str = "Left-click to add a waypoint. Double-click to set home."


class InteractiveMissionPlanner:
    """Tk map-based waypoint mission planner bound to a :class:`PlannerModel`."""

    def __init__(self, master: Any, model: PlannerModel | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self.model = model or PlannerModel()
        self.view = _ViewState()
        self._result: Any = None
        self._playback: PlaybackController | None = None
        self._animation_job: Any = None
        self.master = master
        master.title("AeroGNC-Lab - Waypoint Mission Planner")
        master.configure(background=BACKGROUND)
        self._configure_style(ttk)

        container = ttk.Frame(master, style="TFrame")
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            container, width=900, height=640, background=FIELD, highlightthickness=0
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(container, style="TFrame", padding=8)
        side.pack(side="right", fill="y")

        self._build_side_panel(ttk, side)
        self._bind_canvas()
        self.redraw()

    # -- styling --------------------------------------------------------------
    def _configure_style(self, ttk: Any) -> None:
        style = ttk.Style(self.master)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("TButton", padding=(8, 5))
        style.configure(
            "TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT
        )
        style.configure(
            "TCombobox", fieldbackground=FIELD, foreground=TEXT, background=PANEL, arrowcolor=TEXT
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", FIELD)],
            foreground=[("readonly", TEXT)],
        )
        for option, value in (
            ("*Entry.background", FIELD),
            ("*Entry.foreground", TEXT),
            ("*Entry.insertBackground", TEXT),
            ("*Listbox.background", FIELD),
            ("*Listbox.foreground", TEXT),
            ("*Listbox.selectBackground", "#21506D"),
            ("*Listbox.selectForeground", "#FFFFFF"),
        ):
            self.master.option_add(option, value)

    # -- side panel -----------------------------------------------------------
    def _build_side_panel(self, ttk: Any, side: Any) -> None:
        tk = self._tk
        ttk.Label(side, text="Waypoints", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self.listbox = tk.Listbox(side, height=10, width=28, exportselection=False)
        self.listbox.pack(fill="x", pady=(2, 6))
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        buttons = ttk.Frame(side, style="TFrame")
        buttons.pack(fill="x")
        for label, command in (
            ("Delete", self._delete_selected),
            ("Duplicate", self._duplicate_selected),
            ("Up", lambda: self._move_selected(-1)),
            ("Down", lambda: self._move_selected(1)),
            ("Clear", self._clear),
            ("Undo", self._undo),
            ("Redo", self._redo),
        ):
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=1)

        ttk.Label(side, text="Selected waypoint", font=("Segoe UI Semibold", 11)).pack(
            anchor="w", pady=(10, 2)
        )
        self._fields: dict[str, Any] = {}
        for key, label in (
            ("altitude_m", "Altitude [m]"),
            ("airspeed_mps", "Airspeed [m/s]"),
            ("acceptance_radius_m", "Accept radius [m]"),
            ("loiter_radius_m", "Loiter radius [m]"),
            ("loiter_duration_s", "Loiter time [s]"),
        ):
            row = ttk.Frame(side, style="TFrame")
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=16).pack(side="left")
            var = tk.StringVar()
            entry = ttk.Entry(row, textvariable=var, width=10)
            entry.pack(side="left")
            entry.bind("<Return>", self._apply_fields)
            entry.bind("<FocusOut>", self._apply_fields)
            self._fields[key] = var

        action_row = ttk.Frame(side, style="TFrame")
        action_row.pack(fill="x", pady=1)
        ttk.Label(action_row, text="Action", width=16).pack(side="left")
        self._action_var = tk.StringVar(value=WaypointAction.FLY_THROUGH.value)
        action_box = ttk.Combobox(
            action_row, textvariable=self._action_var, values=_ACTIONS, state="readonly", width=12
        )
        action_box.pack(side="left")
        action_box.bind("<<ComboboxSelected>>", self._apply_fields)

        wind_row = ttk.Frame(side, style="TFrame")
        wind_row.pack(fill="x", pady=(8, 1))
        ttk.Label(wind_row, text="Wind N/E [m/s]", width=16).pack(side="left")
        self._wind_north = tk.StringVar(value="0")
        self._wind_east = tk.StringVar(value="0")
        ttk.Entry(wind_row, textvariable=self._wind_north, width=5).pack(side="left")
        ttk.Entry(wind_row, textvariable=self._wind_east, width=5).pack(side="left", padx=(2, 0))

        ttk.Label(side, text="Mission", font=("Segoe UI Semibold", 11)).pack(
            anchor="w", pady=(10, 2)
        )
        mission_buttons = ttk.Frame(side, style="TFrame")
        mission_buttons.pack(fill="x")
        for label, command in (
            ("Import", self._import),
            ("Export", self._export),
            ("Validate", self._validate),
            ("Simulate", self._run),
        ):
            ttk.Button(mission_buttons, text=label, command=command).pack(side="left", padx=1)

        ttk.Label(side, text="Playback", font=("Segoe UI Semibold", 11)).pack(
            anchor="w", pady=(10, 2)
        )
        playback_buttons = ttk.Frame(side, style="TFrame")
        playback_buttons.pack(fill="x")
        for label, command in (
            ("Play", self._play),
            ("Pause", self._pause),
            ("Reset", self._reset_playback),
            ("3D plot", self._save_3d_plot),
        ):
            ttk.Button(playback_buttons, text=label, command=command).pack(side="left", padx=1)

        self.hud_var = tk.StringVar(value="")
        ttk.Label(
            side, textvariable=self.hud_var, style="Muted.TLabel", font=("Consolas", 9)
        ).pack(anchor="w", pady=(6, 0))

        self.status_var = tk.StringVar(value=self.view.status)
        ttk.Label(side, textvariable=self.status_var, style="Muted.TLabel", wraplength=240).pack(
            anchor="w", pady=(8, 0)
        )

    # -- canvas bindings ------------------------------------------------------
    def _bind_canvas(self) -> None:
        self.canvas.bind("<Button-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        self.canvas.bind("<Button-3>", self._on_right)
        self.canvas.bind("<MouseWheel>", self._on_wheel)

    def _on_left_press(self, event: Any) -> None:
        index = self.model.nearest_waypoint(event.x, event.y)
        if index is None:
            latitude, longitude = self.model.pixel_to_geo(event.x, event.y)
            self.model.add_waypoint_geo(latitude, longitude)
            self.view.selected_index = len(self.model.waypoints) - 1
        else:
            self.view.selected_index = index
            self.view.dragging = True
        self.view.actual_track_px.clear()
        self._refresh()

    def _on_drag(self, event: Any) -> None:
        if self.view.dragging and self.view.selected_index is not None:
            latitude, longitude = self.model.pixel_to_geo(event.x, event.y)
            self.model.move_waypoint_geo(self.view.selected_index, latitude, longitude)
            self.redraw()

    def _on_release(self, event: Any) -> None:
        self.view.dragging = False

    def _on_double(self, event: Any) -> None:
        latitude, longitude = self.model.pixel_to_geo(event.x, event.y)
        self.model.set_home_geo(latitude, longitude)
        self._set_status("Home set.")
        self._refresh()

    def _on_right(self, event: Any) -> None:
        index = self.model.nearest_waypoint(event.x, event.y)
        if index is None:
            return
        self.view.selected_index = index
        menu = self._tk.Menu(self.master, tearoff=0)
        menu.add_command(label="Delete", command=self._delete_selected)
        menu.add_command(label="Duplicate", command=self._duplicate_selected)
        menu.add_command(label="Set loiter", command=lambda: self._set_action("loiter"))
        menu.add_command(label="Set return home", command=lambda: self._set_action("return_home"))
        menu.tk_popup(event.x_root, event.y_root)

    def _on_wheel(self, event: Any) -> None:
        factor = 1.2 if event.delta > 0 else 1.0 / 1.2
        self.model.zoom(factor, anchor_px=(event.x, event.y))
        self.redraw()

    # -- side-panel actions ---------------------------------------------------
    def _on_list_select(self, _event: Any) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.view.selected_index = int(selection[0])
            self._load_fields()
            self.redraw()

    def _delete_selected(self) -> None:
        if self.view.selected_index is not None:
            self.model.delete_waypoint(self.view.selected_index)
            self.view.selected_index = None
            self._refresh()

    def _duplicate_selected(self) -> None:
        if self.view.selected_index is not None:
            self.model.duplicate_waypoint(self.view.selected_index)
            self._refresh()

    def _move_selected(self, delta: int) -> None:
        if self.view.selected_index is not None:
            new_index = self.view.selected_index + delta
            self.model.reorder_waypoint(self.view.selected_index, new_index)
            self.view.selected_index = int(
                np.clip(new_index, 0, len(self.model.waypoints) - 1)
            )
            self._refresh()

    def _clear(self) -> None:
        self.model.clear()
        self.view.selected_index = None
        self.view.actual_track_px.clear()
        self._refresh()

    def _undo(self) -> None:
        self.model.undo()
        self._refresh()

    def _redo(self) -> None:
        self.model.redo()
        self._refresh()

    def _set_action(self, action_value: str) -> None:
        self._action_var.set(action_value)
        self._apply_fields()

    def _apply_fields(self, _event: Any = None) -> None:
        if self.view.selected_index is None:
            return
        updates: dict[str, Any] = {"action": WaypointAction(self._action_var.get())}
        for key, var in self._fields.items():
            text = var.get().strip()
            if key == "altitude_m":
                updates[key] = float(text) if text else 120.0
            else:
                updates[key] = float(text) if text else None
        self.model.update_waypoint_fields(self.view.selected_index, **updates)
        self._refresh()

    def _load_fields(self) -> None:
        if self.view.selected_index is None:
            return
        waypoint = self.model.waypoints[self.view.selected_index]
        self._action_var.set(waypoint.action.value)
        for key, var in self._fields.items():
            value = getattr(waypoint, key)
            var.set("" if value is None else f"{value:g}")

    def _import(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("Mission YAML", "*.yaml *.yml")])
        if path:
            self.model.import_mission(path)
            self._set_status(f"Imported {path}")
            self._refresh()

    def _export(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.asksaveasfilename(defaultextension=".yaml")
        if not path:
            return
        try:
            self.model.export_mission(path)
            self._set_status(f"Exported {path}")
        except ValueError as error:
            messagebox.showerror("Export failed", str(error))

    def _validate(self) -> None:
        issues = self.model.validation_issues()
        self._set_status("Mission valid." if not issues else "Invalid:\n" + "\n".join(issues))

    def _run(self) -> None:
        from aerognc.simulation.waypoint_mission import (
            WaypointMissionConfig,
            run_waypoint_mission,
        )

        issues = self.model.validation_issues()
        if issues:
            self._set_status("Cannot run - invalid:\n" + "\n".join(issues))
            return
        self._stop_animation()
        self._set_status("Simulating mission...")
        self.master.update_idletasks()
        wind = (self._wind_value("n"), self._wind_value("e"), 0.0)
        config = WaypointMissionConfig(wind_ned_mps=wind)
        result = run_waypoint_mission(self.model.build_mission(), config)
        self._result = result
        self._playback = PlaybackController(result.samples)
        self.view.actual_track_px = [self._sample_pixel(s) for s in result.samples]
        final_xte = result.summary().get("final_cross_track_m")
        self._set_status(
            f"Simulated: {result.outcome}, final cross-track {final_xte} m. Press Play."
        )
        self._update_hud(self._playback.current())
        self.redraw()

    def _wind_value(self, axis: str) -> float:
        var = self._wind_north if axis == "n" else self._wind_east
        try:
            return float(var.get().strip() or 0.0)
        except ValueError:
            return 0.0

    def _sample_pixel(self, sample: Any) -> tuple[float, float]:
        """Project a mission sample (home NED frame) to canvas pixels."""
        px = self.model.center_px[0] + sample.east_m / self.model.meters_per_pixel
        py = self.model.center_px[1] - sample.north_m / self.model.meters_per_pixel
        return px, py

    # -- playback -------------------------------------------------------------
    def _play(self) -> None:
        if self._playback is None:
            self._set_status("Run a simulation first (Simulate).")
            return
        if self._playback.finished:
            self._playback.reset()
        self._playback.playing = True
        self._animate()

    def _pause(self) -> None:
        if self._playback is not None:
            self._playback.playing = False
        self._stop_animation()

    def _reset_playback(self) -> None:
        self._stop_animation()
        if self._playback is not None:
            self._playback.reset()
            self._update_hud(self._playback.current())
        self.redraw()

    def _stop_animation(self) -> None:
        if self._animation_job is not None:
            self.master.after_cancel(self._animation_job)
            self._animation_job = None

    def _animate(self) -> None:
        if self._playback is None or not self._playback.playing:
            return
        sample = self._playback.advance(int(self._playback.speed))
        self._update_hud(sample)
        self._draw_aircraft_glyph(sample)
        if self._playback.finished:
            self._playback.playing = False
            self._set_status(f"Playback complete ({self._result.outcome}).")
            return
        self._animation_job = self.master.after(40, self._animate)

    def _update_hud(self, sample: Any) -> None:
        if sample is None:
            self.hud_var.set("")
            return
        self.hud_var.set(
            f"t {sample.time_s:6.1f}s  alt {sample.altitude_m:6.1f}m\n"
            f"TAS {sample.airspeed_mps:5.1f}  GS {sample.groundspeed_mps:5.1f} m/s\n"
            f"WP {sample.active_waypoint_id}  {sample.mission_state}\n"
            f"XTE {sample.cross_track_error_m:+6.1f}m"
        )

    def _draw_aircraft_glyph(self, sample: Any) -> None:
        """Draw the moving aircraft marker + heading tick (tagged for redraw)."""
        self.canvas.delete("aircraft")
        if sample is None:
            return
        px, py = self._sample_pixel(sample)
        heading = sample.yaw_rad
        nose = (px + 14.0 * np.sin(heading), py - 14.0 * np.cos(heading))
        self.canvas.create_line(px, py, nose[0], nose[1], fill=CYAN, width=2, tags="aircraft")
        self.canvas.create_oval(
            px - 5, py - 5, px + 5, py + 5, fill=CYAN, outline=BACKGROUND, tags="aircraft"
        )

    def _save_3d_plot(self) -> None:
        if self._result is None:
            self._set_status("Run a simulation first (Simulate).")
            return
        from tkinter import filedialog

        from aerognc.visualisation.waypoint_mission import plot_waypoint_mission

        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="mission_3d.png")
        if not path:
            return
        plot_waypoint_mission(self._result, path)
        self._set_status(f"Saved 3D dashboard to {path}")
        try:  # best-effort open on Windows
            import os

            os.startfile(path)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass

    # -- rendering ------------------------------------------------------------
    def _refresh(self) -> None:
        self._sync_listbox()
        self._load_fields()
        self.redraw()

    def _sync_listbox(self) -> None:
        self.listbox.delete(0, self._tk.END)
        for index, waypoint in enumerate(self.model.waypoints):
            self.listbox.insert(
                self._tk.END, f"{index + 1}. {waypoint.name} [{waypoint.action.value}]"
            )
        if self.view.selected_index is not None and self.model.waypoints:
            self.listbox.selection_clear(0, self._tk.END)
            self.listbox.selection_set(self.view.selected_index)

    def _set_status(self, text: str) -> None:
        self.view.status = text
        self.status_var.set(text)

    def redraw(self) -> None:
        """Redraw the whole map (planned route, markers, circles, actual track)."""
        canvas = self.canvas
        canvas.delete("all")
        model = self.model

        home_px = model.geo_to_pixel(model.home.latitude_deg, model.home.longitude_deg)
        if model.geofence_radius_m is not None:
            radius = model.meters_to_pixels(model.geofence_radius_m)
            canvas.create_oval(
                home_px[0] - radius, home_px[1] - radius, home_px[0] + radius, home_px[1] + radius,
                outline=MUTED, dash=(4, 3),
            )

        points = [home_px] + [
            model.geo_to_pixel(w.latitude_deg, w.longitude_deg) for w in model.waypoints
        ]
        if len(points) >= 2:
            flat = [coord for point in points for coord in point]
            canvas.create_line(*flat, fill=ROUTE, width=2)

        if self.view.actual_track_px and len(self.view.actual_track_px) >= 2:
            flat = [coord for point in self.view.actual_track_px for coord in point]
            canvas.create_line(*flat, fill=GREEN, width=1)

        self._draw_marker(home_px, "H", CYAN)
        for index, waypoint in enumerate(model.waypoints):
            px = model.geo_to_pixel(waypoint.latitude_deg, waypoint.longitude_deg)
            accept_m = waypoint.acceptance_radius_m or model.defaults.acceptance_radius_m
            radius = model.meters_to_pixels(accept_m)
            canvas.create_oval(
                px[0] - radius, px[1] - radius, px[0] + radius, px[1] + radius,
                outline=MUTED,
            )
            if waypoint.loiter_radius_m:
                loiter = model.meters_to_pixels(waypoint.loiter_radius_m)
                canvas.create_oval(
                    px[0] - loiter, px[1] - loiter, px[0] + loiter, px[1] + loiter,
                    outline=AMBER, dash=(3, 2),
                )
            selected = index == self.view.selected_index
            self._draw_marker(px, str(index + 1), AMBER if selected else ROUTE)

    def _draw_marker(self, px: tuple[float, float], label: str, color: str) -> None:
        x, y = px
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill=color, outline=BACKGROUND)
        self.canvas.create_text(x, y, text=label, fill=BACKGROUND, font=("Segoe UI Semibold", 8))


def launch_mission_planner(mission_path: str | None = None) -> None:  # pragma: no cover - UI entry
    """Open the interactive planner window (blocks until closed)."""
    import tkinter as tk

    root = tk.Tk()
    model = PlannerModel()
    if mission_path:
        model.import_mission(mission_path)
    planner = InteractiveMissionPlanner(root, model)
    planner._refresh()
    root.mainloop()
