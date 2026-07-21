"""Beginner-friendly Tk desktop front end for interplanetary mission design."""

from __future__ import annotations

import importlib.util
import threading
import tkinter as tk
from collections.abc import Callable, Mapping
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TypeVar, cast

import numpy as np
import yaml

from aerognc.astrodynamics.maneuvers import ImpulsiveManeuver, ManeuverFrame
from aerognc.astrodynamics.mission_design import (
    compute_porkchop_grid,
    design_gravity_assist,
    evaluate_lambert_transfer,
)
from aerognc.configuration.interplanetary_loader import load_interplanetary_configuration
from aerognc.configuration.planetary_catalog import PlanetaryCatalog, load_planetary_catalog
from aerognc.simulation.interplanetary import InterplanetaryMission, simulate_interplanetary
from aerognc.simulation.mission_planner import (
    MissionMethod,
    MissionPlanRequest,
    PlannedMission,
    plan_mission,
)
from aerognc.simulation.mission_uncertainty import UncertaintySummary, run_seeded_uncertainty

T = TypeVar("T")

DIRECT_LABEL = "Direct Lambert transfer"
ASSIST_LABEL = "One gravity assist (preliminary)"


class MissionDesignerApp:
    """Guided mission inputs, numerical design actions, and 3D playback launcher."""

    def __init__(
        self,
        root: tk.Tk,
        catalog: PlanetaryCatalog,
        verified_configuration_path: Path,
    ) -> None:
        self.root = root
        self.catalog = catalog
        self.verified_configuration_path = verified_configuration_path
        self.current_plan: PlannedMission | None = None
        self.current_mission: InterplanetaryMission | None = None
        self.maneuvers: list[ImpulsiveManeuver] = []
        self.busy = False
        self._configure_window()
        self._create_variables()
        self._build_layout()
        self._load_example_values()

    def _configure_window(self) -> None:
        self.root.title("AeroGNC-Lab — Civilian Interplanetary Mission Designer")
        self.root.geometry("1180x800")
        self.root.minsize(980, 690)
        self.root.configure(background="#07111F")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background="#0D1B2A", foreground="#D9E8F2")
        style.configure("TFrame", background="#0D1B2A")
        style.configure("Card.TFrame", background="#102335", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#0D1B2A", foreground="#D9E8F2")
        style.configure("Muted.TLabel", foreground="#8FA6B8")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20), foreground="#F1F7FA")
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 12), foreground="#39C6E8")
        style.configure("Safety.TLabel", foreground="#5FD19A")
        style.configure("TButton", padding=(10, 7))
        style.configure(
            "Primary.TButton",
            font=("Segoe UI Semibold", 10),
            foreground="#07111F",
            background="#39C6E8",
        )
        style.map("Primary.TButton", background=[("active", "#6DDAF2"), ("disabled", "#49657A")])
        style.configure(
            "Success.TButton",
            font=("Segoe UI Semibold", 10),
            foreground="#07111F",
            background="#5FD19A",
        )
        style.map("Success.TButton", background=[("active", "#87E4B8")])
        style.configure("TNotebook", background="#07111F", borderwidth=0)
        style.configure(
            "TNotebook.Tab", padding=(16, 8), background="#102335", foreground="#AFC2CF"
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1A3850")],
            foreground=[("selected", "#FFFFFF")],
        )
        style.configure(
            "Treeview",
            rowheight=26,
            background="#0A1724",
            fieldbackground="#0A1724",
            foreground="#D9E8F2",
        )
        style.configure("Treeview.Heading", background="#1A3850", foreground="#FFFFFF")

    def _create_variables(self) -> None:
        names = [body.name for body in self.catalog.bodies]
        self.body_names = tuple(names)
        self.method_var = tk.StringVar(value=DIRECT_LABEL)
        self.departure_var = tk.StringVar(value=names[1])
        self.assist_var = tk.StringVar(value=names[-2])
        self.destination_var = tk.StringVar(value=names[2])
        self.departure_day_var = tk.StringVar(value="0")
        self.assist_day_var = tk.StringVar(value="900")
        self.arrival_day_var = tk.StringVar(value="260")
        self.parking_altitude_var = tk.StringVar(value="200")
        self.capture_altitude_var = tk.StringVar(value="200")
        self.flyby_altitude_var = tk.StringVar(value="200")
        self.initial_mass_var = tk.StringVar(value="1450")
        self.dry_mass_var = tk.StringVar(value="200")
        self.isp_var = tk.StringVar(value="450")
        self.sample_count_var = tk.StringVar(value="900")
        self.maximum_c3_var = tk.StringVar(value="30")
        self.maximum_arrival_speed_var = tk.StringVar(value="8")
        self.uncertainty_samples_var = tk.StringVar(value="100")
        self.random_seed_var = tk.StringVar(value="218")
        self.porkchop_resolution_var = tk.StringVar(value="18")
        self.ephemeris_var = tk.StringVar(value="Analytical synthetic")
        self.maneuver_name_var = tk.StringVar(value="Course correction 1")
        self.maneuver_day_var = tk.StringVar(value="100")
        self.maneuver_frame_var = tk.StringVar(value="RTN local")
        self.maneuver_r_var = tk.StringVar(value="0")
        self.maneuver_t_var = tk.StringVar(value="5")
        self.maneuver_n_var = tk.StringVar(value="0")
        self.maneuver_isp_var = tk.StringVar(value="450")
        self.status_var = tk.StringVar(value="Ready — choose inputs or run the verified example.")

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(22, 14, 22, 10))
        header.pack(fill="x")
        ttk.Label(header, text="AeroGNC-Lab Mission Designer", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text="Fictional civilian planetary system • synthetic SI parameters • no targeting or homing logic",
            style="Safety.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        action_bar = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        action_bar.pack(fill="x", padx=22, pady=(0, 10))
        ttk.Button(
            action_bar, text="DESIGN MISSION", style="Primary.TButton", command=self._design
        ).pack(side="left", padx=3)
        self.open_button = ttk.Button(
            action_bar, text="OPEN 3D SIMULATION", command=self._open_3d, state="disabled"
        )
        self.open_button.pack(side="left", padx=3)
        ttk.Button(
            action_bar,
            text="RUN VERIFIED N-BODY EXAMPLE",
            style="Success.TButton",
            command=self._run_verified,
        ).pack(side="left", padx=3)
        ttk.Button(action_bar, text="PORKCHOP PLOT", command=self._porkchop).pack(
            side="left", padx=3
        )
        ttk.Button(action_bar, text="UNCERTAINTY", command=self._uncertainty).pack(
            side="left", padx=3
        )
        ttk.Button(action_bar, text="SAVE INPUTS", command=self._save_inputs).pack(
            side="right", padx=3
        )

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=22)
        ttk.Label(self.root, textvariable=self.status_var, style="Muted.TLabel").pack(
            fill="x", padx=24, pady=(4, 8)
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=22, pady=(0, 18))
        self.mission_tab = ttk.Frame(self.notebook, padding=18)
        self.spacecraft_tab = ttk.Frame(self.notebook, padding=18)
        self.maneuver_tab = ttk.Frame(self.notebook, padding=18)
        self.advanced_tab = ttk.Frame(self.notebook, padding=18)
        self.results_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(self.mission_tab, text="1  Mission")
        self.notebook.add(self.spacecraft_tab, text="2  Spacecraft")
        self.notebook.add(self.maneuver_tab, text="3  Maneuvers")
        self.notebook.add(self.advanced_tab, text="4  Analysis")
        self.notebook.add(self.results_tab, text="5  Results")
        self._build_mission_tab()
        self._build_spacecraft_tab()
        self._build_maneuver_tab()
        self._build_advanced_tab()
        self._build_results_tab()

    def _field(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        unit: str,
        help_text: str,
        *,
        width: int = 18,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=7)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=7)
        ttk.Label(parent, text=unit, style="Muted.TLabel").grid(
            row=row, column=2, sticky="w", padx=8
        )
        ttk.Label(parent, text=help_text, style="Muted.TLabel", wraplength=450).grid(
            row=row, column=3, sticky="w", padx=(12, 0)
        )
        return entry

    def _build_mission_tab(self) -> None:
        tab = self.mission_tab
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=2)
        ttk.Label(tab, text="Route and timing", style="Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10)
        )
        ttk.Label(tab, text="Design method").grid(row=1, column=0, sticky="w", pady=7)
        method = ttk.Combobox(
            tab, textvariable=self.method_var, values=(DIRECT_LABEL, ASSIST_LABEL), state="readonly"
        )
        method.grid(row=1, column=1, sticky="ew", pady=7)
        method.bind("<<ComboboxSelected>>", self._method_changed)
        ttk.Label(
            tab,
            text="Direct Lambert is the easiest starting point; gravity assist joins two legs and checks B-plane feasibility.",
            style="Muted.TLabel",
            wraplength=500,
        ).grid(row=1, column=3, sticky="w", padx=(12, 0))
        for row, label, variable in (
            (2, "Departure world", self.departure_var),
            (3, "Flyby world", self.assist_var),
            (4, "Destination world", self.destination_var),
        ):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", pady=7)
            box = ttk.Combobox(tab, textvariable=variable, values=self.body_names, state="readonly")
            box.grid(row=row, column=1, sticky="ew", pady=7)
            if row == 3:
                self.assist_box = box
        self.assist_day_entry = self._field(
            tab,
            5,
            "Departure epoch",
            self.departure_day_var,
            "catalog day",
            "Day 0 is the synthetic catalog epoch; this is not a calendar date.",
        )
        self._field(
            tab,
            6,
            "Flyby encounter",
            self.assist_day_var,
            "catalog day",
            "Used only for the one-assist method; it must fall between departure and arrival.",
        )
        self._field(
            tab,
            7,
            "Destination arrival",
            self.arrival_day_var,
            "catalog day",
            "The solver uses arrival minus departure as time of flight.",
        )
        ttk.Button(
            tab, text="Restore understandable example values", command=self._load_example_values
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(16, 0))

    def _build_spacecraft_tab(self) -> None:
        tab = self.spacecraft_tab
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=2)
        ttk.Label(tab, text="Spacecraft and encounter assumptions", style="Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10)
        )
        self._field(
            tab,
            1,
            "Initial wet mass",
            self.initial_mass_var,
            "kg",
            "Mass before ideal departure injection.",
        )
        self._field(
            tab,
            2,
            "Dry mass",
            self.dry_mass_var,
            "kg",
            "Structural/payload mass that propellant use may never cross.",
        )
        self._field(
            tab,
            3,
            "Specific impulse",
            self.isp_var,
            "s",
            "Used only for ideal rocket-equation feasibility; no propulsion hardware is implied.",
        )
        self._field(
            tab,
            4,
            "Departure parking altitude",
            self.parking_altitude_var,
            "km",
            "Circular parking-orbit altitude used to estimate injection delta-v.",
        )
        self._field(
            tab,
            5,
            "Destination parking altitude",
            self.capture_altitude_var,
            "km",
            "Circular orbit used to estimate ideal arrival capture delta-v.",
        )
        self._field(
            tab,
            6,
            "Minimum flyby altitude",
            self.flyby_altitude_var,
            "km",
            "B-plane periapsis constraint for the fictional assist world.",
        )
        ttk.Label(
            tab,
            text="The preliminary designer reports injection/capture numbers separately from manual midcourse corrections. A green overall result requires both propellant margin and destination-corridor arrival.",
            style="Muted.TLabel",
            wraplength=800,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(18, 0))

    def _build_maneuver_tab(self) -> None:
        tab = self.maneuver_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        left = ttk.Frame(tab)
        right = ttk.Frame(tab, style="Card.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(left, text="Manual direct-transfer corrections", style="Section.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        self.maneuver_tree = ttk.Treeview(
            left, columns=("name", "day", "frame", "dv", "isp"), show="headings", height=12
        )
        for key, title, width in (
            ("name", "Name", 150),
            ("day", "Elapsed day", 100),
            ("frame", "Frame", 100),
            ("dv", "Vector [m/s]", 210),
            ("isp", "Isp [s]", 80),
        ):
            self.maneuver_tree.heading(key, text=title)
            self.maneuver_tree.column(key, width=width, anchor="center")
        self.maneuver_tree.pack(fill="both", expand=True)
        ttk.Button(left, text="Remove selected correction", command=self._remove_maneuver).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(
            left,
            text="Corrections are applied at exact event times, alter the 3D path and consume propellant. They are intentionally disabled for preliminary gravity-assist leg matching.",
            style="Muted.TLabel",
            wraplength=650,
        ).pack(anchor="w", pady=(12, 0))
        ttk.Label(right, text="Add correction", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        fields = (
            ("Name", self.maneuver_name_var),
            ("Elapsed day", self.maneuver_day_var),
            ("Radial / X [m/s]", self.maneuver_r_var),
            ("Transverse / Y [m/s]", self.maneuver_t_var),
            ("Normal / Z [m/s]", self.maneuver_n_var),
            ("Specific impulse [s]", self.maneuver_isp_var),
        )
        for row, (label, variable) in enumerate(fields, start=1):
            ttk.Label(right, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(right, textvariable=variable, width=18).grid(
                row=row, column=1, sticky="ew", padx=(8, 0), pady=5
            )
        ttk.Label(right, text="Frame").grid(row=7, column=0, sticky="w", pady=5)
        ttk.Combobox(
            right,
            textvariable=self.maneuver_frame_var,
            values=("RTN local", "Inertial XYZ"),
            state="readonly",
            width=16,
        ).grid(row=7, column=1, sticky="ew", padx=(8, 0), pady=5)
        ttk.Button(
            right, text="ADD CORRECTION", style="Primary.TButton", command=self._add_maneuver
        ).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(12, 0))

    def _build_advanced_tab(self) -> None:
        tab = self.advanced_tab
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=2)
        ttk.Label(
            tab, text="Constraints, launch windows and repeatability", style="Section.TLabel"
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self._field(
            tab,
            1,
            "Maximum departure C3",
            self.maximum_c3_var,
            "km²/s²",
            "Cells above this are marked infeasible in the porkchop plot.",
        )
        self._field(
            tab,
            2,
            "Maximum arrival excess speed",
            self.maximum_arrival_speed_var,
            "km/s",
            "Arrival corridor constraint used for launch-window screening.",
        )
        self._field(
            tab,
            3,
            "Trajectory samples",
            self.sample_count_var,
            "samples",
            "100–5000 points for smooth 3D playback.",
        )
        self._field(
            tab,
            4,
            "Porkchop resolution",
            self.porkchop_resolution_var,
            "points/axis",
            "8–40 is responsive; each cell solves one Lambert problem.",
        )
        self._field(
            tab,
            5,
            "Uncertainty samples",
            self.uncertainty_samples_var,
            "runs",
            "Seeded fast design-space dispersions; failed solves are recorded.",
        )
        self._field(
            tab,
            6,
            "Random seed",
            self.random_seed_var,
            "integer",
            "Using the same inputs and seed reproduces the same draws.",
        )
        ttk.Label(tab, text="Ephemeris source").grid(row=7, column=0, sticky="w", pady=7)
        ephemeris_values = ["Analytical synthetic"]
        if importlib.util.find_spec("spiceypy") is not None:
            ephemeris_values.append("SPICE (kernels required)")
        ttk.Combobox(
            tab, textvariable=self.ephemeris_var, values=ephemeris_values, state="readonly"
        ).grid(row=7, column=1, sticky="ew", pady=7)
        availability = (
            "SPICE adapter detected; user kernels are still required."
            if len(ephemeris_values) > 1
            else "SPICE is optional and not installed; analytical synthetic ephemerides remain fully reproducible."
        )
        ttk.Label(tab, text=availability, style="Muted.TLabel", wraplength=520).grid(
            row=7, column=3, sticky="w", padx=(12, 0)
        )
        ttk.Label(
            tab,
            text="Fidelity note: the verified example uses custom restricted N-body RK4 propagation. J2, solar-radiation pressure, relativity and full mutually interacting N-body models are available as explicit engineering modules; they are not silently mixed into Lambert screening.",
            style="Muted.TLabel",
            wraplength=850,
        ).grid(row=8, column=0, columnspan=4, sticky="w", pady=(18, 0))

    def _build_results_tab(self) -> None:
        tab = self.results_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)
        ttk.Label(tab, text="Engineering result", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(tab, text="Event timeline", style="Section.TLabel").grid(
            row=0, column=1, sticky="w", padx=(14, 0)
        )
        self.result_text = tk.Text(
            tab,
            wrap="word",
            background="#081521",
            foreground="#D9E8F2",
            insertbackground="white",
            relief="flat",
            padx=14,
            pady=12,
            font=("Cascadia Mono", 10),
        )
        self.result_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.event_tree = ttk.Treeview(tab, columns=("day", "body", "distance"), show="headings")
        for key, title, width in (
            ("day", "Day", 80),
            ("body", "Reference", 110),
            ("distance", "Distance", 140),
        ):
            self.event_tree.heading(key, text=title)
            self.event_tree.column(key, width=width, anchor="center")
        self.event_tree.grid(row=1, column=1, sticky="nsew", padx=(14, 0), pady=(8, 0))
        self.result_text.insert(
            "1.0", "No result yet. Click DESIGN MISSION or RUN VERIFIED N-BODY EXAMPLE.\n"
        )
        self.result_text.configure(state="disabled")

    def _load_example_values(self) -> None:
        self.method_var.set(DIRECT_LABEL)
        self.departure_var.set("Asteria")
        self.destination_var.set("Neria")
        self.assist_var.set("Brontes")
        self.departure_day_var.set("0")
        self.assist_day_var.set("900")
        self.arrival_day_var.set("260")
        self._method_changed()

    def _method_changed(self, _event: object | None = None) -> None:
        direct = self.method_var.get() == DIRECT_LABEL
        state = "disabled" if direct else "readonly"
        self.assist_box.configure(state=state)
        self.assist_day_entry.configure(state="disabled" if direct else "normal")

    @staticmethod
    def _number(variable: tk.StringVar, label: str, *, minimum: float | None = None) -> float:
        raw = variable.get().strip()
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"{label} must be a number; received {raw!r}") from error
        if not np.isfinite(value) or (minimum is not None and value < minimum):
            relation = f" at least {minimum:g}" if minimum is not None else " finite"
            raise ValueError(f"{label} must be{relation}")
        return value

    def _request(self) -> MissionPlanRequest:
        if self.ephemeris_var.get() != "Analytical synthetic":
            raise ValueError(
                "SPICE design requires explicitly selected public kernels; none are configured in this form"
            )
        method: MissionMethod = (
            "direct" if self.method_var.get() == DIRECT_LABEL else "gravity_assist"
        )
        samples = int(self._number(self.sample_count_var, "Trajectory samples", minimum=100))
        return MissionPlanRequest(
            method=method,
            departure_body=self.departure_var.get(),
            assist_body=self.assist_var.get() if method == "gravity_assist" else None,
            destination_body=self.destination_var.get(),
            departure_day=self._number(self.departure_day_var, "Departure epoch", minimum=0.0),
            assist_day=self._number(self.assist_day_var, "Flyby day", minimum=0.0)
            if method == "gravity_assist"
            else None,
            arrival_day=self._number(self.arrival_day_var, "Arrival epoch", minimum=0.0),
            parking_altitude_m=1_000.0
            * self._number(self.parking_altitude_var, "Departure parking altitude", minimum=0.001),
            destination_parking_altitude_m=1_000.0
            * self._number(
                self.capture_altitude_var, "Destination parking altitude", minimum=0.001
            ),
            minimum_flyby_altitude_m=1_000.0
            * self._number(self.flyby_altitude_var, "Minimum flyby altitude", minimum=0.001),
            initial_mass_kg=self._number(self.initial_mass_var, "Initial wet mass", minimum=0.001),
            dry_mass_kg=self._number(self.dry_mass_var, "Dry mass", minimum=0.001),
            specific_impulse_s=self._number(self.isp_var, "Specific impulse", minimum=0.001),
            sample_count=samples,
            maneuvers=tuple(self.maneuvers),
        )

    def _run_background(
        self, label: str, worker: Callable[[], T], on_success: Callable[[T], None]
    ) -> None:
        if self.busy:
            messagebox.showinfo("AeroGNC-Lab", "Another calculation is already running.")
            return
        self.busy = True
        self.status_var.set(label)
        self.progress.start(12)

        def task() -> None:
            try:
                result = worker()
            except Exception as error:  # GUI boundary: keep the desktop app usable.

                def report_error(captured_error: Exception = error) -> None:
                    self._background_error(captured_error)

                self.root.after(0, report_error)
            else:
                self.root.after(0, lambda: self._background_success(result, on_success))

        threading.Thread(target=task, daemon=True, name="aerognc-mission-worker").start()

    def _background_error(self, error: Exception) -> None:
        self.progress.stop()
        self.busy = False
        self.status_var.set(f"Stopped: {error}")
        messagebox.showerror("Input or solver error", str(error))

    def _background_success(self, result: T, callback: Callable[[T], None]) -> None:
        self.progress.stop()
        self.busy = False
        callback(result)

    def _design(self) -> None:
        try:
            request = self._request()
        except ValueError as error:
            self._background_error(error)
            return
        self._run_background(
            "Solving Lambert legs and preparing the 3D trajectory…",
            lambda: plan_mission(self.catalog, request),
            self._design_ready,
        )

    def _design_ready(self, plan: PlannedMission) -> None:
        self.current_plan = plan
        self.current_mission = plan.mission
        self.open_button.configure(state="normal")
        self._display_plan(plan)
        self.notebook.select(self.results_tab)  # type: ignore[no-untyped-call]
        self.status_var.set("Design complete — inspect the result, then open the 3D simulation.")

    def _run_verified(self) -> None:
        def worker() -> InterplanetaryMission:
            configuration = load_interplanetary_configuration(self.verified_configuration_path)
            return simulate_interplanetary(configuration)

        self._run_background(
            "Running the verified restricted N-body example (usually 20–30 s)…",
            worker,
            self._verified_ready,
        )

    def _verified_ready(self, mission: InterplanetaryMission) -> None:
        self.current_plan = None
        self.current_mission = mission
        self.open_button.configure(state="normal")
        self._display_mission(mission, heading="VERIFIED RESTRICTED N-BODY REFERENCE")
        self.notebook.select(self.results_tab)  # type: ignore[no-untyped-call]
        self.status_var.set("Verified example complete — OPEN 3D SIMULATION is ready.")

    def _open_3d(self) -> None:
        if self.current_mission is None:
            messagebox.showinfo("AeroGNC-Lab", "Design or run a mission first.")
            return
        from aerognc.visualisation.mission_control import play_interplanetary_mission

        self.status_var.set("3D mission control is open; close it to return to Mission Designer.")
        play_interplanetary_mission(self.current_mission, show_window=True)
        self.status_var.set("3D mission control closed. Inputs and results are preserved.")

    def _display_plan(self, plan: PlannedMission) -> None:
        metrics = plan.metrics
        request = plan.request
        flyby = "not used"
        if metrics.flyby_altitude_m is not None:
            flyby = f"{metrics.flyby_altitude_m / 1_000.0:,.1f} km; powered mismatch {metrics.powered_flyby_delta_v_mps or 0.0:,.2f} m/s"
        maximum_c3 = self._number(self.maximum_c3_var, "Maximum C3", minimum=0.0)
        maximum_arrival = self._number(
            self.maximum_arrival_speed_var, "Maximum arrival speed", minimum=0.0
        )
        c3_pass = metrics.departure_c3_m2_s2 / 1.0e6 <= maximum_c3
        arrival_pass = metrics.arrival_excess_speed_mps / 1_000.0 <= maximum_arrival
        overall = metrics.feasible and c3_pass and arrival_pass
        text = (
            f"{'PASS' if overall else 'REVIEW'} — PRELIMINARY PATCHED-CONIC DESIGN\n"
            f"Method                    {request.method.replace('_', ' ')}\n"
            f"Route                     {request.departure_body} → {request.destination_body}\n"
            f"Mission elapsed time      {request.arrival_day - request.departure_day:,.2f} days\n\n"
            f"Departure C3              {metrics.departure_c3_m2_s2 / 1.0e6:,.3f} km²/s²   {'PASS' if c3_pass else 'FAIL'}\n"
            f"Departure excess speed    {metrics.departure_excess_speed_mps / 1_000.0:,.3f} km/s\n"
            f"Ideal injection Δv        {metrics.injection_delta_v_mps:,.1f} m/s\n"
            f"Manual midcourse Δv       {metrics.midcourse_delta_v_mps:,.1f} m/s\n"
            f"Arrival excess speed      {metrics.arrival_excess_speed_mps / 1_000.0:,.3f} km/s   {'PASS' if arrival_pass else 'FAIL'}\n"
            f"Ideal capture Δv          {metrics.ideal_capture_delta_v_mps:,.1f} m/s\n"
            f"Ideal total Δv            {metrics.ideal_total_delta_v_mps:,.1f} m/s\n"
            f"Ideal propellant required {metrics.ideal_propellant_required_kg:,.1f} kg\n"
            f"Destination miss          {metrics.destination_miss_distance_m / 1_000.0:,.3f} km\n"
            f"Flyby geometry            {flyby}\n\n"
            "Interpretation\n"
            "• This result is a preliminary Lambert/patched-conic design, not an operational mission.\n"
            "• Manual corrections are propagated and can create a destination miss.\n"
            "• Use the verified N-body example to inspect numerical gravity-assist propagation.\n"
            "• All worlds, vehicle values and epochs are fictional and synthetic.\n"
        )
        self._set_result_text(text)
        self._set_events(plan.mission)

    def _display_mission(self, mission: InterplanetaryMission, *, heading: str) -> None:
        summary = mission.result.maximum_summary
        text = (
            f"{heading}\n"
            f"Scenario                   {mission.result.scenario_name}\n"
            f"Samples                    {mission.result.time_s.size:,}\n"
            f"Simulated duration         {mission.result.time_s[-1] / 86_400.0:,.2f} days\n"
            f"Execution time             {mission.result.execution_time_s:,.3f} s\n"
            f"Assist closest approach    {float(summary['assist_closest_approach']['value']) / 1.0e3:,.1f} km\n"
            f"Assist speed gain          {float(summary['assist_heliocentric_speed_gain']['value']) / 1.0e3:,.3f} km/s\n"
            f"Destination closest        {float(summary['destination_closest_approach']['value']) / 1.0e6:,.3f} Mm\n"
            f"Destination corridor       {'PASS' if float(summary['destination_arrival']['value']) > 0.5 else 'NOT REACHED'}\n\n"
            "This trajectory was integrated with the custom restricted N-body RK4 model.\n"
            "It remains a synthetic civilian research example, not a real mission ephemeris.\n"
        )
        self._set_result_text(text)
        self._set_events(mission)

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _set_events(self, mission: InterplanetaryMission) -> None:
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        for event in mission.result.event_summary:
            distance = float(event["distance_m"])
            display_distance = (
                f"{distance / 1.0e6:,.3f} Mm"
                if distance >= 1.0e6
                else f"{distance / 1.0e3:,.2f} km"
            )
            self.event_tree.insert(
                "",
                "end",
                values=(
                    f"{float(event['time_days']):.2f}",
                    event["reference_body"],
                    display_distance,
                ),
                text=str(event["name"]),
            )

    def _add_maneuver(self) -> None:
        try:
            frame: ManeuverFrame = (
                "rtn" if self.maneuver_frame_var.get() == "RTN local" else "inertial"
            )
            maneuver = ImpulsiveManeuver(
                name=self.maneuver_name_var.get().strip(),
                epoch_s=86_400.0
                * self._number(self.maneuver_day_var, "Maneuver elapsed day", minimum=0.0),
                delta_velocity_mps=(
                    self._number(self.maneuver_r_var, "First maneuver component"),
                    self._number(self.maneuver_t_var, "Second maneuver component"),
                    self._number(self.maneuver_n_var, "Third maneuver component"),
                ),
                frame=frame,
                specific_impulse_s=self._number(
                    self.maneuver_isp_var, "Maneuver specific impulse", minimum=0.001
                ),
            )
            if any(
                existing.name.casefold() == maneuver.name.casefold() for existing in self.maneuvers
            ):
                raise ValueError("Maneuver names must be unique")
        except ValueError as error:
            messagebox.showerror("Maneuver input", str(error))
            return
        self.maneuvers.append(maneuver)
        self._refresh_maneuvers()
        self.maneuver_name_var.set(f"Course correction {len(self.maneuvers) + 1}")

    def _remove_maneuver(self) -> None:
        selected = self.maneuver_tree.selection()
        if not selected:
            return
        indices = sorted((int(item) for item in selected), reverse=True)
        for index in indices:
            self.maneuvers.pop(index)
        self._refresh_maneuvers()

    def _refresh_maneuvers(self) -> None:
        for item in self.maneuver_tree.get_children():
            self.maneuver_tree.delete(item)
        for index, maneuver in enumerate(self.maneuvers):
            vector = ", ".join(f"{value:g}" for value in maneuver.delta_velocity_mps)
            self.maneuver_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    maneuver.name,
                    f"{maneuver.epoch_s / 86_400.0:g}",
                    maneuver.frame.upper(),
                    vector,
                    f"{maneuver.specific_impulse_s:g}",
                ),
                text=maneuver.name,
            )

    def _porkchop(self) -> None:
        try:
            request = self._request()
            resolution = int(
                self._number(self.porkchop_resolution_var, "Porkchop resolution", minimum=8)
            )
            if resolution > 40:
                raise ValueError("Porkchop resolution cannot exceed 40 points per axis")
            departure = self.catalog.body(request.departure_body, role="departure")
            if request.method == "gravity_assist":
                if request.assist_body is None or request.assist_day is None:
                    raise ValueError("Select a flyby body and day")
                arrival = self.catalog.body(request.assist_body, role="destination")
                centre_arrival_day = request.assist_day
            else:
                arrival = self.catalog.body(request.destination_body, role="destination")
                centre_arrival_day = request.arrival_day
            departure_days = np.linspace(
                max(0.0, request.departure_day - 180.0), request.departure_day + 180.0, resolution
            )
            time_of_flight = centre_arrival_day - request.departure_day
            arrival_days = np.linspace(
                centre_arrival_day - 0.35 * time_of_flight,
                centre_arrival_day + 0.35 * time_of_flight,
                resolution,
            )
            maximum_c3 = self._number(self.maximum_c3_var, "Maximum C3", minimum=0.001) * 1.0e6
            maximum_arrival = (
                self._number(self.maximum_arrival_speed_var, "Maximum arrival speed", minimum=0.001)
                * 1.0e3
            )
        except ValueError as error:
            messagebox.showerror("Launch-window inputs", str(error))
            return

        def worker() -> object:
            return compute_porkchop_grid(
                departure,
                arrival,
                self.catalog.primary.gravitational_parameter_m3_s2,
                departure_days * 86_400.0,
                arrival_days * 86_400.0,
                maximum_c3_m2_s2=maximum_c3,
                maximum_arrival_excess_speed_mps=maximum_arrival,
            )

        self._run_background(
            "Computing the launch-window grid…",
            worker,
            lambda grid: self._show_porkchop(grid, departure.name, arrival.name),
        )

    def _show_porkchop(self, grid: object, departure: str, arrival: str) -> None:
        from aerognc.astrodynamics.mission_design import PorkchopGrid
        from aerognc.visualisation.mission_design import show_porkchop_grid

        typed_grid = cast(PorkchopGrid, grid)
        self.status_var.set("Launch-window grid complete; close the plot to return.")
        show_porkchop_grid(typed_grid, title=f"{departure} → {arrival} synthetic launch window")

    def _uncertainty(self) -> None:
        try:
            request = self._request()
            sample_count = int(
                self._number(self.uncertainty_samples_var, "Uncertainty samples", minimum=5)
            )
            if sample_count > 2_000:
                raise ValueError("Uncertainty sample count cannot exceed 2000")
            seed = int(self._number(self.random_seed_var, "Random seed", minimum=0))
        except ValueError as error:
            messagebox.showerror("Uncertainty inputs", str(error))
            return
        primary_mu = self.catalog.primary.gravitational_parameter_m3_s2
        departure = self.catalog.body(request.departure_body, role="departure")
        destination = self.catalog.body(request.destination_body, role="destination")

        def evaluator(parameters: Mapping[str, float]) -> Mapping[str, float]:
            departure_time = (request.departure_day + parameters["departure_day"]) * 86_400.0
            arrival_time = (request.arrival_day + parameters["arrival_day"]) * 86_400.0
            if request.method == "direct":
                transfer = evaluate_lambert_transfer(
                    departure, destination, primary_mu, departure_time, arrival_time
                )
                return {
                    "departure_c3_km2_s2": transfer.departure_c3_m2_s2 / 1.0e6,
                    "arrival_excess_kmps": transfer.arrival_excess_speed_mps / 1.0e3,
                }
            if request.assist_body is None or request.assist_day is None:
                raise ValueError("assist inputs are missing")
            assist = self.catalog.body(request.assist_body, role="assist")
            assist_time = (request.assist_day + parameters["assist_day"]) * 86_400.0
            design = design_gravity_assist(
                departure,
                assist,
                destination,
                primary_mu,
                departure_time,
                assist_time,
                arrival_time,
                minimum_flyby_altitude_m=request.minimum_flyby_altitude_m,
                excess_speed_tolerance_mps=1.0e9,
            )
            return {
                "departure_c3_km2_s2": design.first_leg.departure_c3_m2_s2 / 1.0e6,
                "arrival_excess_kmps": design.second_leg.arrival_excess_speed_mps / 1.0e3,
                "flyby_powered_mismatch_mps": design.flyby.powered_flyby_delta_v_mps,
            }

        sigmas = {"departure_day": 0.5, "arrival_day": 1.0, "assist_day": 1.0}
        self._run_background(
            "Running seeded mission-design uncertainty samples…",
            lambda: run_seeded_uncertainty(
                sigmas, evaluator, sample_count=sample_count, seed=seed, workers=1
            ),
            self._uncertainty_ready,
        )

    def _uncertainty_ready(self, summary: UncertaintySummary) -> None:
        lines = [
            "\n\nSEEDED UNCERTAINTY SUMMARY",
            f"Seed {summary.seed}; {summary.successful_count} successful; {summary.failed_count} failed",
        ]
        for metric, statistics in summary.metric_statistics.items():
            lines.append(
                f"{metric:<30} mean {statistics['mean']:,.4g}   95% [{statistics['p02_5']:,.4g}, {statistics['p97_5']:,.4g}]"
            )
        self.result_text.configure(state="normal")
        self.result_text.insert("end", "\n".join(lines) + "\n")
        self.result_text.configure(state="disabled")
        self.notebook.select(self.results_tab)  # type: ignore[no-untyped-call]
        self.status_var.set("Uncertainty analysis complete and appended to Results.")

    def _save_inputs(self) -> None:
        try:
            request = self._request()
        except ValueError as error:
            messagebox.showerror("Cannot save inputs", str(error))
            return
        path = filedialog.asksaveasfilename(
            title="Save synthetic mission-design inputs",
            defaultextension=".yaml",
            filetypes=(("YAML", "*.yaml"), ("All files", "*.*")),
        )
        if not path:
            return
        payload = {
            "metadata": {
                "safety_scope": "Fictional civilian synthetic mission-design input; no operational targeting use.",
                "model": "preliminary Lambert/patched-conic",
            },
            "route": {
                "method": request.method,
                "departure_body": request.departure_body,
                "assist_body": request.assist_body,
                "destination_body": request.destination_body,
                "departure_day": request.departure_day,
                "assist_day": request.assist_day,
                "arrival_day": request.arrival_day,
            },
            "spacecraft": {
                "initial_mass_kg": request.initial_mass_kg,
                "dry_mass_kg": request.dry_mass_kg,
                "specific_impulse_s": request.specific_impulse_s,
                "parking_altitude_m": request.parking_altitude_m,
                "destination_parking_altitude_m": request.destination_parking_altitude_m,
            },
            "maneuvers": [
                {
                    "name": maneuver.name,
                    "elapsed_day": maneuver.epoch_s / 86_400.0,
                    "frame": maneuver.frame,
                    "delta_velocity_mps": list(maneuver.delta_velocity_mps),
                    "specific_impulse_s": maneuver.specific_impulse_s,
                }
                for maneuver in request.maneuvers
            ],
        }
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.status_var.set(f"Inputs saved to {path}")


def launch_mission_designer(
    catalog_path: str | Path = "configs/fictional_planetary_system.yaml",
    verified_configuration_path: str | Path = "configs/interplanetary_gravity_assist.yaml",
) -> None:
    """Load the synthetic catalog and run the desktop event loop."""
    catalog = load_planetary_catalog(catalog_path)
    root = tk.Tk()
    MissionDesignerApp(root, catalog, Path(verified_configuration_path))
    root.mainloop()
