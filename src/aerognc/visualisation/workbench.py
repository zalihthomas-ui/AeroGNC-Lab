"""Unified desktop workbench for AeroGNC-Lab's approachable simulation workflows."""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np

from aerognc.catalogs import (
    ConfirmedExoplanet,
    ExoplanetCatalog,
    MilkyWayMetadata,
    SolarSystemPlanet,
    load_milky_way_metadata,
    load_solar_system_planets,
)
from aerognc.configuration.planetary_catalog import PlanetaryCatalog, load_planetary_catalog
from aerognc.project import CancellationToken, StoredRun
from aerognc.simulation.aircraft_sandbox import AircraftSandboxSimulation
from aerognc.simulation.logging import SimulationResult, write_result_csv, write_summary_json
from aerognc.simulation.orbit_assisted_tour import (
    OrbitTourSimulation,
    write_orbit_tour_results,
)
from aerognc.simulation.orbit_sandbox import OrbitSandboxSimulation
from aerognc.simulation.workbench import (
    CatalogWorkbenchInputs,
    OrbitTourWorkbenchInputs,
    RocketWorkbenchInputs,
    load_workbench_catalog,
    run_orbit_tour_workbench,
    run_rocket_workbench,
    search_exoplanet_catalog,
)
from aerognc.simulation.workbench_project import (
    ProjectRunComparison,
    ProjectWorkbenchService,
    ProjectWorkbenchSnapshot,
)

if TYPE_CHECKING:
    from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
    from aerognc.configuration.orbit_sandbox_loader import OrbitSandboxConfiguration
    from aerognc.simulation.aircraft_training import AircraftPresetName, TrainingTask
    from aerognc.visualisation.aircraft_controls import AircraftControlMode, PilotControlProfile
    from aerognc.visualisation.aircraft_experience import TrailColorSource, TrailMode
    from aerognc.visualisation.aircraft_live import LiveCameraMode
    from aerognc.visualisation.mesh import MeshAxisConvention, MeshCenterMode, MeshTransform

T = TypeVar("T")

BACKGROUND = "#07111F"
PANEL = "#0D1B2A"
CARD = "#102335"
CARD_LIGHT = "#153048"
TEXT = "#D9E8F2"
MUTED = "#8FA6B8"
CYAN = "#39C6E8"
GREEN = "#5FD19A"
AMBER = "#F2B84B"

ORBIT_MODEL_CHOICES = {
    "No force - straight line (1 moving body)": "free",
    "Planet + satellite (2-body orbit)": "two_body",
    "Planet + moon + satellite (restricted 3-body)": "restricted_three_body",
    "All configured bodies pull each other (full N-body)": "full_n_body",
    "Orbit lifetime with J2 + atmospheric drag": "perturbed_decay",
}
ORBIT_SPEED_CHOICES = {
    "Circular orbit speed (calculated)": "circular",
    "Escape speed (calculated)": "escape",
    "Enter my own speed": "custom",
}
MESH_AXIS_CHOICES = {
    "+X forward, +Y right, +Z down (AeroGNC body axes)": "body_frd",
    "+X forward, +Y left, +Z up (common CAD)": "x_forward_z_up",
    "+Y forward, +X right, +Z up": "y_forward_z_up",
}
AIRCRAFT_PRESET_CHOICES = {
    "Level Flight - easiest first run": "level_flight",
    "Coordinated 360 Turn": "coordinated_turn",
    "Stall Demonstration + Recovery": "stall_demonstration",
    "Crosswind + Seeded Gust": "crosswind_response",
    "High-Altitude Research": "high_altitude_research",
}
AIRCRAFT_TRAINING_TASK_BY_PRESET = {
    "level_flight": "altitude_speed_hold",
    "coordinated_turn": "coordinated_360_turn",
    "stall_demonstration": "stall_recovery",
    "crosswind_response": "altitude_speed_hold",
    "high_altitude_research": "research_altitude_crossing",
}
AIRCRAFT_CONTROL_MODE_CHOICES = {
    "Stability assisted - recommended": "stability_assisted",
    "Direct control surfaces - engineering": "direct",
}
AIRCRAFT_CAMERA_CHOICES = {
    "Smooth chase": "chase",
    "Cockpit / forward": "cockpit",
    "Orbit around path": "orbit",
    "Top-down": "top",
    "Free Matplotlib view": "free",
}
AIRCRAFT_TRAIL_CHOICES = {
    "Fading recent trail": "fading",
    "Full-session trail": "full",
    "No trail": "off",
}
AIRCRAFT_TRAIL_COLOR_CHOICES = {
    "Constant blue": "constant",
    "Colour by altitude": "altitude",
    "Colour by airspeed": "airspeed",
}


@dataclass(frozen=True, slots=True)
class WorkbenchPaths:
    """Local files required by the unified UI; no network service is needed."""

    six_dof_configuration: Path
    orbit_tour_configuration: Path
    planetary_catalog: Path
    verified_interplanetary_configuration: Path
    exoplanet_csv: Path
    exoplanet_metadata: Path
    milky_way_metadata: Path
    solar_system_planets: Path
    project_file: Path | None = None
    orbit_sandbox_configuration: Path | None = None
    aircraft_configuration: Path | None = None
    aircraft_mesh: Path | None = None

    def validate(self) -> None:
        """Fail with one readable message if an installed data file is absent."""
        required_paths: tuple[Path, ...] = (
            self.six_dof_configuration,
            self.orbit_tour_configuration,
            self.planetary_catalog,
            self.verified_interplanetary_configuration,
            self.exoplanet_csv,
            self.exoplanet_metadata,
            self.milky_way_metadata,
            self.solar_system_planets,
        )
        if self.project_file is not None:
            required_paths = (*required_paths, self.project_file)
        for optional_path in (
            self.orbit_sandbox_configuration,
            self.aircraft_configuration,
            self.aircraft_mesh,
        ):
            if optional_path is not None:
                required_paths = (*required_paths, optional_path)
        missing = [path for path in required_paths if not path.is_file()]
        if missing:
            joined = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"AeroGNC-Lab workbench files are missing:\n{joined}")


@dataclass(frozen=True, slots=True)
class RocketRun:
    """One completed editable rocket case."""

    result: SimulationResult
    playback_speed: float
    output_directory: Path


@dataclass(frozen=True, slots=True)
class OrbitSandboxRun:
    """One completed editable satellite case."""

    simulation: OrbitSandboxSimulation
    output_directory: Path


@dataclass(frozen=True, slots=True)
class AircraftSandboxRun:
    """One completed hands-off fictional-aircraft case."""

    simulation: AircraftSandboxSimulation
    output_directory: Path


class AeroGNCWorkbenchApp:
    """One approachable UI for rocket, planetary, and astronomy-data exploration."""

    def __init__(
        self,
        root: tk.Tk,
        paths: WorkbenchPaths,
        fictional_catalog: PlanetaryCatalog,
        exoplanet_catalog: ExoplanetCatalog,
        milky_way: MilkyWayMetadata,
        solar_planets: tuple[SolarSystemPlanet, ...],
    ) -> None:
        self.root = root
        self.paths = paths
        self.fictional_catalog = fictional_catalog
        self.exoplanet_catalog = exoplanet_catalog
        self.milky_way = milky_way
        self.solar_planets = solar_planets
        self.catalog_selection: tuple[ConfirmedExoplanet, ...] = ()
        self.project_service = ProjectWorkbenchService()
        self.project_snapshot: ProjectWorkbenchSnapshot | None = None
        self.project_cancellation: CancellationToken | None = None
        self.last_project_comparison_report: Path | None = None
        self._active_aircraft_players: list[object] = []
        self.busy = False
        self.action_buttons: list[ttk.Button] = []
        self._configure_window()
        self._create_variables()
        self._build_layout()
        self._reset_rocket()
        self._reset_orbit()
        self._reset_aircraft()
        self._reset_tour()
        self._search_catalog()
        if self.paths.project_file is not None:
            self._load_project_path(self.paths.project_file, show_error=False)
        self.status_var.set("Ready - choose one of the four green examples on the Start page.")

    def _configure_window(self) -> None:
        self.root.title("AeroGNC-Lab - Simulation Workbench")
        self.root.geometry("1280x860")
        self.root.minsize(1000, 780)
        self.root.configure(background=BACKGROUND)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=PANEL, foreground=TEXT)
        style.configure("TFrame", background=PANEL)
        style.configure("Header.TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=CARD, relief="solid", borderwidth=1)
        style.configure("CardBody.TFrame", background=CARD)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Card.TLabel", background=CARD, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("CardMuted.TLabel", background=CARD, foreground=MUTED)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22), foreground="#F1F7FA")
        style.configure(
            "HeaderTitle.TLabel",
            background=BACKGROUND,
            font=("Segoe UI Semibold", 22),
            foreground="#F1F7FA",
        )
        style.configure("HeaderSafety.TLabel", background=BACKGROUND, foreground=GREEN)
        style.configure("CardSafety.TLabel", background=CARD, foreground=GREEN)
        style.configure("Hero.TLabel", font=("Segoe UI Semibold", 17), foreground="#F1F7FA")
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 12), foreground=CYAN)
        style.configure("Safety.TLabel", foreground=GREEN)
        style.configure("Metric.TLabel", font=("Segoe UI Semibold", 13), foreground=AMBER)
        style.configure("TLabelframe", background=PANEL, foreground=CYAN)
        style.configure("TLabelframe.Label", background=PANEL, foreground=CYAN)
        style.configure("TButton", padding=(11, 7))
        style.configure(
            "Primary.TButton",
            font=("Segoe UI Semibold", 10),
            foreground=BACKGROUND,
            background=CYAN,
        )
        style.map("Primary.TButton", background=[("active", "#6DDAF2")])
        style.configure(
            "Success.TButton",
            font=("Segoe UI Semibold", 10),
            foreground=BACKGROUND,
            background=GREEN,
        )
        style.map("Success.TButton", background=[("active", "#87E4B8")])
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(10, 9), background=CARD, foreground="#AFC2CF")
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1A3850")],
            foreground=[("selected", "#FFFFFF")],
        )
        style.configure(
            "Treeview",
            rowheight=25,
            background="#0A1724",
            fieldbackground="#0A1724",
            foreground=TEXT,
        )
        style.map("Treeview", background=[("selected", "#21506D")])
        style.configure("Treeview.Heading", background="#1A3850", foreground="#FFFFFF")

        # White-on-white fix: the "." style above makes the default foreground
        # near-white, but ttk entry/combobox/spinbox fields and classic tk widgets
        # keep a WHITE field background by default -> invisible text. Give every
        # editable field an explicit dark field background with light text/caret.
        field_background = "#0A1724"
        style.configure(
            "TEntry", fieldbackground=field_background, foreground=TEXT, insertcolor=TEXT
        )
        style.map(
            "TEntry",
            fieldbackground=[("readonly", CARD), ("disabled", PANEL)],
            foreground=[("disabled", MUTED)],
        )
        style.configure(
            "TSpinbox",
            fieldbackground=field_background,
            foreground=TEXT,
            insertcolor=TEXT,
            arrowcolor=TEXT,
            background=CARD,
        )
        style.configure(
            "TCombobox",
            fieldbackground=field_background,
            foreground=TEXT,
            background=CARD,
            arrowcolor=TEXT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", field_background)],
            foreground=[("readonly", TEXT), ("disabled", MUTED)],
        )
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TRadiobutton", background=PANEL, foreground=TEXT)
        # Classic (non-ttk) tk widgets ignore ttk styles; set their defaults too.
        for option, value in (
            ("*Entry.background", field_background),
            ("*Entry.foreground", TEXT),
            ("*Entry.insertBackground", TEXT),
            ("*Text.background", field_background),
            ("*Text.foreground", TEXT),
            ("*Text.insertBackground", TEXT),
            ("*Listbox.background", field_background),
            ("*Listbox.foreground", TEXT),
            ("*Listbox.selectBackground", "#21506D"),
            ("*Listbox.selectForeground", "#FFFFFF"),
            ("*Spinbox.background", field_background),
            ("*Spinbox.foreground", TEXT),
            ("*Spinbox.insertBackground", TEXT),
        ):
            self.root.option_add(option, value)

    def _create_variables(self) -> None:
        self.status_var = tk.StringVar(
            value="Ready - choose one of the four green examples on the Start page."
        )
        self.rocket_vars = {
            "duration": tk.StringVar(),
            "step": tk.StringVar(),
            "speed": tk.StringVar(),
            "roll": tk.StringVar(),
            "pitch": tk.StringVar(),
            "yaw": tk.StringVar(),
            "roll_rate": tk.StringVar(),
            "pitch_rate": tk.StringVar(),
            "yaw_rate": tk.StringVar(),
            "playback": tk.StringVar(),
        }
        self.tour_vars = {
            "departure": tk.StringVar(),
            "assist": tk.StringVar(),
            "destination": tk.StringVar(),
            "departure_day": tk.StringVar(),
            "assist_day": tk.StringVar(),
            "destination_day": tk.StringVar(),
            "departure_altitude": tk.StringVar(),
            "assist_altitude": tk.StringVar(),
            "destination_altitude": tk.StringVar(),
            "dwell": tk.StringVar(),
            "initial_mass": tk.StringVar(),
            "dry_mass": tk.StringVar(),
            "isp": tk.StringVar(),
            "maximum_delta_v": tk.StringVar(),
            "minimum_final_mass": tk.StringVar(),
            "playback": tk.StringVar(),
        }
        self.orbit_vars = {
            "model": tk.StringVar(),
            "altitude": tk.StringVar(),
            "speed_mode": tk.StringVar(),
            "custom_speed": tk.StringVar(),
            "inclination": tk.StringVar(),
            "duration_days": tk.StringVar(),
            "mass": tk.StringVar(),
            "dry_mass": tk.StringVar(),
            "area": tk.StringVar(),
            "drag_coefficient": tk.StringVar(),
            "density_scale": tk.StringVar(),
            "reentry_altitude": tk.StringVar(),
            "integration_step": tk.StringVar(),
            "output_step": tk.StringVar(),
            "correction_altitude": tk.StringVar(),
            "maximum_corrections": tk.StringVar(),
        }
        self.orbit_correction_var = tk.BooleanVar(value=False)
        self.aircraft_vars = {
            "preset": tk.StringVar(),
            "altitude": tk.StringVar(),
            "airspeed": tk.StringVar(),
            "heading": tk.StringVar(),
            "flight_path_angle": tk.StringVar(),
            "bank": tk.StringVar(),
            "alpha": tk.StringVar(),
            "throttle": tk.StringVar(),
            "duration": tk.StringVar(),
            "mass": tk.StringVar(),
            "dry_mass": tk.StringVar(),
            "wing_area": tk.StringVar(),
            "cl_zero": tk.StringVar(),
            "cl_alpha": tk.StringVar(),
            "cl_maximum": tk.StringVar(),
            "stall_angle": tk.StringVar(),
            "cd_zero": tk.StringVar(),
            "induced_drag": tk.StringVar(),
            "pitch_alpha": tk.StringVar(),
            "wind_north": tk.StringVar(),
            "wind_east": tk.StringVar(),
            "mesh_path": tk.StringVar(),
            "mesh_axes": tk.StringVar(),
            "real_time_factor": tk.StringVar(),
            "control_mode": tk.StringVar(),
            "camera": tk.StringVar(),
            "trail_mode": tk.StringVar(),
            "trail_duration": tk.StringVar(),
            "trail_color": tk.StringVar(),
            "mesh_rotation_x": tk.StringVar(),
            "mesh_rotation_y": tk.StringVar(),
            "mesh_rotation_z": tk.StringVar(),
            "mesh_center": tk.StringVar(),
            "mesh_scale_mode": tk.StringVar(),
            "recorder_directory": tk.StringVar(),
            "turbulence_north": tk.StringVar(),
            "turbulence_east": tk.StringVar(),
            "turbulence_down": tk.StringVar(),
            "turbulence_correlation": tk.StringVar(),
            "wind_seed": tk.StringVar(),
            "gust_start": tk.StringVar(),
            "gust_duration": tk.StringVar(),
            "gust_north": tk.StringVar(),
            "gust_east": tk.StringVar(),
            "gust_down": tk.StringVar(),
            "pilot_profile_path": tk.StringVar(),
            "roll_sensitivity": tk.StringVar(),
            "pitch_sensitivity": tk.StringVar(),
            "yaw_sensitivity": tk.StringVar(),
            "input_expo": tk.StringVar(),
            "analog_deadzone": tk.StringVar(),
            "keyboard_ramp": tk.StringVar(),
            "keyboard_recentering": tk.StringVar(),
        }
        self.aircraft_gamepad_var = tk.BooleanVar(value=True)
        self.aircraft_invert_pitch_var = tk.BooleanVar(value=False)
        self.aircraft_mesh_flip_x_var = tk.BooleanVar(value=False)
        self.aircraft_mesh_flip_y_var = tk.BooleanVar(value=False)
        self.aircraft_mesh_flip_z_var = tk.BooleanVar(value=False)
        self.aircraft_preflight_var = tk.StringVar(value="Preflight values will appear here.")
        self.aircraft_validation_var = tk.StringVar(value="Restore or select a preset to begin.")
        self.catalog_query_var = tk.StringVar()
        self.catalog_distance_var = tk.StringVar()
        self.catalog_method_var = tk.StringVar(value="All methods")
        self.catalog_min_year_var = tk.StringVar()
        self.catalog_max_year_var = tk.StringVar()
        self.catalog_limit_var = tk.StringVar(value="250")
        self.catalog_status_var = tk.StringVar()
        self.catalog_detail_var = tk.StringVar(value="Select a row to inspect reported values.")
        self.project_identity_var = tk.StringVar(value="No engineering project is open.")
        self.project_validation_var = tk.StringVar(
            value="Open a .aerognc.yaml project to browse scenarios and immutable runs."
        )
        self.project_scenario_detail_var = tk.StringVar(
            value="Select a scenario to inspect its purpose and tags."
        )
        self._hint_active = False
        self._status_before_hint = ""

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 14, 24, 10))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="AeroGNC-Lab Simulation Workbench",
            style="HeaderTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Predict motion, play it in 3D, and inspect the engineering evidence | "
                "fictional civilian vehicles and synthetic missions only"
            ),
            style="HeaderSafety.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            style="Muted.TLabel",
        )
        self.status_label.pack(fill="x", padx=26, pady=(4, 8))
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self._scrollable_canvases: dict[str, tk.Canvas] = {}
        self.home_tab = ttk.Frame(self.notebook, padding=18)
        self.project_tab = ttk.Frame(self.notebook, padding=18)
        self.rocket_tab = ttk.Frame(self.notebook, padding=18)
        self.orbit_tab = ttk.Frame(self.notebook, padding=18)
        self.aircraft_tab = ttk.Frame(self.notebook, padding=18)
        self.tour_tab = ttk.Frame(self.notebook, padding=18)
        self.catalog_tab = ttk.Frame(self.notebook, padding=18)
        self.verification_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(self.home_tab, text="Start")
        self.notebook.add(self.rocket_tab, text="Rocket")
        self.notebook.add(self.orbit_tab, text="Satellite Orbit")
        self.notebook.add(self.aircraft_tab, text="Aircraft Flight")
        self.notebook.add(self.tour_tab, text="Planet Trip")
        self.notebook.add(self.project_tab, text="Saved Runs")
        self.notebook.add(self.catalog_tab, text="Astronomy Data")
        self.notebook.add(self.verification_tab, text="Checks")
        self._build_home_tab()
        self._build_project_tab()
        self._build_rocket_tab()
        self._build_orbit_tab()
        self._build_aircraft_tab()
        self._build_tour_tab()
        self._build_catalog_tab()
        self._build_verification_tab()
        self.root.bind("<MouseWheel>", self._scroll_active_page, add="+")

    def _build_home_tab(self) -> None:
        tab, self.home_canvas = self._scrollable_content(self.home_tab)
        tab.columnconfigure((0, 1), weight=1, uniform="home")
        ttk.Label(tab, text="What do you want to find out?", style="Hero.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Label(
            tab,
            text=(
                "For aerospace students, educators, engineers comparing models, and technical "
                "reviewers. Start a prepared example with one click; change inputs only when "
                "you want to ask a different question."
            ),
            style="Muted.TLabel",
            wraplength=900,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 18))

        explanation = ttk.LabelFrame(tab, text="What is a motion solver?", padding=12)
        explanation.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 8))
        explanation.columnconfigure((0, 1, 2), weight=1, uniform="solver")
        for column, heading, body in (
            (
                0,
                "1. YOU PROVIDE",
                "A starting position, speed and orientation, plus a vehicle, environment "
                "and commands.",
            ),
            (
                1,
                "2. IT CALCULATES",
                "The equations of motion are stepped forward in time. Forces and moments "
                "change velocity, position and attitude.",
            ),
            (
                2,
                "3. YOU SEE",
                "A 3D playback, event timeline, performance limits, plots and saved numerical "
                "results that can be checked and repeated.",
            ),
        ):
            cell = ttk.Frame(explanation)
            cell.grid(row=0, column=column, sticky="nsew", padx=8)
            ttk.Label(cell, text=heading, style="Section.TLabel").pack(anchor="w")
            ttk.Label(cell, text=body, wraplength=270, justify="left").pack(anchor="w", pady=(4, 0))

        self._home_card(
            tab,
            3,
            0,
            "How does a rocket move and stay stable?",
            "Predict the 3D path, speed and orientation of a fictional research rocket. "
            "See whether its attitude controller holds the commanded direction.",
            "PLAY ROCKET EXAMPLE",
            self._play_verified_rocket,
            CYAN,
            secondary_text="Change rocket inputs",
            secondary_command=self._open_rocket_inputs,
        )
        self._home_card(
            tab,
            3,
            1,
            "How long will a satellite stay in orbit?",
            "Place a fictional satellite by altitude and speed. Compare force-free, two-body, "
            "three-body, full N-body and atmospheric-decay models in seekable 3D.",
            "PLAY SATELLITE EXAMPLE",
            self._play_verified_orbit,
            GREEN,
            secondary_text="Change orbit inputs",
            secondary_command=self._open_orbit_inputs,
        )
        self._home_card(
            tab,
            4,
            0,
            "What happens when I fly the aircraft?",
            "Fly a fictional research aircraft with keyboard or controller. Lift, drag, pitching "
            "moment, mass, stall and actuator limits all change the calculated path.",
            "FLY AIRCRAFT EXAMPLE",
            self._play_verified_aircraft,
            CYAN,
            secondary_text="Change aircraft inputs",
            secondary_command=self._open_aircraft_inputs,
        )
        self._home_card(
            tab,
            4,
            1,
            "Can a spacecraft complete a three-world trip?",
            "Estimate a fictional departure, transfer, orbit capture, powered departure and "
            "destination capture. See the ideal burns, propellant use and 3D route.",
            "PLAY PLANET TRIP EXAMPLE",
            self._play_verified_tour,
            AMBER,
            secondary_text="Change trip inputs",
            secondary_command=self._open_tour_inputs,
        )

        other = ttk.LabelFrame(
            tab, text="Other tools (not needed for your first simulation)", padding=12
        )
        other.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 0))
        ttk.Label(
            other,
            text=(
                "Saved Runs stores repeatable engineering evidence. Astronomy Data is a "
                "read-only reference catalog, not a trajectory solver. Engineering Checks is "
                "for reviewers. Advanced Designer exposes specialist orbit-analysis controls."
            ),
            style="Muted.TLabel",
            wraplength=890,
        ).pack(anchor="w", pady=(0, 8))
        for label, command in (
            ("OPEN SAVED RUNS", lambda: self._select_tab(self.project_tab)),
            ("BROWSE REFERENCE DATA", lambda: self._select_tab(self.catalog_tab)),
            ("VIEW ENGINEERING CHECKS", lambda: self._select_tab(self.verification_tab)),
            ("OPEN ADVANCED DESIGNER", self._open_advanced_designer),
        ):
            button = ttk.Button(other, text=label, command=command)
            button.pack(side="left", padx=(0, 7))
            self.action_buttons.append(button)
        ttk.Label(
            tab,
            text=(
                "Safety boundary: all executable worlds, spacecraft and epochs are fictional and "
                "synthetic. The NASA catalog is descriptive observational context, not a flight "
                "ephemeris. Target interception and terminal homing are intentionally excluded."
            ),
            style="Safety.TLabel",
            wraplength=900,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(14, 0))

    def _home_card(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        title: str,
        description: str,
        button_text: str,
        command: Callable[[], None],
        accent: str,
        *,
        secondary_text: str | None = None,
        secondary_command: Callable[[], None] | None = None,
    ) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=row, column=column, sticky="nsew", padx=6, pady=5)
        tk.Frame(card, width=5, height=58, background=accent).pack(
            side="left", fill="y", padx=(0, 14)
        )
        body = ttk.Frame(card, style="CardBody.TFrame")
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=title, style="Card.TLabel", font=("Segoe UI Semibold", 13)).pack(
            anchor="w"
        )
        ttk.Label(
            body,
            text=description,
            style="CardMuted.TLabel",
            wraplength=470,
            justify="left",
        ).pack(anchor="w", pady=(5, 10))
        actions = ttk.Frame(body, style="CardBody.TFrame")
        actions.pack(anchor="w")
        button = ttk.Button(actions, text=button_text, style="Success.TButton", command=command)
        button.pack(side="left", padx=(0, 7))
        self.action_buttons.append(button)
        if secondary_text is not None and secondary_command is not None:
            secondary = ttk.Button(actions, text=secondary_text, command=secondary_command)
            secondary.pack(side="left")
            self.action_buttons.append(secondary)

    def _select_tab(self, tab: ttk.Frame) -> None:
        self.notebook.select(tab)  # type: ignore[no-untyped-call]

    def _scrollable_content(self, page: ttk.Frame) -> tuple[ttk.Frame, tk.Canvas]:
        """Return a vertically scrollable content frame for a notebook page."""
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        canvas = tk.Canvas(
            page,
            background=PANEL,
            highlightthickness=0,
            borderwidth=0,
            yscrollincrement=24,
        )
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        content = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=content, anchor="nw")

        def update_region(_event: object) -> None:
            bounds = canvas.bbox("all")
            if bounds is not None:
                canvas.configure(scrollregion=bounds)

        def fill_width(event: tk.Event[tk.Misc]) -> None:
            canvas.itemconfigure(window, width=event.width)

        content.bind("<Configure>", update_region)
        canvas.bind("<Configure>", fill_width)
        self._scrollable_canvases[str(page)] = canvas
        return content, canvas

    def _scroll_active_page(self, event: tk.Event[tk.Misc]) -> None:
        """Route the Windows mouse wheel to the selected scrollable solver page."""
        if isinstance(event.widget, tk.Text) or event.delta == 0:
            return
        selected = str(self.notebook.select())  # type: ignore[no-untyped-call]
        canvas = self._scrollable_canvases.get(selected)
        if canvas is None:
            return
        steps = -1 if event.delta > 0 else 1
        canvas.yview_scroll(steps, "units")

    def _open_rocket_inputs(self) -> None:
        self.rocket_canvas.yview_moveto(0.0)
        self._select_tab(self.rocket_tab)

    def _open_tour_inputs(self) -> None:
        self.tour_canvas.yview_moveto(0.0)
        self._select_tab(self.tour_tab)

    def _open_orbit_inputs(self) -> None:
        self.orbit_canvas.yview_moveto(0.0)
        self._select_tab(self.orbit_tab)

    def _open_aircraft_inputs(self) -> None:
        self.aircraft_canvas.yview_moveto(0.0)
        self._select_tab(self.aircraft_tab)

    def _play_verified_rocket(self) -> None:
        """Restore and play the verified rocket example from the start page."""
        self._reset_rocket()
        self._select_tab(self.rocket_tab)
        self._run_rocket(open_playback=True)

    def _play_verified_tour(self) -> None:
        """Restore and play the verified planetary example from the start page."""
        self._reset_tour()
        self._select_tab(self.tour_tab)
        self._run_tour(open_playback=True)

    def _play_verified_orbit(self) -> None:
        """Restore, propagate, and play the prepared satellite-decay example."""
        self._reset_orbit()
        self._select_tab(self.orbit_tab)
        self._run_orbit(open_playback=True)

    def _play_verified_aircraft(self) -> None:
        """Restore and open the prepared live coefficient-driven aircraft example."""
        self._reset_aircraft()
        self._select_tab(self.aircraft_tab)
        self._fly_aircraft()

    def _field(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        variable: tk.StringVar,
        unit: str,
        help_text: str,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, sticky="ew", padx=(10, 6), pady=5)
        ttk.Label(parent, text=unit, style="Muted.TLabel").grid(
            row=row, column=2, sticky="w", pady=5
        )
        hint = ttk.Label(parent, text="?", style="Muted.TLabel")
        hint.grid(row=row, column=3, sticky="w", padx=(8, 0), pady=5)
        self._bind_hint(entry, help_text)
        self._bind_hint(hint, help_text)
        return entry

    def _bind_hint(self, widget: tk.Misc, help_text: str) -> None:
        widget.bind("<Enter>", lambda _event: self._show_hint(help_text))
        widget.bind("<Leave>", lambda _event: self._clear_hint())

    def _show_hint(self, help_text: str) -> None:
        if self.busy:
            return
        if not self._hint_active:
            self._status_before_hint = self.status_var.get()
        self._hint_active = True
        self.status_var.set(f"Field help: {help_text}")

    def _clear_hint(self) -> None:
        if self._hint_active and not self.busy:
            self.status_var.set(self._status_before_hint)
        self._hint_active = False

    def _build_project_tab(self) -> None:
        tab = self.project_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)
        ttk.Label(
            tab,
            text="Saved, repeatable runs for engineers and reviewers",
            style="Hero.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            tab,
            text=(
                "A project groups prepared scenarios and their evidence. Run one scenario, "
                "open its report, or select two completed runs to measure their differences. "
                "You do not need this page for your first simulation."
            ),
            style="Muted.TLabel",
            wraplength=900,
        ).grid(row=1, column=0, sticky="w", pady=(3, 10))

        toolbar = ttk.Frame(tab)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 9))
        open_button = ttk.Button(toolbar, text="OPEN PROJECT", command=self._open_project_dialog)
        open_button.pack(side="left", padx=(0, 6))
        save_button = ttk.Button(toolbar, text="SAVE", command=self._save_project)
        save_button.pack(side="left", padx=6)
        save_as_button = ttk.Button(toolbar, text="SAVE AS", command=self._save_project_as)
        save_as_button.pack(side="left", padx=6)
        validate_button = ttk.Button(
            toolbar,
            text="VALIDATE",
            style="Primary.TButton",
            command=self._validate_open_project,
        )
        validate_button.pack(side="left", padx=6)
        ttk.Label(toolbar, textvariable=self.project_identity_var, style="Muted.TLabel").pack(
            side="left", padx=14
        )
        self.action_buttons.extend((open_button, save_button, save_as_button, validate_button))

        browser = ttk.Panedwindow(tab, orient="vertical")
        browser.grid(row=3, column=0, sticky="nsew")
        scenario_frame = ttk.LabelFrame(browser, text="Scenarios", padding=9)
        history_frame = ttk.LabelFrame(browser, text="Run history and comparison", padding=9)
        browser.add(scenario_frame, weight=2)
        browser.add(history_frame, weight=3)

        scenario_frame.columnconfigure(0, weight=1)
        scenario_frame.rowconfigure(1, weight=1)
        ttk.Label(
            scenario_frame,
            textvariable=self.project_validation_var,
            style="Safety.TLabel",
            wraplength=900,
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        scenario_columns = ("name", "workflow", "configuration", "enabled", "seed")
        self.project_scenario_tree = ttk.Treeview(
            scenario_frame,
            columns=scenario_columns,
            show="headings",
            height=6,
            selectmode="browse",
        )
        scenario_headings = (
            ("name", "Scenario", 170),
            ("workflow", "Workflow", 150),
            ("configuration", "Configuration", 300),
            ("enabled", "Enabled", 80),
            ("seed", "Seed", 110),
        )
        for name, label, width in scenario_headings:
            self.project_scenario_tree.heading(name, text=label)
            self.project_scenario_tree.column(name, width=width, minwidth=65, anchor="w")
        self.project_scenario_tree.grid(row=1, column=0, sticky="nsew")
        scenario_scroll = ttk.Scrollbar(
            scenario_frame, orient="vertical", command=self.project_scenario_tree.yview
        )
        scenario_scroll.grid(row=1, column=1, sticky="ns")
        self.project_scenario_tree.configure(yscrollcommand=scenario_scroll.set)
        self.project_scenario_tree.bind("<<TreeviewSelect>>", self._project_scenario_selected)
        ttk.Label(
            scenario_frame,
            textvariable=self.project_scenario_detail_var,
            style="Muted.TLabel",
            wraplength=900,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 4))
        scenario_actions = ttk.Frame(scenario_frame)
        scenario_actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        run_button = ttk.Button(
            scenario_actions,
            text="RUN SELECTED SCENARIO",
            style="Success.TButton",
            command=self._run_project_scenario,
        )
        run_button.pack(side="left")
        self.project_cancel_button = ttk.Button(
            scenario_actions,
            text="CANCEL RUN",
            command=self._cancel_project_run,
            state="disabled",
        )
        self.project_cancel_button.pack(side="left", padx=8)
        refresh_button = ttk.Button(
            scenario_actions, text="REFRESH HISTORY", command=self._refresh_project
        )
        refresh_button.pack(side="left", padx=8)
        self.action_buttons.extend((run_button, refresh_button))

        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        history_columns = ("created", "scenario", "workflow", "status", "run_id")
        self.project_history_tree = ttk.Treeview(
            history_frame,
            columns=history_columns,
            show="headings",
            height=7,
            selectmode="extended",
        )
        history_headings = (
            ("created", "Created UTC", 185),
            ("scenario", "Scenario", 155),
            ("workflow", "Workflow", 135),
            ("status", "Status", 85),
            ("run_id", "Run ID", 300),
        )
        for name, label, width in history_headings:
            self.project_history_tree.heading(name, text=label)
            self.project_history_tree.column(name, width=width, minwidth=65, anchor="w")
        self.project_history_tree.grid(row=0, column=0, sticky="nsew")
        history_scroll = ttk.Scrollbar(
            history_frame, orient="vertical", command=self.project_history_tree.yview
        )
        history_scroll.grid(row=0, column=1, sticky="ns")
        self.project_history_tree.configure(yscrollcommand=history_scroll.set)
        history_actions = ttk.Frame(history_frame)
        history_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 5))
        compare_button = ttk.Button(
            history_actions,
            text="COMPARE TWO SELECTED RUNS",
            style="Primary.TButton",
            command=self._compare_project_runs,
        )
        compare_button.pack(side="left")
        report_button = ttk.Button(
            history_actions, text="OPEN SELECTED REPORT", command=self._open_project_report
        )
        report_button.pack(side="left", padx=8)
        comparison_report_button = ttk.Button(
            history_actions,
            text="OPEN LAST COMPARISON",
            command=self._open_last_comparison_report,
        )
        comparison_report_button.pack(side="left", padx=8)
        self.action_buttons.extend((compare_button, report_button, comparison_report_button))
        self.project_comparison_text = tk.Text(
            history_frame,
            height=5,
            background="#081621",
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Cascadia Mono", 9),
            wrap="word",
        )
        self.project_comparison_text.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        self._replace_text(
            self.project_comparison_text,
            "Select exactly two completed runs to calculate same-unit differences.",
        )

    def _build_rocket_tab(self) -> None:
        tab, self.rocket_canvas = self._scrollable_content(self.rocket_tab)
        tab.columnconfigure((0, 1), weight=1, uniform="rocket")
        ttk.Label(
            tab, text="Question: how will this rocket move and point?", style="Hero.TLabel"
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            tab,
            text=(
                "New here? Leave the prepared values unchanged and click the green button. "
                "The solver predicts the fictional rocket's 3D translation and rotation."
            ),
            style="Muted.TLabel",
            wraplength=800,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 12))

        essential = ttk.LabelFrame(tab, text="Basic inputs", padding=14)
        essential.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        essential.columnconfigure(1, weight=1)
        self._field(
            essential,
            0,
            "How long to calculate",
            self.rocket_vars["duration"],
            "s",
            "Model time from launch; verified range is 0.25 to 8 seconds",
        )
        self._field(
            essential,
            1,
            "Starting speed",
            self.rocket_vars["speed"],
            "m/s",
            "Speed when the modeled rocket leaves its launch rail",
        )
        self._field(
            essential,
            2,
            "3D playback speed",
            self.rocket_vars["playback"],
            "times",
            "Two means the animation advances two model seconds per real second",
        )

        meaning = ttk.LabelFrame(tab, text="What the calculation will show", padding=14)
        meaning.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        for row, heading, body in (
            (
                0,
                "MOTION",
                "3D position, total and vertical speed, and acceleration over time.",
            ),
            (
                1,
                "STABILITY",
                "Orientation, body rotation rates, and commanded-versus-actual error.",
            ),
            (
                2,
                "LOADS AND EVENTS",
                "Aerodynamic angles, Mach, dynamic pressure, mass, commands, and burnout.",
            ),
        ):
            ttk.Label(meaning, text=heading, style="Section.TLabel").grid(
                row=2 * row, column=0, sticky="w"
            )
            ttk.Label(meaning, text=body, wraplength=400, justify="left").grid(
                row=2 * row + 1, column=0, sticky="w", pady=(2, 7)
            )

        disclosure = ttk.Frame(tab, padding=(0, 10, 0, 4))
        disclosure.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.rocket_advanced_button = ttk.Button(
            disclosure,
            text="SHOW ADVANCED ORIENTATION AND NUMERICAL INPUTS",
            command=self._toggle_rocket_advanced,
        )
        self.rocket_advanced_button.pack(side="left")
        ttk.Label(
            disclosure,
            text="Optional - the prepared values are already ready to run.",
            style="Muted.TLabel",
        ).pack(side="left", padx=10)
        self.action_buttons.append(self.rocket_advanced_button)

        self.rocket_advanced_frame = ttk.Frame(tab)
        self.rocket_advanced_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.rocket_advanced_frame.columnconfigure((0, 1), weight=1, uniform="rocket-advanced")
        numerical = ttk.LabelFrame(
            self.rocket_advanced_frame, text="Numerical accuracy", padding=14
        )
        numerical.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        numerical.columnconfigure(1, weight=1)
        self._field(
            numerical,
            0,
            "RK4 calculation step",
            self.rocket_vars["step"],
            "s",
            "Smaller steps usually improve numerical accuracy but take longer; maximum 0.02 s",
        )
        ttk.Label(
            numerical,
            text=(
                "This is the solver's time increment, not the animation frame rate. "
                "Change it only for a convergence study."
            ),
            style="Muted.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))

        attitude = ttk.LabelFrame(
            self.rocket_advanced_frame, text="Starting orientation and rotation", padding=14
        )
        attitude.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        attitude.columnconfigure(1, weight=1)
        self._field(
            attitude,
            0,
            "Roll",
            self.rocket_vars["roll"],
            "deg",
            "Positive right-wing-down rotation",
        )
        self._field(
            attitude,
            1,
            "Pitch",
            self.rocket_vars["pitch"],
            "deg",
            "Nominal near-vertical attitude is 86 deg",
        )
        self._field(
            attitude, 2, "Yaw", self.rocket_vars["yaw"], "deg", "Heading angle in the NED frame"
        )
        self._field(
            attitude,
            3,
            "Roll rate",
            self.rocket_vars["roll_rate"],
            "deg/s",
            "Initial body x-axis angular rate",
        )
        self._field(
            attitude,
            4,
            "Pitch rate",
            self.rocket_vars["pitch_rate"],
            "deg/s",
            "Initial body y-axis angular rate",
        )
        self._field(
            attitude,
            5,
            "Yaw rate",
            self.rocket_vars["yaw_rate"],
            "deg/s",
            "Initial body z-axis angular rate",
        )
        ttk.Label(
            attitude,
            text=(
                "Aerospace 3-2-1 roll-pitch-yaw; NED navigation frame and "
                "forward-right-down body axes."
            ),
            style="Muted.TLabel",
            wraplength=350,
            justify="left",
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.rocket_advanced_visible = False
        self.rocket_advanced_frame.grid_remove()

        controls = ttk.Frame(tab, padding=(0, 12, 0, 8))
        controls.grid(row=5, column=0, columnspan=2, sticky="ew")
        play = ttk.Button(
            controls,
            text="PLAY THIS ROCKET IN 3D",
            style="Success.TButton",
            command=lambda: self._run_rocket(open_playback=True),
        )
        play.pack(side="left", padx=(0, 7))
        run = ttk.Button(
            controls,
            text="CALCULATE + SAVE ONLY",
            style="Primary.TButton",
            command=lambda: self._run_rocket(open_playback=False),
        )
        run.pack(side="left", padx=7)
        reset = ttk.Button(controls, text="RESTORE EXAMPLE", command=self._reset_rocket)
        reset.pack(side="left", padx=7)
        self.action_buttons.extend((play, run, reset))
        result_frame = ttk.LabelFrame(tab, text="What happened in the last run", padding=10)
        result_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")
        tab.rowconfigure(6, weight=1)
        self.rocket_summary = tk.Text(
            result_frame,
            height=8,
            background="#081621",
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Cascadia Mono", 9),
            wrap="word",
        )
        rocket_scrollbar = ttk.Scrollbar(
            result_frame,
            orient="vertical",
            command=self.rocket_summary.yview,
        )
        rocket_scrollbar.pack(side="right", fill="y")
        self.rocket_summary.configure(yscrollcommand=rocket_scrollbar.set)
        self.rocket_summary.pack(side="left", fill="both", expand=True)
        self._replace_text(
            self.rocket_summary,
            (
                "NOTHING HAS BEEN CALCULATED YET\n\n"
                "Leave the prepared values unchanged and click the green button. The 3D window "
                "will show the computed path and orientation; this panel will then explain the "
                "highest altitude, speed, stability error and detected events."
            ),
        )

    def _toggle_rocket_advanced(self) -> None:
        self.rocket_advanced_visible = not self.rocket_advanced_visible
        if self.rocket_advanced_visible:
            self.rocket_advanced_frame.grid()
            self.rocket_advanced_button.configure(text="HIDE ADVANCED INPUTS")
        else:
            self.rocket_advanced_frame.grid_remove()
            self.rocket_advanced_button.configure(
                text="SHOW ADVANCED ORIENTATION AND NUMERICAL INPUTS"
            )

    def _build_orbit_tab(self) -> None:
        tab, self.orbit_canvas = self._scrollable_content(self.orbit_tab)
        tab.columnconfigure((0, 1), weight=1, uniform="orbit")
        ttk.Label(
            tab,
            text="Question: what path will this satellite follow, and for how long?",
            style="Hero.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            tab,
            text=(
                "Choose the physical model first. The prepared case starts 200 km above a "
                "fictional planet and predicts when it crosses the 120 km reentry boundary."
            ),
            style="Muted.TLabel",
            wraplength=900,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 12))

        basic = ttk.LabelFrame(tab, text="Basic orbit question", padding=14)
        basic.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        basic.columnconfigure(1, weight=1)
        ttk.Label(basic, text="Physics model").grid(row=0, column=0, sticky="w", pady=5)
        model_box = ttk.Combobox(
            basic,
            textvariable=self.orbit_vars["model"],
            values=tuple(ORBIT_MODEL_CHOICES),
            state="readonly",
            width=44,
        )
        model_box.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 6), pady=5)
        self._bind_hint(
            model_box,
            "Two body is the first true orbit. The one-moving-body case applies no force and "
            "therefore travels in a straight line.",
        )
        self._field(
            basic,
            1,
            "Starting height above planet",
            self.orbit_vars["altitude"],
            "km",
            "Radial height above the fictional planet's reference surface",
        )
        ttk.Label(basic, text="Starting speed rule").grid(row=2, column=0, sticky="w", pady=5)
        speed_box = ttk.Combobox(
            basic,
            textvariable=self.orbit_vars["speed_mode"],
            values=tuple(ORBIT_SPEED_CHOICES),
            state="readonly",
            width=31,
        )
        speed_box.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(10, 6), pady=5)
        self._field(
            basic,
            3,
            "Custom speed (used only if selected)",
            self.orbit_vars["custom_speed"],
            "m/s",
            "Tangential initial speed; ignored for calculated circular or escape speed",
        )
        self._field(
            basic,
            4,
            "Orbit tilt",
            self.orbit_vars["inclination"],
            "deg",
            "Zero is equatorial; 90 degrees crosses over the poles",
        )
        self._field(
            basic,
            5,
            "How long to test survival",
            self.orbit_vars["duration_days"],
            "days",
            "Finite prediction horizon, from minutes up to 366 modeled days",
        )

        meaning = ttk.LabelFrame(tab, text="What each model answers", padding=14)
        meaning.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        for row, heading, body in (
            (
                0,
                "FREE / 1 MOVING BODY",
                "A verification control: no gravity, so velocity is constant and the path is "
                "straight. It is not an orbit.",
            ),
            (
                1,
                "2-BODY / 3-BODY / N-BODY",
                "Adds one central gravity source, then a prescribed moon, then pairwise gravity "
                "between every configured body.",
            ),
            (
                2,
                "ORBIT LIFETIME",
                "Adds oblateness (J2), rotating atmosphere, area, Cd and mass. It reports a "
                "threshold time or only a finite lower bound on lifetime.",
            ),
        ):
            ttk.Label(meaning, text=heading, style="Section.TLabel").grid(
                row=2 * row, column=0, sticky="w"
            )
            ttk.Label(meaning, text=body, wraplength=410, justify="left").grid(
                row=2 * row + 1, column=0, sticky="w", pady=(2, 8)
            )
        ttk.Label(
            meaning,
            text=(
                "Lifetime is sensitive to thermospheric density. This fixed synthetic reference "
                "does not forecast real space weather."
            ),
            style="Safety.TLabel",
            wraplength=390,
        ).grid(row=6, column=0, sticky="w", pady=(7, 0))

        disclosure = ttk.Frame(tab, padding=(0, 10, 0, 4))
        disclosure.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.orbit_advanced_button = ttk.Button(
            disclosure,
            text="SHOW SATELLITE, DRAG, CORRECTION AND NUMERICAL INPUTS",
            command=self._toggle_orbit_advanced,
        )
        self.orbit_advanced_button.pack(side="left")
        ttk.Label(
            disclosure,
            text="Optional - use these to study why lifetime changes.",
            style="Muted.TLabel",
        ).pack(side="left", padx=10)
        self.action_buttons.append(self.orbit_advanced_button)

        self.orbit_advanced_frame = ttk.Frame(tab)
        self.orbit_advanced_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.orbit_advanced_frame.columnconfigure((0, 1), weight=1, uniform="orbit-advanced")
        satellite = ttk.LabelFrame(self.orbit_advanced_frame, text="Satellite and drag", padding=14)
        satellite.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        satellite.columnconfigure(1, weight=1)
        self._field(
            satellite,
            0,
            "Initial total mass",
            self.orbit_vars["mass"],
            "kg",
            "A heavier satellite decelerates less for the same Cd and area",
        )
        self._field(
            satellite,
            1,
            "Dry mass",
            self.orbit_vars["dry_mass"],
            "kg",
            "Correction burns may not consume mass below this limit",
        )
        self._field(
            satellite,
            2,
            "Area facing the flow",
            self.orbit_vars["area"],
            "m2",
            "Reference drag area; larger area normally shortens low-orbit lifetime",
        )
        self._field(
            satellite,
            3,
            "Drag coefficient Cd",
            self.orbit_vars["drag_coefficient"],
            "-",
            "Dimensionless drag coefficient used directly by the force model",
        )
        self._field(
            satellite,
            4,
            "Atmosphere density multiplier",
            self.orbit_vars["density_scale"],
            "times",
            "Sensitivity factor: zero disables density; one uses the reference table",
        )

        lifetime = ttk.LabelFrame(
            self.orbit_advanced_frame, text="Lifetime boundary and optional correction", padding=14
        )
        lifetime.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        lifetime.columnconfigure(1, weight=1)
        self._field(
            lifetime,
            0,
            "Reentry threshold",
            self.orbit_vars["reentry_altitude"],
            "km",
            "The run stops when descending altitude crosses this defined boundary",
        )
        self._field(
            lifetime,
            1,
            "RK4 integration step",
            self.orbit_vars["integration_step"],
            "s",
            "Smaller is more expensive; output step must be an exact multiple",
        )
        self._field(
            lifetime,
            2,
            "Saved output spacing",
            self.orbit_vars["output_step"],
            "s",
            "Spacing of CSV/3D playback samples, not the integration step",
        )
        correction = ttk.Checkbutton(
            lifetime,
            text="Enable idealized perigee correction burns",
            variable=self.orbit_correction_var,
        )
        correction.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 4))
        self._field(
            lifetime,
            4,
            "Correct below perigee altitude",
            self.orbit_vars["correction_altitude"],
            "km",
            "At a low perigee, an ideal instantaneous tangential recircularization is attempted",
        )
        self._field(
            lifetime,
            5,
            "Maximum correction burns",
            self.orbit_vars["maximum_corrections"],
            "burns",
            "Whole-number safety cap; zero prevents all burns",
        )
        self.orbit_advanced_visible = False
        self.orbit_advanced_frame.grid_remove()

        controls = ttk.Frame(tab, padding=(0, 12, 0, 8))
        controls.grid(row=5, column=0, columnspan=2, sticky="ew")
        play = ttk.Button(
            controls,
            text="CALCULATE + PLAY ORBIT IN 3D",
            style="Success.TButton",
            command=lambda: self._run_orbit(open_playback=True),
        )
        play.pack(side="left", padx=(0, 7))
        calculate = ttk.Button(
            controls,
            text="CALCULATE + SAVE ONLY",
            style="Primary.TButton",
            command=lambda: self._run_orbit(open_playback=False),
        )
        calculate.pack(side="left", padx=7)
        reset = ttk.Button(controls, text="RESTORE EXAMPLE", command=self._reset_orbit)
        reset.pack(side="left", padx=7)
        self.action_buttons.extend((play, calculate, reset))
        result_frame = ttk.LabelFrame(tab, text="What happened in the last orbit run", padding=10)
        result_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.orbit_summary = tk.Text(
            result_frame,
            height=8,
            background="#081621",
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Cascadia Mono", 9),
            wrap="word",
        )
        orbit_scrollbar = ttk.Scrollbar(
            result_frame, orient="vertical", command=self.orbit_summary.yview
        )
        orbit_scrollbar.pack(side="right", fill="y")
        self.orbit_summary.configure(yscrollcommand=orbit_scrollbar.set)
        self.orbit_summary.pack(side="left", fill="both", expand=True)
        self._replace_text(
            self.orbit_summary,
            "NOTHING HAS BEEN CALCULATED YET\n\nChoose a model and click the green button. "
            "The result will state whether reentry occurred, how many modeled revolutions were "
            "completed, and what the selected physics includes.",
        )

    def _toggle_orbit_advanced(self) -> None:
        self.orbit_advanced_visible = not self.orbit_advanced_visible
        if self.orbit_advanced_visible:
            self.orbit_advanced_frame.grid()
            self.orbit_advanced_button.configure(text="HIDE OPTIONAL ORBIT INPUTS")
        else:
            self.orbit_advanced_frame.grid_remove()
            self.orbit_advanced_button.configure(
                text="SHOW SATELLITE, DRAG, CORRECTION AND NUMERICAL INPUTS"
            )

    def _build_aircraft_tab(self) -> None:
        tab, self.aircraft_canvas = self._scrollable_content(self.aircraft_tab)
        tab.columnconfigure((0, 1), weight=1, uniform="aircraft")
        ttk.Label(
            tab,
            text="Fly a nonlinear aircraft, inspect every cause, then replay the exact states",
            style="Hero.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            tab,
            text=(
                "Start in three steps: 1) choose a preset, 2) check the green preflight card, "
                "3) click FLY LIVE and press Space in the flight window. Every control moves "
                "the actual 18-state model; F9 saves exact states and a debrief."
            ),
            style="Muted.TLabel",
            wraplength=980,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 14))

        quick = ttk.LabelFrame(tab, text="1. Quick Flight", padding=14)
        quick.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        quick.columnconfigure(1, weight=1)
        ttk.Label(quick, text="Exercise preset").grid(row=0, column=0, sticky="w", pady=5)
        preset = ttk.Combobox(
            quick,
            textvariable=self.aircraft_vars["preset"],
            values=tuple(AIRCRAFT_PRESET_CHOICES),
            state="readonly",
            width=36,
        )
        preset.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 6), pady=5)
        apply_preset = ttk.Button(quick, text="APPLY", command=self._apply_aircraft_preset)
        apply_preset.grid(row=0, column=3, sticky="e", pady=5)
        self._field(
            quick,
            1,
            "Starting altitude",
            self.aircraft_vars["altitude"],
            "m",
            "Height above the fictional spherical planet surface",
        )
        self._field(
            quick,
            2,
            "Starting true airspeed",
            self.aircraft_vars["airspeed"],
            "m/s",
            "Speed relative to the moving atmosphere; preflight compares it with 1-g stall speed",
        )
        self._field(
            quick,
            3,
            "Starting heading",
            self.aircraft_vars["heading"],
            "deg",
            "Clockwise from local north; 90 degrees points east",
        )
        self._field(
            quick,
            4,
            "Starting throttle",
            self.aircraft_vars["throttle"],
            "0 to 1",
            "Zero is idle and one requests maximum available air-breathing thrust",
        )
        self._field(
            quick,
            5,
            "Hands-off batch duration",
            self.aircraft_vars["duration"],
            "s",
            "Used by Calculate + Save; live flight has a separate one-hour safety limit",
        )
        self._field(
            quick,
            6,
            "Live speed factor",
            self.aircraft_vars["real_time_factor"],
            "sim s / real s",
            "One is real time; fixed-step physics remains independent of display FPS",
        )
        ttk.Label(quick, text="Control mode").grid(row=7, column=0, sticky="w", pady=5)
        control_mode = ttk.Combobox(
            quick,
            textvariable=self.aircraft_vars["control_mode"],
            values=tuple(AIRCRAFT_CONTROL_MODE_CHOICES),
            state="readonly",
            width=36,
        )
        control_mode.grid(row=7, column=1, columnspan=3, sticky="ew", padx=(10, 0), pady=5)
        ttk.Label(quick, text="Starting camera").grid(row=8, column=0, sticky="w", pady=5)
        camera = ttk.Combobox(
            quick,
            textvariable=self.aircraft_vars["camera"],
            values=tuple(AIRCRAFT_CAMERA_CHOICES),
            state="readonly",
            width=36,
        )
        camera.grid(row=8, column=1, columnspan=3, sticky="ew", padx=(10, 0), pady=5)

        preflight_card = ttk.LabelFrame(tab, text="2. What this solver will calculate", padding=14)
        preflight_card.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        ttk.Label(
            preflight_card,
            text=(
                "Your inputs define gravity, atmosphere, wind, thrust, fuel use, mass, inertia, "
                "CL/CD/Cm and control surfaces. Fixed-step RK4 propagates position, velocity, "
                "quaternion attitude, body rates, mass, actuators and throttle. The flight deck "
                "shows the resulting path, forces, stall margin and warnings—not a canned "
                "animation."
            ),
            wraplength=430,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(
            preflight_card,
            textvariable=self.aircraft_preflight_var,
            style="Safety.TLabel",
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(14, 8))
        ttk.Label(
            preflight_card,
            textvariable=self.aircraft_validation_var,
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        ttk.Label(
            preflight_card,
            text=(
                "PUBLIC-SAFE LIMIT: fictional civilian vehicle, synthetic data, no classified "
                "or proprietary source, no interception or terminal homing. Stall is a scoped "
                "coefficient-break model; spins, structural failure, landing gear and heating "
                "remain outside the model."
            ),
            style="Safety.TLabel",
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        physical = ttk.LabelFrame(tab, text="3. Aircraft Physics", padding=14)
        physical.grid(row=3, column=0, sticky="nsew", padx=(0, 7), pady=(10, 0))
        physical.columnconfigure(1, weight=1)
        for row, label, key, unit, help_text in (
            (0, "Initial mass", "mass", "kg", "Direct divisor in translational acceleration"),
            (1, "Dry mass", "dry_mass", "kg", "Hard floor for fuel depletion"),
            (2, "Wing reference area", "wing_area", "m2", "Scales every aerodynamic force/moment"),
            (3, "Lift at zero AoA (CL0)", "cl_zero", "-", "Baseline lift coefficient"),
            (4, "Lift slope (CL alpha)", "cl_alpha", "1/rad", "Pre-stall lift slope"),
            (5, "Maximum lift (CL max)", "cl_maximum", "-", "Lift cap at stall onset"),
            (6, "Stall angle", "stall_angle", "deg", "Begins lift loss and stall drag"),
            (7, "Zero-lift drag (CD0)", "cd_zero", "-", "Parasitic drag coefficient"),
            (8, "Induced-drag factor", "induced_drag", "-", "Multiplies CL squared"),
            (
                9,
                "Pitch stability (Cm alpha)",
                "pitch_alpha",
                "1/rad",
                "Negative is statically restoring",
            ),
            (10, "Starting angle of attack", "alpha", "deg", "Body-to-airflow angle"),
            (11, "Starting climb angle", "flight_path_angle", "deg", "Positive climbs"),
            (12, "Starting bank angle", "bank", "deg", "Positive right-wing-down"),
        ):
            self._field(physical, row, label, self.aircraft_vars[key], unit, help_text)

        environment = ttk.LabelFrame(tab, text="4. Environment", padding=14)
        environment.grid(row=3, column=1, sticky="nsew", padx=(7, 0), pady=(10, 0))
        environment.columnconfigure(1, weight=1)
        for row, label, key, unit, help_text in (
            (0, "Steady wind north", "wind_north", "m/s", "Positive toward local north"),
            (1, "Steady wind east", "wind_east", "m/s", "Positive toward local east"),
            (2, "Turbulence std north", "turbulence_north", "m/s", "Seeded Gauss-Markov"),
            (3, "Turbulence std east", "turbulence_east", "m/s", "Seeded Gauss-Markov"),
            (4, "Turbulence std down", "turbulence_down", "m/s", "Positive body-down NED"),
            (5, "Turbulence correlation", "turbulence_correlation", "s", "Correlation time"),
            (6, "Random seed", "wind_seed", "integer", "Makes disturbance reproducible"),
            (7, "Gust start", "gust_start", "s", "Start of smooth finite pulse"),
            (8, "Gust duration", "gust_duration", "s", "Zero-peak-zero duration"),
            (9, "Gust north", "gust_north", "m/s", "1-cosine pulse amplitude"),
            (10, "Gust east", "gust_east", "m/s", "1-cosine pulse amplitude"),
            (11, "Gust down", "gust_down", "m/s", "1-cosine pulse amplitude"),
        ):
            self._field(environment, row, label, self.aircraft_vars[key], unit, help_text)
        ttk.Label(
            environment,
            text=(
                "Steady wind, seeded turbulence and the discrete gust are separate. The seed "
                "makes repeated runs identical; this is not a weather forecast."
            ),
            style="Muted.TLabel",
            wraplength=410,
            justify="left",
        ).grid(row=12, column=0, columnspan=4, sticky="w", pady=(12, 0))

        control_feel = ttk.LabelFrame(tab, text="5. Control Feel and Reusable Profile", padding=14)
        control_feel.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        control_feel.columnconfigure(1, weight=1)
        ttk.Label(control_feel, text="Pilot profile JSON").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_feel, textvariable=self.aircraft_vars["pilot_profile_path"]).grid(
            row=0, column=1, sticky="ew", padx=(10, 6), pady=5
        )
        choose_profile = ttk.Button(
            control_feel, text="CHOOSE...", command=self._choose_aircraft_pilot_profile
        )
        choose_profile.grid(row=0, column=2, padx=3)
        load_profile = ttk.Button(
            control_feel, text="LOAD", command=self._load_aircraft_pilot_profile
        )
        load_profile.grid(row=0, column=3, padx=3)
        save_profile = ttk.Button(
            control_feel, text="SAVE AS...", command=self._save_aircraft_pilot_profile
        )
        save_profile.grid(row=0, column=4, padx=3)
        response = ttk.Frame(control_feel)
        response.grid(row=1, column=0, columnspan=5, sticky="ew", pady=5)
        response.columnconfigure(tuple(range(8)), weight=1)
        for column, (label, key) in enumerate(
            (
                ("Roll sensitivity", "roll_sensitivity"),
                ("Pitch sensitivity", "pitch_sensitivity"),
                ("Yaw sensitivity", "yaw_sensitivity"),
                ("Expo", "input_expo"),
                ("Deadzone", "analog_deadzone"),
                ("Keyboard ramp /s", "keyboard_ramp"),
                ("Recentre /s", "keyboard_recentering"),
            )
        ):
            ttk.Label(response, text=label).grid(row=0, column=column, padx=4)
            ttk.Entry(response, textvariable=self.aircraft_vars[key], width=9).grid(
                row=1, column=column, padx=4
            )
        ttk.Checkbutton(
            response,
            text="Invert pitch",
            variable=self.aircraft_invert_pitch_var,
        ).grid(row=1, column=7, padx=8)
        ttk.Label(
            control_feel,
            text=(
                "Profiles also contain editable key bindings, trim, damping and attitude-hold "
                "gains. Save a copy, edit the readable JSON if needed, then LOAD it here."
            ),
            style="Muted.TLabel",
            wraplength=900,
        ).grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))

        appearance = ttk.LabelFrame(tab, text="6. Appearance, Trail and Recording", padding=14)
        appearance.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        appearance.columnconfigure(1, weight=1)
        ttk.Label(appearance, text="OBJ or STL model").grid(row=0, column=0, sticky="w", pady=5)
        mesh_entry = ttk.Entry(appearance, textvariable=self.aircraft_vars["mesh_path"])
        mesh_entry.grid(row=0, column=1, sticky="ew", padx=(10, 6), pady=5)
        choose_mesh = ttk.Button(appearance, text="CHOOSE...", command=self._choose_aircraft_mesh)
        choose_mesh.grid(row=0, column=2, sticky="e", padx=4)
        preview_mesh = ttk.Button(appearance, text="PREVIEW", command=self._preview_aircraft_mesh)
        preview_mesh.grid(row=0, column=3, sticky="e", padx=4)
        ttk.Label(appearance, text="Source axes").grid(row=1, column=0, sticky="w", pady=5)
        mesh_axes = ttk.Combobox(
            appearance,
            textvariable=self.aircraft_vars["mesh_axes"],
            values=tuple(MESH_AXIS_CHOICES),
            state="readonly",
            width=44,
        )
        mesh_axes.grid(row=1, column=1, sticky="ew", padx=(10, 6), pady=5)
        transform_frame = ttk.Frame(appearance)
        transform_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)
        for column, (label, key) in enumerate(
            (
                ("Rotate X", "mesh_rotation_x"),
                ("Rotate Y", "mesh_rotation_y"),
                ("Rotate Z", "mesh_rotation_z"),
            )
        ):
            ttk.Label(transform_frame, text=f"{label} [deg]").grid(row=0, column=2 * column, padx=3)
            ttk.Entry(transform_frame, textvariable=self.aircraft_vars[key], width=8).grid(
                row=0, column=2 * column + 1, padx=3
            )
        ttk.Checkbutton(
            transform_frame, text="Flip X", variable=self.aircraft_mesh_flip_x_var
        ).grid(row=0, column=6, padx=6)
        ttk.Checkbutton(
            transform_frame, text="Flip Y", variable=self.aircraft_mesh_flip_y_var
        ).grid(row=0, column=7, padx=6)
        ttk.Checkbutton(
            transform_frame, text="Flip Z", variable=self.aircraft_mesh_flip_z_var
        ).grid(row=0, column=8, padx=6)
        ttk.Label(appearance, text="Center / visual scale").grid(row=3, column=0, sticky="w")
        centre = ttk.Combobox(
            appearance,
            textvariable=self.aircraft_vars["mesh_center"],
            values=("Centroid", "Bounds centre", "Keep source origin"),
            state="readonly",
            width=20,
        )
        centre.grid(row=3, column=1, sticky="w", padx=(10, 6), pady=5)
        scale = ttk.Combobox(
            appearance,
            textvariable=self.aircraft_vars["mesh_scale_mode"],
            values=("Enlarged visible marker", "True physical scale"),
            state="readonly",
            width=24,
        )
        scale.grid(row=3, column=2, columnspan=2, sticky="w", padx=4, pady=5)
        ttk.Label(appearance, text="Trail").grid(row=4, column=0, sticky="w", pady=5)
        trail = ttk.Combobox(
            appearance,
            textvariable=self.aircraft_vars["trail_mode"],
            values=tuple(AIRCRAFT_TRAIL_CHOICES),
            state="readonly",
            width=24,
        )
        trail.grid(row=4, column=1, sticky="w", padx=(10, 6), pady=5)
        ttk.Entry(appearance, textvariable=self.aircraft_vars["trail_duration"], width=8).grid(
            row=4, column=2, sticky="w", padx=4
        )
        ttk.Label(appearance, text="seconds when fading").grid(row=4, column=3, sticky="w")
        ttk.Label(appearance, text="Trail colour").grid(row=5, column=0, sticky="w", pady=5)
        trail_color = ttk.Combobox(
            appearance,
            textvariable=self.aircraft_vars["trail_color"],
            values=tuple(AIRCRAFT_TRAIL_COLOR_CHOICES),
            state="readonly",
            width=24,
        )
        trail_color.grid(row=5, column=1, sticky="w", padx=(10, 6), pady=5)
        ttk.Checkbutton(
            appearance, text="Enable Windows XInput controller", variable=self.aircraft_gamepad_var
        ).grid(row=5, column=2, columnspan=2, sticky="w", padx=4)
        ttk.Label(appearance, text="Recording folder").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(appearance, textvariable=self.aircraft_vars["recorder_directory"]).grid(
            row=6, column=1, columnspan=3, sticky="ew", padx=(10, 0), pady=5
        )
        ttk.Label(
            appearance,
            text=(
                "Imported geometry is visual only. It never changes mass, inertia, CL, CD, Cm "
                "or propulsion. PREVIEW shows axes, dimensions and live decimation before flight."
            ),
            style="Muted.TLabel",
            wraplength=900,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

        controls = ttk.Frame(tab, padding=(0, 14, 0, 8))
        controls.grid(row=6, column=0, columnspan=2, sticky="ew")
        fly = ttk.Button(
            controls,
            text="FLY LIVE 3D",
            style="Success.TButton",
            command=self._fly_aircraft,
        )
        fly.pack(side="left", padx=(0, 7))
        calculate = ttk.Button(
            controls,
            text="RUN HANDS-OFF + SAVE",
            style="Primary.TButton",
            command=self._run_aircraft_batch,
        )
        calculate.pack(side="left", padx=7)
        reset = ttk.Button(controls, text="RESTORE EXAMPLE", command=self._reset_aircraft)
        reset.pack(side="left", padx=7)
        self.action_buttons.extend(
            (
                apply_preset,
                choose_profile,
                load_profile,
                save_profile,
                choose_mesh,
                preview_mesh,
                fly,
                calculate,
                reset,
            )
        )
        result_frame = ttk.LabelFrame(
            tab, text="Last hands-off result / saved evidence", padding=10
        )
        result_frame.grid(row=7, column=0, columnspan=2, sticky="nsew")
        self.aircraft_summary = tk.Text(
            result_frame,
            height=8,
            background="#081621",
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Cascadia Mono", 9),
            wrap="word",
        )
        aircraft_scrollbar = ttk.Scrollbar(
            result_frame, orient="vertical", command=self.aircraft_summary.yview
        )
        aircraft_scrollbar.pack(side="right", fill="y")
        self.aircraft_summary.configure(yscrollcommand=aircraft_scrollbar.set)
        self.aircraft_summary.pack(side="left", fill="both", expand=True)
        self._replace_text(
            self.aircraft_summary,
            "NOTHING HAS BEEN CALCULATED YET\n\nClick the green button to fly, or run the "
            "hands-off trim case to save deterministic CSV, plots and a limitations report.",
        )
        self._aircraft_validation_pending = False
        self.aircraft_advanced_visible = False
        for variable in self.aircraft_vars.values():
            variable.trace_add("write", self._queue_aircraft_validation)

    def _queue_aircraft_validation(self, *_args: object) -> None:
        if not self._aircraft_validation_pending:
            self._aircraft_validation_pending = True
            self.root.after_idle(self._validate_aircraft_inputs_inline)

    def _validate_aircraft_inputs_inline(self) -> None:
        self._aircraft_validation_pending = False
        try:
            configuration = self._aircraft_configuration()
            from aerognc.simulation.aircraft_training import aircraft_preflight

            real_time_factor = self._number(
                self.aircraft_vars["real_time_factor"], "Live speed factor"
            )
            if not 0.1 <= real_time_factor <= 10.0:
                raise ValueError("Live speed factor must lie between 0.1 and 10")
            if self._number(self.aircraft_vars["trail_duration"], "Trail duration") <= 0.0:
                raise ValueError("Trail duration must be positive")
            if not self._aircraft_mesh_path().is_file():
                raise ValueError("The selected OBJ/STL model does not exist")
            self._aircraft_mesh_axes()
            self._aircraft_mesh_transform()
            self._aircraft_pilot_profile()
            if self.aircraft_vars["control_mode"].get() not in AIRCRAFT_CONTROL_MODE_CHOICES:
                raise ValueError("Choose a listed aircraft control mode")
            if self.aircraft_vars["camera"].get() not in AIRCRAFT_CAMERA_CHOICES:
                raise ValueError("Choose a listed starting camera")
            if self.aircraft_vars["trail_mode"].get() not in AIRCRAFT_TRAIL_CHOICES:
                raise ValueError("Choose a listed trail mode")
            if self.aircraft_vars["trail_color"].get() not in AIRCRAFT_TRAIL_COLOR_CHOICES:
                raise ValueError("Choose a listed trail colour")
            preflight = aircraft_preflight(configuration)
        except (ValueError, OSError) as error:
            self.aircraft_validation_var.set(f"CHECK INPUT: {error}")
            self.aircraft_preflight_var.set(
                "Preflight unavailable until the highlighted input is fixed."
            )
            return
        self.aircraft_validation_var.set("READY: all aircraft inputs are valid and in SI units.")
        endurance_minutes = preflight.estimated_fuel_endurance_s / 60.0
        self.aircraft_preflight_var.set(
            "CALCULATED PREFLIGHT\n"
            f"Wing loading: {preflight.wing_loading_kgpm2:.1f} kg/m2\n"
            f"Max air-breathing T/W: {preflight.maximum_air_breathing_thrust_to_weight:.3f}\n"
            f"1-g synthetic stall reference: {preflight.stall_speed_1g_mps:.1f} m/s\n"
            f"Initial Mach: {preflight.initial_mach:.3f}\n"
            f"Fuel: {preflight.fuel_mass_kg:.0f} kg; first-order endurance: "
            f"{endurance_minutes:.1f} min\n{preflight.warning}"
        )

    def _apply_aircraft_preset(self) -> None:
        selected = AIRCRAFT_PRESET_CHOICES.get(self.aircraft_vars["preset"].get())
        if selected is None:
            self.aircraft_validation_var.set("CHECK INPUT: choose a listed flight preset.")
            return
        from aerognc.configuration import load_aircraft_configuration
        from aerognc.simulation.aircraft_training import apply_aircraft_preset

        base = load_aircraft_configuration(self._aircraft_configuration_path())
        configuration = apply_aircraft_preset(base, cast("AircraftPresetName", selected))
        self._set_aircraft_configuration_values(configuration)
        self.status_var.set(f"Aircraft preset applied: {self.aircraft_vars['preset'].get()}")

    def _set_aircraft_pilot_profile_values(self, profile: PilotControlProfile) -> None:
        from aerognc.visualisation.aircraft_controls import PilotControlProfile

        if not isinstance(profile, PilotControlProfile):
            raise TypeError("aircraft pilot profile has the wrong type")
        values = self.aircraft_vars
        mapping = {
            "roll_sensitivity": profile.roll_sensitivity,
            "pitch_sensitivity": profile.pitch_sensitivity,
            "yaw_sensitivity": profile.yaw_sensitivity,
            "input_expo": profile.input_expo,
            "analog_deadzone": profile.analog_deadzone,
            "keyboard_ramp": profile.keyboard_ramp_per_s,
            "keyboard_recentering": profile.keyboard_recentering_per_s,
        }
        for key, value in mapping.items():
            values[key].set(f"{value:g}")
        self.aircraft_invert_pitch_var.set(profile.invert_pitch)
        matching_mode = next(
            label
            for label, mode in AIRCRAFT_CONTROL_MODE_CHOICES.items()
            if mode == profile.control_mode
        )
        values["control_mode"].set(matching_mode)

    def _aircraft_pilot_profile(self) -> PilotControlProfile:
        from aerognc.visualisation.aircraft_controls import load_pilot_profile

        path_text = self.aircraft_vars["pilot_profile_path"].get().strip()
        if not path_text:
            raise ValueError("Choose a pilot profile JSON file")
        profile = load_pilot_profile(Path(path_text))
        selected_mode = AIRCRAFT_CONTROL_MODE_CHOICES.get(self.aircraft_vars["control_mode"].get())
        if selected_mode is None:
            raise ValueError("Choose a listed aircraft control mode")
        return replace(
            profile,
            control_mode=cast("AircraftControlMode", selected_mode),
            roll_sensitivity=self._number(
                self.aircraft_vars["roll_sensitivity"], "Roll sensitivity"
            ),
            pitch_sensitivity=self._number(
                self.aircraft_vars["pitch_sensitivity"], "Pitch sensitivity"
            ),
            yaw_sensitivity=self._number(self.aircraft_vars["yaw_sensitivity"], "Yaw sensitivity"),
            input_expo=self._number(self.aircraft_vars["input_expo"], "Control expo"),
            analog_deadzone=self._number(self.aircraft_vars["analog_deadzone"], "Analog deadzone"),
            invert_pitch=self.aircraft_invert_pitch_var.get(),
            keyboard_ramp_per_s=self._number(self.aircraft_vars["keyboard_ramp"], "Keyboard ramp"),
            keyboard_recentering_per_s=self._number(
                self.aircraft_vars["keyboard_recentering"], "Keyboard recentering"
            ),
        )

    def _choose_aircraft_pilot_profile(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose an AeroGNC aircraft pilot profile",
            filetypes=(("JSON pilot profile", "*.json"),),
        )
        if selected:
            self.aircraft_vars["pilot_profile_path"].set(selected)
            self._load_aircraft_pilot_profile()

    def _load_aircraft_pilot_profile(self) -> None:
        try:
            from aerognc.visualisation.aircraft_controls import load_pilot_profile

            path = Path(self.aircraft_vars["pilot_profile_path"].get())
            profile = load_pilot_profile(path)
            self._set_aircraft_pilot_profile_values(profile)
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot load pilot profile", str(error))
            return
        self.status_var.set(f"Pilot profile loaded: {profile.name}")

    def _save_aircraft_pilot_profile(self) -> None:
        try:
            from aerognc.visualisation.aircraft_controls import write_pilot_profile

            profile = self._aircraft_pilot_profile()
            selected = filedialog.asksaveasfilename(
                title="Save aircraft pilot profile",
                defaultextension=".json",
                filetypes=(("JSON pilot profile", "*.json"),),
            )
            if not selected:
                return
            output = write_pilot_profile(profile, selected)
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot save pilot profile", str(error))
            return
        self.aircraft_vars["pilot_profile_path"].set(str(output))
        self.status_var.set(f"Pilot profile saved: {output}")

    def _choose_aircraft_mesh(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a visual aircraft mesh",
            filetypes=(("3D mesh", "*.obj *.stl"), ("OBJ", "*.obj"), ("STL", "*.stl")),
        )
        if selected:
            self.aircraft_vars["mesh_path"].set(selected)
            self.status_var.set(
                "3D appearance selected. Physical mass and aerodynamics remain the entered values."
            )

    def _build_tour_tab(self) -> None:
        tab, self.tour_canvas = self._scrollable_content(self.tour_tab)
        tab.columnconfigure((0, 1), weight=1, uniform="tour")
        ttk.Label(
            tab,
            text="Question: can this spacecraft complete the fictional trip?",
            style="Hero.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            tab,
            text=(
                "New here? Leave the prepared route unchanged and click the green button. "
                "The planner estimates transfers, orbit captures, ideal burns and remaining mass."
            ),
            style="Muted.TLabel",
            wraplength=800,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 12))

        route = ttk.LabelFrame(tab, text="Basic route inputs", padding=14)
        route.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        route.columnconfigure(1, weight=1)
        world_names = tuple(body.name for body in self.fictional_catalog.bodies)
        for row, label, key, help_text in (
            (0, "Start at", "departure", "The fictional world where the spacecraft begins"),
            (
                1,
                "Stop and orbit at",
                "assist",
                "The spacecraft captures here, completes parking orbits, then departs",
            ),
            (2, "Finish at", "destination", "The fictional destination and final parking orbit"),
        ):
            ttk.Label(route, text=label).grid(row=row, column=0, sticky="w", pady=5)
            box = ttk.Combobox(
                route,
                textvariable=self.tour_vars[key],
                values=world_names,
                state="readonly",
                width=16,
            )
            box.grid(row=row, column=1, sticky="ew", padx=(10, 6), pady=5)
            hint = ttk.Label(route, text="?", style="Muted.TLabel")
            hint.grid(row=row, column=2, sticky="w", padx=(8, 0))
            self._bind_hint(box, help_text)
            self._bind_hint(hint, help_text)
        self._field(
            route,
            3,
            "Leave on mission day",
            self.tour_vars["departure_day"],
            "day",
            "Synthetic elapsed mission day; zero starts the example immediately",
        )
        self._field(
            route,
            4,
            "Reach orbit stop on day",
            self.tour_vars["assist_day"],
            "day",
            "Must be later than departure",
        )
        self._field(
            route,
            5,
            "Reach destination on day",
            self.tour_vars["destination_day"],
            "day",
            "Must be later than the orbit-stop arrival",
        )

        meaning = ttk.LabelFrame(tab, text="What the calculation will show", padding=14)
        meaning.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        for row, heading, body in (
            (
                0,
                "ROUTE",
                "Two transfer arcs, capture orbit, whole revolutions, departure, and arrival.",
            ),
            (
                1,
                "ENERGY AND PROPELLANT",
                "Each ideal delta-v, mass after every burn, and departure energy change.",
            ),
            (
                2,
                "PASS OR CHECK",
                "Entered limits plus route-event, mass-floor, and endpoint checks.",
            ),
        ):
            ttk.Label(meaning, text=heading, style="Section.TLabel").grid(
                row=2 * row, column=0, sticky="w"
            )
            ttk.Label(meaning, text=body, wraplength=400, justify="left").grid(
                row=2 * row + 1, column=0, sticky="w", pady=(2, 7)
            )
        ttk.Label(
            meaning,
            text=(
                "This is preliminary patched-conic analysis using fictional circular-orbit "
                "worlds. It is not real navigation or a launch approval tool."
            ),
            style="Safety.TLabel",
            wraplength=370,
            justify="left",
        ).grid(row=6, column=0, sticky="w", pady=(8, 0))

        disclosure = ttk.Frame(tab, padding=(0, 10, 0, 4))
        disclosure.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.tour_advanced_button = ttk.Button(
            disclosure,
            text="SHOW OPTIONAL ORBIT, SPACECRAFT AND LIMIT INPUTS",
            command=self._toggle_tour_advanced,
        )
        self.tour_advanced_button.pack(side="left")
        ttk.Label(
            disclosure,
            text="Optional - the prepared values are already ready to run.",
            style="Muted.TLabel",
        ).pack(side="left", padx=10)
        self.action_buttons.append(self.tour_advanced_button)

        self.tour_advanced_frame = ttk.Frame(tab)
        self.tour_advanced_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.tour_advanced_frame.columnconfigure((0, 1), weight=1, uniform="tour-advanced")
        orbits = ttk.LabelFrame(self.tour_advanced_frame, text="Parking orbits", padding=14)
        orbits.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        orbits.columnconfigure(1, weight=1)
        self._field(
            orbits,
            0,
            "Starting-orbit altitude",
            self.tour_vars["departure_altitude"],
            "km",
            "Height above the starting world's reference radius",
        )
        self._field(
            orbits,
            1,
            "Stop-orbit altitude",
            self.tour_vars["assist_altitude"],
            "km",
            "Height above the intermediate world's reference radius",
        )
        self._field(
            orbits,
            2,
            "Final-orbit altitude",
            self.tour_vars["destination_altitude"],
            "km",
            "Height above the destination world's reference radius",
        )
        self._field(
            orbits,
            3,
            "Orbits completed at stop",
            self.tour_vars["dwell"],
            "revolutions",
            "A whole number of captured parking-orbit revolutions",
        )

        spacecraft = ttk.LabelFrame(
            self.tour_advanced_frame, text="Spacecraft, pass limits and playback", padding=14
        )
        spacecraft.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        spacecraft.columnconfigure(1, weight=1)
        self._field(
            spacecraft,
            0,
            "Mass before first burn",
            self.tour_vars["initial_mass"],
            "kg",
            "Wet mass in the starting parking orbit",
        )
        self._field(
            spacecraft,
            1,
            "Mass without propellant",
            self.tour_vars["dry_mass"],
            "kg",
            "Hard lower mass bound; burns may not consume structure or payload",
        )
        self._field(
            spacecraft,
            2,
            "Ideal engine efficiency (Isp)",
            self.tour_vars["isp"],
            "s",
            "Ideal propulsion performance used by the rocket equation",
        )
        self._field(
            spacecraft,
            3,
            "Pass if total delta-v is at most",
            self.tour_vars["maximum_delta_v"],
            "m/s",
            "User-defined feasibility threshold, not a physical speed limit",
        )
        self._field(
            spacecraft,
            4,
            "Pass if final mass is at least",
            self.tour_vars["minimum_final_mass"],
            "kg",
            "User-defined remaining-mass threshold",
        )
        self._field(
            spacecraft,
            5,
            "3D playback speed",
            self.tour_vars["playback"],
            "days/s",
            "Modeled mission days advanced per real second",
        )
        self.tour_advanced_visible = False
        self.tour_advanced_frame.grid_remove()

        controls = ttk.Frame(tab, padding=(0, 12, 0, 8))
        controls.grid(row=5, column=0, columnspan=2, sticky="ew")
        play = ttk.Button(
            controls,
            text="PLAY THIS TRIP IN 3D",
            style="Success.TButton",
            command=lambda: self._run_tour(open_playback=True),
        )
        play.pack(side="left", padx=(0, 7))
        run = ttk.Button(
            controls,
            text="CALCULATE + SAVE ONLY",
            style="Primary.TButton",
            command=lambda: self._run_tour(open_playback=False),
        )
        run.pack(side="left", padx=7)
        reset = ttk.Button(controls, text="RESTORE EXAMPLE", command=self._reset_tour)
        reset.pack(side="left", padx=7)
        advanced = ttk.Button(
            controls, text="ADVANCED DESIGNER", command=self._open_advanced_designer
        )
        advanced.pack(side="right", padx=7)
        self.action_buttons.extend((play, run, reset, advanced))
        result_frame = ttk.LabelFrame(
            tab, text="What happened in the last calculated trip", padding=10
        )
        result_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")
        tab.rowconfigure(6, weight=1)
        self.tour_summary = tk.Text(
            result_frame,
            height=7,
            background="#081621",
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Cascadia Mono", 9),
            wrap="word",
        )
        tour_scrollbar = ttk.Scrollbar(
            result_frame,
            orient="vertical",
            command=self.tour_summary.yview,
        )
        tour_scrollbar.pack(side="right", fill="y")
        self.tour_summary.configure(yscrollcommand=tour_scrollbar.set)
        self.tour_summary.pack(side="left", fill="both", expand=True)
        self._replace_text(
            self.tour_summary,
            (
                "NOTHING HAS BEEN CALCULATED YET\n\n"
                "Leave the prepared route unchanged and click the green button. The 3D window "
                "will show the two transfer arcs and the intermediate parking orbits; this panel "
                "will then explain the burn sequence, propellant use and pass limits."
            ),
        )

    def _toggle_tour_advanced(self) -> None:
        self.tour_advanced_visible = not self.tour_advanced_visible
        if self.tour_advanced_visible:
            self.tour_advanced_frame.grid()
            self.tour_advanced_button.configure(text="HIDE OPTIONAL INPUTS")
        else:
            self.tour_advanced_frame.grid_remove()
            self.tour_advanced_button.configure(
                text="SHOW OPTIONAL ORBIT, SPACECRAFT AND LIMIT INPUTS"
            )

    def _build_catalog_tab(self) -> None:
        tab = self.catalog_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        context = ttk.Frame(tab, style="Card.TFrame", padding=12)
        context.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        context.columnconfigure(4, weight=1)
        ttk.Label(
            context, text=self.milky_way.name, style="Card.TLabel", font=("Segoe UI Semibold", 14)
        ).grid(row=0, column=0, sticky="w")
        facts = (
            f"{self.milky_way.morphology}  |  approx. "
            f"{self.milky_way.disk_diameter_light_year_approx:,.0f} light-years across  |  "
            f"approx. {self.milky_way.star_count_lower_estimate / 1e9:.0f}-"
            f"{self.milky_way.star_count_upper_estimate / 1e9:.0f} billion stars  |  "
            f"Sun: {self.milky_way.solar_arm}"
        )
        ttk.Label(context, text=facts, style="CardMuted.TLabel", wraplength=780).grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            context,
            text="Approximate sourced context - not a complete galaxy mass model or ephemeris.",
            style="CardSafety.TLabel",
        ).grid(row=2, column=0, columnspan=5, sticky="w", pady=(3, 0))
        data_notebook = ttk.Notebook(tab)
        data_notebook.grid(row=2, column=0, sticky="nsew")
        exoplanets = ttk.Frame(data_notebook, padding=10)
        solar = ttk.Frame(data_notebook, padding=10)
        data_notebook.add(exoplanets, text="Confirmed exoplanets")
        data_notebook.add(solar, text="Eight Solar System planets")
        self._build_exoplanet_table(exoplanets)
        self._build_solar_table(solar)

    def _build_exoplanet_table(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        filters = ttk.Frame(tab)
        filters.grid(row=0, column=0, sticky="ew")
        for column in range(7):
            filters.columnconfigure(column, weight=1 if column in {1, 3, 6} else 0)
        ttk.Label(filters, text="Name or host").grid(row=0, column=0, sticky="w")
        ttk.Entry(filters, textvariable=self.catalog_query_var, width=18).grid(
            row=0, column=1, sticky="ew", padx=(5, 18)
        )
        ttk.Label(filters, text="Within").grid(row=0, column=2, sticky="w")
        ttk.Entry(filters, textvariable=self.catalog_distance_var, width=9).grid(
            row=0, column=3, sticky="ew", padx=(5, 3)
        )
        ttk.Label(filters, text="pc").grid(row=0, column=4, sticky="w", padx=(0, 18))
        methods = tuple(
            sorted({planet.discovery_method for planet in self.exoplanet_catalog.planets})
        )
        ttk.Label(filters, text="Method").grid(row=0, column=5, sticky="w")
        ttk.Combobox(
            filters,
            textvariable=self.catalog_method_var,
            values=("All methods", *methods),
            state="readonly",
            width=19,
        ).grid(row=0, column=6, sticky="ew", padx=(5, 12))
        ttk.Label(filters, text="Years").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(filters, textvariable=self.catalog_min_year_var, width=7).grid(
            row=1, column=1, sticky="ew", padx=(5, 18), pady=(7, 0)
        )
        ttk.Label(filters, text="to").grid(row=1, column=2, sticky="w", pady=(7, 0))
        ttk.Entry(filters, textvariable=self.catalog_max_year_var, width=7).grid(
            row=1, column=3, sticky="ew", padx=(5, 18), pady=(7, 0)
        )
        ttk.Label(filters, text="Limit").grid(row=1, column=5, sticky="w", pady=(7, 0))
        ttk.Entry(filters, textvariable=self.catalog_limit_var, width=7).grid(
            row=1, column=6, sticky="ew", padx=(5, 0), pady=(7, 0)
        )
        actions = ttk.Frame(tab, padding=(0, 8, 0, 6))
        actions.grid(row=1, column=0, sticky="ew")
        search = ttk.Button(
            actions, text="SEARCH DATA", style="Primary.TButton", command=self._search_catalog
        )
        search.pack(side="left")
        explore = ttk.Button(
            actions, text="OPEN SELECTION IN 3D", command=self._open_galaxy_explorer
        )
        explore.pack(side="left", padx=8)
        ttk.Label(actions, textvariable=self.catalog_status_var, style="Muted.TLabel").pack(
            side="left", padx=10
        )
        self.action_buttons.extend((search, explore))
        columns = ("planet", "host", "method", "year", "distance", "period", "radius", "mass")
        tree_frame = ttk.Frame(tab)
        tree_frame.grid(row=2, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.catalog_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        labels = {
            "planet": "Planet",
            "host": "Host star",
            "method": "Discovery method",
            "year": "Year",
            "distance": "Distance [pc]",
            "period": "Period [d]",
            "radius": "Radius [Earth]",
            "mass": "Mass [Earth]",
        }
        widths = {
            "planet": 145,
            "host": 135,
            "method": 145,
            "year": 60,
            "distance": 95,
            "period": 90,
            "radius": 95,
            "mass": 95,
        }
        for name in columns:
            self.catalog_tree.heading(name, text=labels[name])
            self.catalog_tree.column(name, width=widths[name], minwidth=55, anchor="w")
        self.catalog_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.catalog_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.catalog_tree.configure(yscrollcommand=scrollbar.set)
        self.catalog_tree.bind("<<TreeviewSelect>>", self._catalog_row_selected)
        ttk.Label(
            tab, textvariable=self.catalog_detail_var, style="Muted.TLabel", wraplength=780
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))

    def _build_solar_table(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        ttk.Label(
            tab,
            text=(
                "Public NASA descriptive values for the eight IAU planets. These rows are facts "
                "for comparison and are not used by the fictional mission solver."
            ),
            style="Safety.TLabel",
            wraplength=760,
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        columns = ("order", "name", "category", "axis", "period", "radius")
        tree = ttk.Treeview(tab, columns=columns, show="headings")
        tree.grid(row=1, column=0, sticky="nsew")
        headings = (
            ("order", "Order from Sun", 110),
            ("name", "Planet", 130),
            ("category", "Category", 160),
            ("axis", "Semimajor axis [AU]", 170),
            ("period", "Sidereal period [d]", 170),
            ("radius", "Mean radius [km]", 170),
        )
        for name, label, width in headings:
            tree.heading(name, text=label)
            tree.column(name, width=width, anchor="center")
        for planet in self.solar_planets:
            tree.insert(
                "",
                "end",
                values=(
                    planet.order_from_sun,
                    planet.name,
                    planet.category,
                    f"{planet.semimajor_axis_au:.8g}",
                    f"{planet.sidereal_orbit_period_days:.7g}",
                    f"{planet.mean_radius_km:.7g}",
                ),
            )

    def _build_verification_tab(self) -> None:
        tab = self.verification_tab
        tab.columnconfigure((0, 1), weight=1, uniform="verify")
        ttk.Label(tab, text="What is calculated and what is not", style="Hero.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        left = ttk.LabelFrame(tab, text="Numerically implemented", padding=16)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        right = ttk.LabelFrame(tab, text="Scope and limitations", padding=16)
        right.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        implemented = (
            "- Custom fixed-step RK4 and event handling\n"
            "- Nonlinear translational and rotational 6-DOF equations\n"
            "- Hamilton quaternions and NED/FRD frame transforms\n"
            "- Atmosphere, wind, gravity, mass, aero and actuator models\n"
            "- Closed-loop attitude stabilization\n"
            "- Lambert transfers, capture/orbit/departure patched conics\n"
            "- Sequential ideal propellant accounting and requirements\n"
            "- Checksum-verified observational catalog loading"
        )
        limitations = (
            "- Executable vehicle and planetary systems are fictional\n"
            "- Planet-tour burns are impulsive and preliminary\n"
            "- NASA exoplanet rows are not propagation ephemerides\n"
            "- The Milky Way view is detection-biased, not a census\n"
            "- No target interception, homing or engagement logic\n"
            "- No physical flight or hardware-in-the-loop claim\n"
            "- External-tool results are claimed only when executed\n"
            "- SI units are used internally throughout"
        )
        ttk.Label(left, text=implemented, justify="left", wraplength=340).pack(anchor="nw")
        ttk.Label(right, text=limitations, justify="left", wraplength=340).pack(anchor="nw")
        evidence = ttk.Frame(tab, style="Card.TFrame", padding=16)
        evidence.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        ttk.Label(
            evidence,
            text="Evidence-driven project",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 13),
        ).pack(anchor="w")
        ttk.Label(
            evidence,
            text=(
                "Plots are presentation, not proof. The repository also contains unit tests, "
                "analytical benchmarks, convergence tests, independent SciPy/MATLAB comparisons, "
                "cross-model consistency tests, requirement traces and deterministic regression "
                "artifacts."
            ),
            style="CardMuted.TLabel",
            wraplength=780,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        ttk.Label(
            tab,
            text=(
                "Tip: close a 3D player window to return here. Drag to rotate; use Space to pause "
                "and C to change camera."
            ),
            style="Muted.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(16, 0))

    @staticmethod
    def _replace_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    @staticmethod
    def _number(variable: tk.StringVar, label: str) -> float:
        raw = variable.get().strip()
        if "," in raw and "." not in raw:
            raw = raw.replace(",", ".")
        try:
            return float(raw)
        except ValueError as error:
            raise ValueError(f"{label} must be a number; received {variable.get()!r}") from error

    @classmethod
    def _integer(cls, variable: tk.StringVar, label: str) -> int:
        value = cls._number(variable, label)
        if not value.is_integer():
            raise ValueError(f"{label} must be a whole number")
        return int(value)

    @classmethod
    def _optional_number(cls, variable: tk.StringVar, label: str) -> float | None:
        return None if not variable.get().strip() else cls._number(variable, label)

    @classmethod
    def _optional_integer(cls, variable: tk.StringVar, label: str) -> int | None:
        return None if not variable.get().strip() else cls._integer(variable, label)

    def _rocket_inputs(self) -> RocketWorkbenchInputs:
        values = self.rocket_vars
        return RocketWorkbenchInputs(
            duration_s=self._number(values["duration"], "Duration"),
            step_s=self._number(values["step"], "RK4 time step"),
            initial_speed_mps=self._number(values["speed"], "Rail-exit speed"),
            initial_euler321_deg=(
                self._number(values["roll"], "Roll"),
                self._number(values["pitch"], "Pitch"),
                self._number(values["yaw"], "Yaw"),
            ),
            initial_angular_rate_body_degps=(
                self._number(values["roll_rate"], "Roll rate"),
                self._number(values["pitch_rate"], "Pitch rate"),
                self._number(values["yaw_rate"], "Yaw rate"),
            ),
            playback_speed=self._number(values["playback"], "Playback rate"),
        )

    def _orbit_configuration_path(self) -> Path:
        return (
            self.paths.orbit_sandbox_configuration
            if self.paths.orbit_sandbox_configuration is not None
            else Path("configs/orbit_sandbox.yaml").resolve()
        )

    def _aircraft_configuration_path(self) -> Path:
        return (
            self.paths.aircraft_configuration
            if self.paths.aircraft_configuration is not None
            else Path("configs/aircraft_sandbox.yaml").resolve()
        )

    def _aircraft_mesh_path(self) -> Path:
        configured = self.aircraft_vars["mesh_path"].get().strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return (
            self.paths.aircraft_mesh
            if self.paths.aircraft_mesh is not None
            else Path("assets/models/aquila_x1.obj").resolve()
        )

    def _orbit_configuration(self) -> OrbitSandboxConfiguration:
        from aerognc.configuration.orbit_sandbox_loader import (
            OrbitModelName,
            OrbitSpeedMode,
            load_orbit_sandbox_configuration,
        )

        base = load_orbit_sandbox_configuration(self._orbit_configuration_path())
        values = self.orbit_vars
        model_value = ORBIT_MODEL_CHOICES.get(values["model"].get())
        speed_value = ORBIT_SPEED_CHOICES.get(values["speed_mode"].get())
        if model_value is None:
            raise ValueError("Choose one orbit physics model")
        if speed_value is None:
            raise ValueError("Choose one starting speed rule")
        custom_speed = self._number(values["custom_speed"], "Custom speed")
        if speed_value == "custom" and custom_speed <= 0.0:
            raise ValueError("Custom speed must be greater than zero when that rule is selected")
        model = cast(OrbitModelName, model_value)
        speed_mode = cast(OrbitSpeedMode, speed_value)
        initial = replace(
            base.initial,
            altitude_m=1_000.0 * self._number(values["altitude"], "Starting height"),
            speed_mode=speed_mode,
            custom_speed_mps=custom_speed,
            inclination_rad=np.deg2rad(self._number(values["inclination"], "Orbit tilt")),
        )
        satellite = replace(
            base.satellite,
            initial_mass_kg=self._number(values["mass"], "Initial satellite mass"),
            dry_mass_kg=self._number(values["dry_mass"], "Satellite dry mass"),
            drag_area_m2=self._number(values["area"], "Drag area"),
            drag_coefficient=self._number(values["drag_coefficient"], "Drag coefficient"),
        )
        correction = replace(
            base.correction,
            enabled=self.orbit_correction_var.get(),
            trigger_altitude_m=1_000.0
            * self._number(values["correction_altitude"], "Correction altitude"),
            maximum_burns=self._integer(values["maximum_corrections"], "Maximum corrections"),
        )
        return replace(
            base,
            model=model,
            satellite=satellite,
            initial=initial,
            correction=correction,
            duration_s=86_400.0 * self._number(values["duration_days"], "Survival duration"),
            integration_step_s=self._number(values["integration_step"], "Integration step"),
            output_step_s=self._number(values["output_step"], "Output spacing"),
            reentry_altitude_m=1_000.0
            * self._number(values["reentry_altitude"], "Reentry threshold"),
            atmosphere_density_scale=self._number(
                values["density_scale"], "Atmosphere density multiplier"
            ),
        )

    def _aircraft_configuration(self) -> AircraftSandboxConfiguration:
        from aerognc.configuration import load_aircraft_configuration

        base = load_aircraft_configuration(self._aircraft_configuration_path())
        values = self.aircraft_vars
        initial = replace(
            base.initial,
            altitude_m=self._number(values["altitude"], "Starting altitude"),
            true_airspeed_mps=self._number(values["airspeed"], "Starting airspeed"),
            heading_rad=np.deg2rad(self._number(values["heading"], "Starting heading")),
            flight_path_angle_rad=np.deg2rad(
                self._number(values["flight_path_angle"], "Starting climb angle")
            ),
            bank_angle_rad=np.deg2rad(self._number(values["bank"], "Starting bank")),
            angle_of_attack_rad=np.deg2rad(
                self._number(values["alpha"], "Starting angle of attack")
            ),
        )
        mass = replace(
            base.mass,
            initial_mass_kg=self._number(values["mass"], "Initial aircraft mass"),
            dry_mass_kg=self._number(values["dry_mass"], "Aircraft dry mass"),
        )
        geometry = replace(
            base.geometry,
            wing_area_m2=self._number(values["wing_area"], "Wing reference area"),
        )
        aerodynamics = replace(
            base.aerodynamics,
            cl_zero=self._number(values["cl_zero"], "CL0"),
            cl_alpha_per_rad=self._number(values["cl_alpha"], "CL alpha"),
            cl_maximum=self._number(values["cl_maximum"], "CL maximum"),
            stall_angle_rad=np.deg2rad(self._number(values["stall_angle"], "Stall angle")),
            cd_zero=self._number(values["cd_zero"], "CD0"),
            induced_drag_factor=self._number(values["induced_drag"], "Induced-drag factor"),
            pitch_alpha_per_rad=self._number(values["pitch_alpha"], "Cm alpha"),
        )
        return replace(
            base,
            initial=initial,
            initial_throttle=self._number(values["throttle"], "Starting throttle"),
            mass=mass,
            geometry=geometry,
            aerodynamics=aerodynamics,
            duration_s=self._number(values["duration"], "Hands-off duration"),
            wind_north_mps=self._number(values["wind_north"], "North wind"),
            wind_east_mps=self._number(values["wind_east"], "East wind"),
            turbulence_std_ned_mps=(
                self._number(values["turbulence_north"], "North turbulence standard deviation"),
                self._number(values["turbulence_east"], "East turbulence standard deviation"),
                self._number(values["turbulence_down"], "Down turbulence standard deviation"),
            ),
            turbulence_correlation_time_s=self._number(
                values["turbulence_correlation"], "Turbulence correlation time"
            ),
            wind_random_seed=self._integer(values["wind_seed"], "Wind random seed"),
            gust_start_time_s=self._number(values["gust_start"], "Gust start time"),
            gust_duration_s=self._number(values["gust_duration"], "Gust duration"),
            gust_amplitude_ned_mps=(
                self._number(values["gust_north"], "North gust amplitude"),
                self._number(values["gust_east"], "East gust amplitude"),
                self._number(values["gust_down"], "Down gust amplitude"),
            ),
        )

    def _aircraft_mesh_axes(self) -> MeshAxisConvention:
        from aerognc.visualisation.mesh import MeshAxisConvention

        value = MESH_AXIS_CHOICES.get(self.aircraft_vars["mesh_axes"].get())
        if value is None:
            raise ValueError("Choose the source axes used by the imported 3D model")
        return cast(MeshAxisConvention, value)

    def _aircraft_mesh_transform(self) -> MeshTransform:
        from aerognc.visualisation.mesh import MeshTransform

        center_modes = {
            "Centroid": "centroid",
            "Bounds centre": "bounds",
            "Keep source origin": "none",
        }
        center_mode = center_modes.get(self.aircraft_vars["mesh_center"].get())
        if center_mode is None:
            raise ValueError("Choose how the imported visual mesh should be centred")
        return MeshTransform(
            rotation_deg_xyz=(
                self._number(self.aircraft_vars["mesh_rotation_x"], "Mesh X rotation"),
                self._number(self.aircraft_vars["mesh_rotation_y"], "Mesh Y rotation"),
                self._number(self.aircraft_vars["mesh_rotation_z"], "Mesh Z rotation"),
            ),
            flip_x=self.aircraft_mesh_flip_x_var.get(),
            flip_y=self.aircraft_mesh_flip_y_var.get(),
            flip_z=self.aircraft_mesh_flip_z_var.get(),
            center_mode=cast("MeshCenterMode", center_mode),
        )

    def _preview_aircraft_mesh(self) -> None:
        try:
            from aerognc.visualisation.mesh import load_triangle_mesh
            from aerognc.visualisation.mesh_preview import show_mesh_preview

            mesh = load_triangle_mesh(
                self._aircraft_mesh_path(),
                axis_convention=self._aircraft_mesh_axes(),
                transform=self._aircraft_mesh_transform(),
            )
            show_mesh_preview(mesh, block=False)
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot preview aircraft mesh", str(error))
            return
        inspection = mesh.inspection()
        self.status_var.set(
            f"Mesh preview open: {inspection.triangle_count:,} source triangles; "
            f"{inspection.live_triangle_count:,} live triangles. Physics inputs are unchanged."
        )

    def _tour_inputs(self) -> OrbitTourWorkbenchInputs:
        values = self.tour_vars
        return OrbitTourWorkbenchInputs(
            departure_body=values["departure"].get(),
            assist_body=values["assist"].get(),
            destination_body=values["destination"].get(),
            departure_day=self._number(values["departure_day"], "Departure day"),
            assist_arrival_day=self._number(values["assist_day"], "Assist arrival day"),
            destination_arrival_day=self._number(
                values["destination_day"], "Destination arrival day"
            ),
            departure_parking_altitude_km=self._number(
                values["departure_altitude"], "Departure orbit altitude"
            ),
            assist_parking_altitude_km=self._number(
                values["assist_altitude"], "Assist orbit altitude"
            ),
            destination_parking_altitude_km=self._number(
                values["destination_altitude"], "Destination orbit altitude"
            ),
            assist_dwell_revolutions=self._integer(values["dwell"], "Assist dwell revolutions"),
            initial_mass_kg=self._number(values["initial_mass"], "Initial mass"),
            dry_mass_kg=self._number(values["dry_mass"], "Dry mass"),
            specific_impulse_s=self._number(values["isp"], "Specific impulse"),
            maximum_total_delta_v_mps=self._number(
                values["maximum_delta_v"], "Maximum total delta-v"
            ),
            minimum_final_mass_kg=self._number(values["minimum_final_mass"], "Minimum final mass"),
            playback_days_per_second=self._number(values["playback"], "Playback rate"),
        )

    def _catalog_inputs(self) -> CatalogWorkbenchInputs:
        method = self.catalog_method_var.get()
        return CatalogWorkbenchInputs(
            text=self.catalog_query_var.get(),
            maximum_distance_pc=self._optional_number(
                self.catalog_distance_var, "Maximum distance"
            ),
            discovery_method=None if method == "All methods" else method,
            minimum_discovery_year=self._optional_integer(
                self.catalog_min_year_var, "Minimum year"
            ),
            maximum_discovery_year=self._optional_integer(
                self.catalog_max_year_var, "Maximum year"
            ),
            limit=self._integer(self.catalog_limit_var, "Result limit"),
        )

    def _open_project_dialog(self) -> None:
        selected = filedialog.askopenfilename(
            title="Open AeroGNC engineering project",
            filetypes=(
                ("AeroGNC project", "*.aerognc.yaml"),
                ("YAML", "*.yaml *.yml"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self._load_project_path(Path(selected), show_error=True)

    def _load_project_path(self, path: Path, *, show_error: bool) -> None:
        try:
            snapshot = self.project_service.open(path)
        except Exception as error:
            self.project_identity_var.set("Project could not be opened.")
            self.project_validation_var.set(str(error))
            if show_error:
                messagebox.showerror("Cannot open project", str(error))
            return
        self._apply_project_snapshot(snapshot)
        self.status_var.set(f"Project opened and structurally validated: {snapshot.project_name}")

    def _apply_project_snapshot(self, snapshot: ProjectWorkbenchSnapshot) -> None:
        self.project_snapshot = snapshot
        self.project_identity_var.set(
            f"{snapshot.project_name} | {snapshot.project_path.name} | "
            f"{len(snapshot.scenarios)} scenarios | {len(snapshot.runs)} runs"
        )
        if snapshot.validation_issues:
            self.project_validation_var.set(
                "Validation issues: " + "; ".join(snapshot.validation_issues)
            )
        else:
            self.project_validation_var.set(
                "VALID - project schema, paths, safety scope, configurations and workflows "
                "are available."
            )
        for item in self.project_scenario_tree.get_children():
            self.project_scenario_tree.delete(item)
        for scenario in snapshot.scenarios:
            self.project_scenario_tree.insert(
                "",
                "end",
                iid=scenario.name,
                values=(
                    scenario.name,
                    scenario.workflow,
                    scenario.configuration,
                    "yes" if scenario.enabled else "no",
                    scenario.seed,
                ),
            )
        for item in self.project_history_tree.get_children():
            self.project_history_tree.delete(item)
        for run in snapshot.runs:
            self.project_history_tree.insert(
                "",
                "end",
                iid=run.run_id,
                values=(
                    run.created_utc,
                    run.scenario_name,
                    run.workflow,
                    run.status,
                    run.run_id,
                ),
            )
        if snapshot.scenarios:
            self.project_scenario_tree.selection_set(snapshot.scenarios[0].name)
            self.project_scenario_tree.focus(snapshot.scenarios[0].name)
            self._project_scenario_selected(None)

    def _save_project(self) -> None:
        try:
            snapshot = self.project_service.save()
        except Exception as error:
            messagebox.showerror("Cannot save project", str(error))
            return
        self._apply_project_snapshot(snapshot)
        self.status_var.set(f"Project saved and reloaded: {snapshot.project_path}")

    def _save_project_as(self) -> None:
        try:
            project = self.project_service.project
        except RuntimeError as error:
            messagebox.showerror("Cannot save project", str(error))
            return
        selected = filedialog.asksaveasfilename(
            title="Save AeroGNC project as",
            initialdir=project.workspace_root,
            initialfile=project.source_path.name,
            defaultextension=".aerognc.yaml",
            filetypes=(("AeroGNC project", "*.aerognc.yaml"), ("YAML", "*.yaml")),
        )
        if not selected:
            return
        try:
            snapshot = self.project_service.save(Path(selected))
        except Exception as error:
            messagebox.showerror("Cannot save project", str(error))
            return
        self._apply_project_snapshot(snapshot)
        self.status_var.set(f"Project saved as {snapshot.project_path}")

    def _validate_open_project(self) -> None:
        try:
            snapshot = self.project_service.snapshot()
        except Exception as error:
            messagebox.showerror("Project validation", str(error))
            return
        self._apply_project_snapshot(snapshot)
        if snapshot.validation_issues:
            messagebox.showwarning(
                "Project validation",
                "The project has workflow issues:\n\n" + "\n".join(snapshot.validation_issues),
            )
        else:
            messagebox.showinfo(
                "Project validation",
                "Project schema, paths, inputs, safety scope and workflows are valid.",
            )

    def _refresh_project(self) -> None:
        try:
            snapshot = self.project_service.snapshot()
        except Exception as error:
            messagebox.showerror("Cannot refresh project", str(error))
            return
        self._apply_project_snapshot(snapshot)
        self.status_var.set("Project scenario and immutable run history refreshed.")

    def _project_scenario_selected(self, _event: object | None) -> None:
        if self.project_snapshot is None:
            return
        selected = self.project_scenario_tree.selection()
        if not selected:
            return
        scenario = next(
            item for item in self.project_snapshot.scenarios if item.name == selected[0]
        )
        tags = ", ".join(scenario.tags) if scenario.tags else "none"
        state = "enabled" if scenario.enabled else "disabled"
        self.project_scenario_detail_var.set(
            f"{scenario.description or 'No description supplied.'} | {state} | "
            f"seed {scenario.seed} | tags: {tags}"
        )

    def _project_progress(self, fraction: float, message: str) -> None:
        self.status_var.set(f"Project run {100.0 * fraction:.0f}% - {message}")

    def _run_project_scenario(self) -> None:
        if self.busy:
            messagebox.showinfo("AeroGNC-Lab", "A calculation is already running.")
            return
        selected = self.project_scenario_tree.selection()
        if len(selected) != 1:
            messagebox.showerror("Project run", "Select one enabled project scenario first.")
            return
        scenario_name = selected[0]
        token = CancellationToken()
        self.project_cancellation = token

        def progress(fraction: float, message: str) -> None:
            self.root.after(0, lambda: self._project_progress(fraction, message))

        self._background(
            f"Starting project scenario {scenario_name}...",
            lambda: self.project_service.run_scenario(
                scenario_name,
                cancellation=token,
                progress=progress,
            ),
            self._project_run_complete,
            on_error=self._project_run_error,
        )
        self.project_cancel_button.configure(state="normal")

    def _cancel_project_run(self) -> None:
        if self.project_cancellation is None:
            return
        self.project_cancellation.cancel()
        self.project_cancel_button.configure(state="disabled")
        self.status_var.set("Cancellation requested - waiting for the next safe solver boundary...")

    def _project_run_complete(self, stored: StoredRun) -> None:
        self.project_cancellation = None
        self.project_cancel_button.configure(state="disabled")
        self._refresh_project()
        if self.project_history_tree.exists(stored.manifest.run_id):
            self.project_history_tree.selection_set(stored.manifest.run_id)
            self.project_history_tree.see(stored.manifest.run_id)
        outcome = "PASS" if all(item.passed for item in stored.manifest.requirements) else "CHECK"
        self.status_var.set(
            f"Project run complete ({outcome}) - {stored.manifest.run_id}; report generated."
        )

    def _project_run_error(self, message: str) -> None:
        self._finish_busy()
        self.project_cancellation = None
        self.project_cancel_button.configure(state="disabled")
        self._refresh_project()
        self.status_var.set("Project run ended as failed or cancelled; terminal evidence retained.")
        messagebox.showerror("Project run", message)

    def _selected_project_run_ids(self, expected_count: int) -> tuple[str, ...] | None:
        selected = tuple(self.project_history_tree.selection())
        if len(selected) != expected_count:
            messagebox.showerror(
                "Run selection",
                f"Select exactly {expected_count} run{'s' if expected_count != 1 else ''}.",
            )
            return None
        return selected

    def _compare_project_runs(self) -> None:
        selected = self._selected_project_run_ids(2)
        if selected is None:
            return
        self._background(
            "Loading, integrity-checking and aligning selected runs...",
            lambda: self.project_service.compare_runs(selected[0], selected[1]),
            self._project_comparison_complete,
        )

    def _project_comparison_complete(self, result: ProjectRunComparison) -> None:
        self.last_project_comparison_report = result.report_path
        lines = [
            "RUN COMPARISON COMPLETE",
            f"Baseline: {result.baseline_run_id}",
            f"Candidate: {result.candidate_run_id}",
            f"Common interval: {result.comparison.start_time_s:.6g} to "
            f"{result.comparison.end_time_s:.6g} s",
            "",
        ]
        for item in result.comparison.channels:
            lines.append(
                f"{item.channel} [{item.unit}]: RMS {item.rms_difference:.6g}, "
                f"max |difference| {item.maximum_absolute_difference:.6g}, "
                f"correlation {item.correlation:.5f}"
            )
        lines.extend(("", f"JSON: {result.json_path}", f"HTML: {result.report_path}"))
        self._replace_text(self.project_comparison_text, "\n".join(lines))
        self.status_var.set("Compatible runs compared - JSON and engineering report generated.")

    def _open_project_report(self) -> None:
        selected = self._selected_project_run_ids(1)
        if selected is None:
            return
        try:
            path = self.project_service.report_path(selected[0])
            webbrowser.open(path.as_uri())
        except Exception as error:
            messagebox.showerror("Cannot open report", str(error))
            return
        self.status_var.set(f"Opened self-contained engineering report: {path.name}")

    def _open_last_comparison_report(self) -> None:
        path = self.last_project_comparison_report
        if path is None or not path.is_file():
            messagebox.showinfo("Comparison report", "Compare two completed runs first.")
            return
        webbrowser.open(path.as_uri())
        self.status_var.set(f"Opened comparison report: {path.name}")

    def _reset_rocket(self) -> None:
        defaults = RocketWorkbenchInputs()
        values = self.rocket_vars
        values["duration"].set(f"{defaults.duration_s:g}")
        values["step"].set(f"{defaults.step_s:g}")
        values["speed"].set(f"{defaults.initial_speed_mps:g}")
        for key, value in zip(("roll", "pitch", "yaw"), defaults.initial_euler321_deg, strict=True):
            values[key].set(f"{value:g}")
        for key, value in zip(
            ("roll_rate", "pitch_rate", "yaw_rate"),
            defaults.initial_angular_rate_body_degps,
            strict=True,
        ):
            values[key].set(f"{value:g}")
        values["playback"].set(f"{defaults.playback_speed:g}")
        if hasattr(self, "rocket_canvas"):
            self.rocket_canvas.yview_moveto(0.0)
        self.status_var.set("Verified rocket preset restored.")

    def _reset_orbit(self) -> None:
        from aerognc.configuration import load_orbit_sandbox_configuration

        configuration = load_orbit_sandbox_configuration(self._orbit_configuration_path())
        values = self.orbit_vars
        values["model"].set(
            next(
                label
                for label, value in ORBIT_MODEL_CHOICES.items()
                if value == configuration.model
            )
        )
        values["altitude"].set(f"{configuration.initial.altitude_m / 1_000.0:g}")
        values["speed_mode"].set(
            next(
                label
                for label, value in ORBIT_SPEED_CHOICES.items()
                if value == configuration.initial.speed_mode
            )
        )
        values["custom_speed"].set(f"{configuration.initial.custom_speed_mps:g}")
        values["inclination"].set(f"{np.rad2deg(configuration.initial.inclination_rad):g}")
        values["duration_days"].set(f"{configuration.duration_s / 86_400.0:g}")
        values["mass"].set(f"{configuration.satellite.initial_mass_kg:g}")
        values["dry_mass"].set(f"{configuration.satellite.dry_mass_kg:g}")
        values["area"].set(f"{configuration.satellite.drag_area_m2:g}")
        values["drag_coefficient"].set(f"{configuration.satellite.drag_coefficient:g}")
        values["density_scale"].set(f"{configuration.atmosphere_density_scale:g}")
        values["reentry_altitude"].set(f"{configuration.reentry_altitude_m / 1_000.0:g}")
        values["integration_step"].set(f"{configuration.integration_step_s:g}")
        values["output_step"].set(f"{configuration.output_step_s:g}")
        values["correction_altitude"].set(
            f"{configuration.correction.trigger_altitude_m / 1_000.0:g}"
        )
        values["maximum_corrections"].set(str(configuration.correction.maximum_burns))
        self.orbit_correction_var.set(configuration.correction.enabled)
        if hasattr(self, "orbit_canvas"):
            self.orbit_canvas.yview_moveto(0.0)
        self.status_var.set("Verified satellite-orbit preset restored.")

    def _set_aircraft_configuration_values(
        self, configuration: AircraftSandboxConfiguration
    ) -> None:
        values = self.aircraft_vars
        mapping = {
            "altitude": configuration.initial.altitude_m,
            "airspeed": configuration.initial.true_airspeed_mps,
            "heading": np.rad2deg(configuration.initial.heading_rad),
            "flight_path_angle": np.rad2deg(configuration.initial.flight_path_angle_rad),
            "bank": np.rad2deg(configuration.initial.bank_angle_rad),
            "alpha": np.rad2deg(configuration.initial.angle_of_attack_rad),
            "throttle": configuration.initial_throttle,
            "duration": configuration.duration_s,
            "mass": configuration.mass.initial_mass_kg,
            "dry_mass": configuration.mass.dry_mass_kg,
            "wing_area": configuration.geometry.wing_area_m2,
            "cl_zero": configuration.aerodynamics.cl_zero,
            "cl_alpha": configuration.aerodynamics.cl_alpha_per_rad,
            "cl_maximum": configuration.aerodynamics.cl_maximum,
            "stall_angle": np.rad2deg(configuration.aerodynamics.stall_angle_rad),
            "cd_zero": configuration.aerodynamics.cd_zero,
            "induced_drag": configuration.aerodynamics.induced_drag_factor,
            "pitch_alpha": configuration.aerodynamics.pitch_alpha_per_rad,
            "wind_north": configuration.wind_north_mps,
            "wind_east": configuration.wind_east_mps,
            "turbulence_north": configuration.turbulence_std_ned_mps[0],
            "turbulence_east": configuration.turbulence_std_ned_mps[1],
            "turbulence_down": configuration.turbulence_std_ned_mps[2],
            "turbulence_correlation": configuration.turbulence_correlation_time_s,
            "wind_seed": configuration.wind_random_seed,
            "gust_start": configuration.gust_start_time_s,
            "gust_duration": configuration.gust_duration_s,
            "gust_north": configuration.gust_amplitude_ned_mps[0],
            "gust_east": configuration.gust_amplitude_ned_mps[1],
            "gust_down": configuration.gust_amplitude_ned_mps[2],
        }
        for key, value in mapping.items():
            values[key].set(f"{value:g}")

    def _reset_aircraft(self) -> None:
        from aerognc.configuration import load_aircraft_configuration
        from aerognc.visualisation.aircraft_controls import load_pilot_profile

        configuration = load_aircraft_configuration(self._aircraft_configuration_path())
        values = self.aircraft_vars
        self._set_aircraft_configuration_values(configuration)
        mesh_path = (
            self.paths.aircraft_mesh
            if self.paths.aircraft_mesh is not None
            else Path("assets/models/aquila_x1.obj").resolve()
        )
        values["mesh_path"].set(str(mesh_path))
        values["mesh_axes"].set(next(iter(MESH_AXIS_CHOICES)))
        values["preset"].set(next(iter(AIRCRAFT_PRESET_CHOICES)))
        values["real_time_factor"].set("1")
        values["control_mode"].set(next(iter(AIRCRAFT_CONTROL_MODE_CHOICES)))
        pilot_profile_path = Path("configs/pilot_profiles/accessible.json").resolve()
        values["pilot_profile_path"].set(str(pilot_profile_path))
        self._set_aircraft_pilot_profile_values(load_pilot_profile(pilot_profile_path))
        values["camera"].set(next(iter(AIRCRAFT_CAMERA_CHOICES)))
        values["trail_mode"].set(next(iter(AIRCRAFT_TRAIL_CHOICES)))
        values["trail_duration"].set("45")
        values["trail_color"].set(next(iter(AIRCRAFT_TRAIL_COLOR_CHOICES)))
        values["mesh_rotation_x"].set("0")
        values["mesh_rotation_y"].set("0")
        values["mesh_rotation_z"].set("0")
        values["mesh_center"].set("Centroid")
        values["mesh_scale_mode"].set("Enlarged visible marker")
        values["recorder_directory"].set(str(Path("results/aircraft_live").resolve()))
        self.aircraft_gamepad_var.set(True)
        self.aircraft_mesh_flip_x_var.set(False)
        self.aircraft_mesh_flip_y_var.set(False)
        self.aircraft_mesh_flip_z_var.set(False)
        if hasattr(self, "aircraft_canvas"):
            self.aircraft_canvas.yview_moveto(0.0)
        self.status_var.set("Verified fictional-aircraft preset restored.")

    def _reset_tour(self) -> None:
        defaults = OrbitTourWorkbenchInputs()
        values = self.tour_vars
        mapping: dict[str, str | float | int] = {
            "departure": defaults.departure_body,
            "assist": defaults.assist_body,
            "destination": defaults.destination_body,
            "departure_day": defaults.departure_day,
            "assist_day": defaults.assist_arrival_day,
            "destination_day": defaults.destination_arrival_day,
            "departure_altitude": defaults.departure_parking_altitude_km,
            "assist_altitude": defaults.assist_parking_altitude_km,
            "destination_altitude": defaults.destination_parking_altitude_km,
            "dwell": defaults.assist_dwell_revolutions,
            "initial_mass": defaults.initial_mass_kg,
            "dry_mass": defaults.dry_mass_kg,
            "isp": defaults.specific_impulse_s,
            "maximum_delta_v": defaults.maximum_total_delta_v_mps,
            "minimum_final_mass": defaults.minimum_final_mass_kg,
            "playback": defaults.playback_days_per_second,
        }
        for key, value in mapping.items():
            values[key].set(f"{value:g}" if isinstance(value, float) else str(value))
        if hasattr(self, "tour_canvas"):
            self.tour_canvas.yview_moveto(0.0)
        self.status_var.set("Verified planetary-tour preset restored.")

    def _background(
        self,
        status: str,
        operation: Callable[[], T],
        on_success: Callable[[T], None],
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if self.busy:
            messagebox.showinfo("AeroGNC-Lab", "A calculation is already running.")
            return
        self.busy = True
        self.status_var.set(status)
        self.progress.pack(fill="x", padx=24, before=self.status_label)
        self.progress.start(10)
        for button in self.action_buttons:
            button.configure(state="disabled")

        def task() -> None:
            try:
                result = operation()
            except Exception as error:
                message = str(error)
                error_callback = self._background_error if on_error is None else on_error
                self.root.after(0, lambda: error_callback(message))
            else:
                self.root.after(0, lambda: self._background_success(result, on_success))

        threading.Thread(target=task, daemon=True, name="aerognc-workbench-worker").start()

    def _finish_busy(self) -> None:
        self.busy = False
        self.progress.stop()
        self.progress.pack_forget()
        for button in self.action_buttons:
            button.configure(state="normal")
        if hasattr(self, "project_cancel_button"):
            self.project_cancel_button.configure(state="disabled")

    def _background_error(self, message: str) -> None:
        self._finish_busy()
        self.status_var.set("Input or solver error - correct the highlighted concept and retry.")
        messagebox.showerror("Input or solver error", message)

    def _background_success(self, result: T, callback: Callable[[T], None]) -> None:
        self._finish_busy()
        callback(result)

    def _run_rocket(self, *, open_playback: bool) -> None:
        try:
            inputs = self._rocket_inputs()
        except ValueError as error:
            messagebox.showerror("Rocket input", str(error))
            return

        def operation() -> RocketRun:
            configuration, result = run_rocket_workbench(
                self.paths.six_dof_configuration,
                inputs,
            )
            output = configuration.output_directory
            write_result_csv(result, output / "six_dof_trajectory.csv")
            write_summary_json(result, output / "six_dof_summary.json")
            return RocketRun(result, inputs.playback_speed, output)

        self._background(
            "Running nonlinear 6-DOF equations and closed-loop controller...",
            operation,
            lambda run: self._rocket_complete(run, open_playback=open_playback),
        )

    def _rocket_complete(self, run: RocketRun, *, open_playback: bool) -> None:
        maxima = run.result.maximum_summary
        events = (
            ", ".join(
                f"{event['name']} at {float(event['time_s']):.3f} s"
                for event in run.result.event_summary
            )
            or "No configured event occurred during this short time window"
        )
        summary = (
            "WHAT HAPPENED\n"
            f"The solver advanced the fictional rocket from 0 to {run.result.time_s[-1]:.3f} s "
            f"in {run.result.time_s.size:,} calculated states. Its highest modeled altitude "
            f"in that window was {float(maxima['altitude']['value']):.3f} m and its highest "
            f"total speed was {float(maxima['speed']['value']):.3f} m/s.\n\n"
            "DID IT STAY STABLE?\n"
            f"Maximum attitude error was {float(maxima['attitude_error']['value']):.4f} deg. "
            "This is the largest angular separation between the commanded and calculated "
            "orientation; smaller is better. Maximum body rotation rate was "
            f"{float(maxima['angular_rate']['value']):.4f} deg/s.\n\n"
            "WHAT TO LOOK FOR IN 3D\n"
            "The moving vehicle follows the numerically calculated position and orientation. "
            "The path is not a prerecorded animation. Use the time plots to relate motion, "
            "loads and controller response.\n\n"
            f"EVENTS\n{events}.\n\n"
            "SAVED EVIDENCE\n"
            f"Deterministic trajectory CSV and summary JSON: {run.output_directory}"
        )
        self._replace_text(self.rocket_summary, summary)
        self.rocket_canvas.update_idletasks()
        self.rocket_canvas.yview_moveto(1.0)
        self.status_var.set("Rocket simulation complete - numerical results saved locally.")
        if open_playback:
            try:
                from aerognc.visualisation.playback_3d import play_six_dof_3d

                self.status_var.set("3D rocket player open - close it to return to the workbench.")
                play_six_dof_3d(run.result, playback_speed=run.playback_speed)
            except Exception as error:
                messagebox.showerror("Cannot open 3D rocket player", str(error))

    def _run_orbit(self, *, open_playback: bool) -> None:
        try:
            configuration = self._orbit_configuration()
        except (ValueError, OSError) as error:
            messagebox.showerror("Satellite orbit input", str(error))
            return

        def operation() -> OrbitSandboxRun:
            from aerognc.simulation.orbit_sandbox import (
                simulate_orbit_sandbox,
                write_orbit_sandbox_results,
            )
            from aerognc.visualisation.orbit_sandbox import plot_orbit_sandbox

            simulation = simulate_orbit_sandbox(configuration)
            output = configuration.output_directory
            write_orbit_sandbox_results(simulation, output)
            plot_orbit_sandbox(simulation, output)
            return OrbitSandboxRun(simulation, output)

        self._background(
            "Propagating the selected gravity, perturbation and drag equations...",
            operation,
            lambda run: self._orbit_complete(run, open_playback=open_playback),
        )

    def _orbit_complete(self, run: OrbitSandboxRun, *, open_playback: bool) -> None:
        from aerognc.simulation.orbit_sandbox import ORBIT_MODEL_DESCRIPTIONS

        simulation = run.simulation
        result = simulation.result
        columns = result.columns
        event_text = (
            "\n".join(
                f"  {event['name']} at {float(event['time_days']):.4f} modeled days"
                for event in result.event_summary
            )
            or "  No configured boundary was crossed in this finite run"
        )
        correction_delta_v = sum(burn.delta_v_mps for burn in simulation.correction_burns)
        summary = (
            "WHAT WAS SOLVED\n"
            f"{ORBIT_MODEL_DESCRIPTIONS[simulation.configuration.model]}\n\n"
            "WHAT HAPPENED\n"
            f"Initial altitude: {simulation.configuration.initial.altitude_m / 1_000.0:,.2f} km. "
            f"Calculated initial speed: {simulation.initial_speed_mps:,.3f} m/s. The run sampled "
            f"{result.time_s.size:,} displayed states and completed "
            f"{columns['revolutions_completed'][-1]:,.3f} modeled revolutions. Minimum altitude "
            f"was {np.min(columns['altitude_m']) / 1_000.0:,.3f} km.\n\n"
            "LIFETIME ANSWER\n"
            f"{simulation.survival_statement}\n\n"
            "EVENTS\n"
            f"{event_text}\n\n"
            "CORRECTIONS\n"
            f"{len(simulation.correction_burns)} idealized burn(s), total delta-v "
            f"{correction_delta_v:,.3f} m/s. Corrections are off in the prepared example.\n\n"
            "HOW TO READ 3D\n"
            "Orange is the propagated satellite path. C switches between satellite-scale and "
            "whole-system views; the time slider seeks the calculated states.\n\n"
            "SAVED EVIDENCE\n"
            f"CSV, JSON limitations report and publication plots: {run.output_directory}"
        )
        self._replace_text(self.orbit_summary, summary)
        self.orbit_canvas.update_idletasks()
        self.orbit_canvas.yview_moveto(1.0)
        self.status_var.set("Satellite propagation complete - finite-horizon result saved.")
        if open_playback:
            try:
                from aerognc.visualisation.orbit_sandbox import play_orbit_sandbox

                self.status_var.set("3D orbit player open - use the slider or press C for focus.")
                play_orbit_sandbox(simulation)
            except Exception as error:
                messagebox.showerror("Cannot open 3D orbit player", str(error))

    def _run_aircraft_batch(self) -> None:
        try:
            configuration = self._aircraft_configuration()
        except (ValueError, OSError) as error:
            messagebox.showerror("Aircraft input", str(error))
            return

        def operation() -> AircraftSandboxRun:
            from aerognc.simulation.aircraft_sandbox import (
                simulate_aircraft,
                write_aircraft_results,
            )
            from aerognc.visualisation.aircraft_sandbox import plot_aircraft_sandbox

            simulation = simulate_aircraft(configuration)
            output = configuration.output_directory
            write_aircraft_results(simulation, output)
            plot_aircraft_sandbox(simulation, output)
            return AircraftSandboxRun(simulation, output)

        self._background(
            "Running coefficient-driven quaternion aircraft equations hands-off...",
            operation,
            self._aircraft_complete,
        )

    def _aircraft_complete(self, run: AircraftSandboxRun) -> None:
        simulation = run.simulation
        result = simulation.result
        maxima = result.maximum_summary
        final_altitude = result.columns["altitude_m"][-1]
        final_speed = result.columns["true_airspeed_mps"][-1]
        summary = (
            "WHAT WAS SOLVED\n"
            "An 18-state nonlinear rigid-body plant propagated planet-centred position and "
            "velocity, quaternion attitude, body rates, fuel mass, three actuator positions "
            "and throttle. Aerodynamic CL, CD and Cm generated forces/moments at every RK4 "
            "stage.\n\n"
            "HANDS-OFF RESULT\n"
            f"After {result.time_s[-1]:,.2f} s, altitude was {final_altitude:,.2f} m and true "
            f"airspeed was {final_speed:,.2f} m/s. Maximum load factor was "
            f"{float(maxima['maximum_load_factor']['value']):,.3f} g; maximum actual heading "
            f"turn rate was {float(maxima['maximum_turn_rate']['value']):,.3f} deg/s.\n\n"
            "STALL / SPACE\n"
            f"Modeled stall duration: {simulation.stalled_duration_s:,.3f} s. "
            f"{simulation.interpretation}\n\n"
            "WHY YOUR INPUTS MATTER\n"
            "CL changes lift, CD changes drag, Cm changes pitch acceleration, mass divides force, "
            "and wing area scales aerodynamic loads. The imported mesh is deliberately visual "
            "only.\n\n"
            "SAVED EVIDENCE\n"
            f"CSV, limitations report, 3D path and coefficient dashboard: {run.output_directory}"
        )
        self._replace_text(self.aircraft_summary, summary)
        self.aircraft_canvas.update_idletasks()
        self.aircraft_canvas.yview_moveto(1.0)
        self.status_var.set("Hands-off aircraft run complete - numerical evidence saved.")

    def _fly_aircraft(self) -> None:
        if self.busy:
            messagebox.showinfo("AeroGNC-Lab", "Wait for the current calculation to finish.")
            return
        try:
            configuration = self._aircraft_configuration()
            mesh_path = self._aircraft_mesh_path()
            mesh_axes = self._aircraft_mesh_axes()
            real_time_factor = self._number(
                self.aircraft_vars["real_time_factor"], "Live speed factor"
            )
            control_mode = AIRCRAFT_CONTROL_MODE_CHOICES.get(
                self.aircraft_vars["control_mode"].get()
            )
            camera_mode = AIRCRAFT_CAMERA_CHOICES.get(self.aircraft_vars["camera"].get())
            trail_mode = AIRCRAFT_TRAIL_CHOICES.get(self.aircraft_vars["trail_mode"].get())
            trail_color = AIRCRAFT_TRAIL_COLOR_CHOICES.get(self.aircraft_vars["trail_color"].get())
            if None in (control_mode, camera_mode, trail_mode, trail_color):
                raise ValueError("Choose listed control, camera, trail, and colour options")
            from aerognc.visualisation.aircraft_experience import TrailSettings
            from aerognc.visualisation.aircraft_live import AircraftLivePlayer
            from aerognc.visualisation.mesh import load_triangle_mesh

            mesh = load_triangle_mesh(
                mesh_path,
                axis_convention=mesh_axes,
                transform=self._aircraft_mesh_transform(),
            )
            player = AircraftLivePlayer(
                configuration,
                mesh,
                real_time_factor=real_time_factor,
                camera_mode=cast("LiveCameraMode", camera_mode),
                enable_gamepad=self.aircraft_gamepad_var.get(),
                control_profile=self._aircraft_pilot_profile(),
                trail_settings=TrailSettings(
                    mode=cast("TrailMode", trail_mode),
                    fading_duration_s=self._number(
                        self.aircraft_vars["trail_duration"], "Fading trail duration"
                    ),
                    color_source=cast("TrailColorSource", trail_color),
                ),
                mesh_scale_mode=(
                    "true_scale"
                    if self.aircraft_vars["mesh_scale_mode"].get() == "True physical scale"
                    else "enlarged_marker"
                ),
                recorder_directory=Path(self.aircraft_vars["recorder_directory"].get()).resolve(),
                training_task=cast(
                    "TrainingTask",
                    AIRCRAFT_TRAINING_TASK_BY_PRESET.get(
                        AIRCRAFT_PRESET_CHOICES.get(self.aircraft_vars["preset"].get(), "")
                    ),
                ),
            )
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot start live aircraft", str(error))
            return
        self._active_aircraft_players.append(player)

        def release_player(_event: object, live_player: object = player) -> None:
            if live_player in self._active_aircraft_players:
                self._active_aircraft_players.remove(live_player)

        player.figure.canvas.mpl_connect("close_event", release_player)
        self.status_var.set(
            "Flight deck open and paused - click it, press Space, and press H for the control card."
        )
        try:
            player.show(block=False)
        except Exception as error:
            messagebox.showerror("Live aircraft stopped", str(error))

    def _run_tour(self, *, open_playback: bool) -> None:
        try:
            inputs = self._tour_inputs()
        except ValueError as error:
            messagebox.showerror("Planetary-tour input", str(error))
            return

        def operation() -> OrbitTourSimulation:
            simulation = run_orbit_tour_workbench(
                self.paths.orbit_tour_configuration,
                inputs,
            )
            write_orbit_tour_results(simulation, simulation.configuration.output_directory)
            return simulation

        self._background(
            "Solving transfer legs, capture orbit, powered departure and mass sequence...",
            operation,
            lambda simulation: self._tour_complete(
                simulation,
                inputs.playback_days_per_second,
                open_playback=open_playback,
            ),
        )

    def _tour_complete(
        self,
        simulation: OrbitTourSimulation,
        playback_days_per_second: float,
        *,
        open_playback: bool,
    ) -> None:
        tour = simulation.tour
        burns = "\n".join(
            f"  {index}. {burn.name}: change speed by {burn.delta_v_mps:,.2f} m/s; "
            f"use {burn.propellant_used_kg:,.2f} kg; {burn.mass_after_kg:,.2f} kg remains"
            for index, burn in enumerate(tour.burns, start=1)
        )
        outcome = "PASS" if simulation.assessment.all_pass else "DOES NOT MEET ALL LIMITS"
        initial_mass = tour.burns[0].mass_before_kg
        propellant_used = initial_mass - tour.final_mass_kg
        summary = (
            "WHAT HAPPENED\n"
            f"The spacecraft left {tour.departure_body.name}, followed a calculated transfer, "
            f"captured into orbit at {tour.assist_body.name}, completed "
            f"{tour.dwell_revolutions} whole parking-orbit revolution(s), accelerated out of "
            f"that orbit, and captured at {tour.destination_body.name}.\n\n"
            f"RESULT: {outcome}\n"
            "PASS means the simplified trajectory meets the entered delta-v, final-mass, event "
            "ordering, sphere-of-influence and endpoint checks. It is not a statement that a "
            "real mission is flight-ready.\n\n"
            "ENERGY AND MASS\n"
            f"Total ideal delta-v: {tour.total_delta_v_mps:,.3f} m/s. Propellant used: "
            f"{propellant_used:,.3f} kg. Final mass: {tour.final_mass_kg:,.3f} kg "
            f"(dry-mass floor {tour.dry_mass_kg:,.3f} kg). The modeled powered departure "
            f"gained {tour.departure_oberth_energy_gain_jpkg / 1.0e6:,.3f} MJ/kg of specific "
            "orbital energy.\n\n"
            "WHAT TO LOOK FOR IN 3D\n"
            "The curves are calculated transfer and parking-orbit samples. Event markers show "
            "when capture, orbiting, departure and arrival occur; they are not a prerecorded "
            "animation.\n\n"
            f"IDEAL BURN SEQUENCE\n{burns}\n\n"
            "SAVED EVIDENCE\n"
            f"Trajectory, event and assessment files: {simulation.configuration.output_directory}"
        )
        self._replace_text(self.tour_summary, summary)
        self.tour_canvas.update_idletasks()
        self.tour_canvas.yview_moveto(1.0)
        self.status_var.set("Planetary tour solved - result limits and endpoint checks evaluated.")
        if open_playback:
            try:
                from aerognc.simulation.orbit_tour_playback import orbit_tour_playback_mission
                from aerognc.visualisation.mission_control import play_interplanetary_mission

                mission = orbit_tour_playback_mission(simulation)
                self.status_var.set("3D planetary player open - press N to jump between events.")
                play_interplanetary_mission(
                    mission,
                    playback_days_per_second=playback_days_per_second,
                )
            except Exception as error:
                messagebox.showerror("Cannot open 3D planetary player", str(error))

    def _search_catalog(self) -> None:
        try:
            self.catalog_selection = search_exoplanet_catalog(
                self.exoplanet_catalog,
                self._catalog_inputs(),
            )
        except ValueError as error:
            messagebox.showerror("Catalog filter", str(error))
            return
        for item in self.catalog_tree.get_children():
            self.catalog_tree.delete(item)
        for index, planet in enumerate(self.catalog_selection):
            self.catalog_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    planet.name,
                    planet.host_name,
                    planet.discovery_method,
                    self._shown(planet.discovery_year),
                    self._shown(planet.system_distance_pc),
                    self._shown(planet.orbital_period_days),
                    self._shown(planet.radius_earth),
                    self._shown(planet.mass_earth),
                ),
            )
        positioned = sum(planet.has_3d_position for planet in self.catalog_selection)
        self.catalog_status_var.set(
            f"{len(self.catalog_selection):,} rows shown | {positioned:,} with 3D position | "
            f"snapshot {self.exoplanet_catalog.provenance.row_count:,} planets"
        )
        self.catalog_detail_var.set(
            "Selection is deterministic. Empty cells mean the source snapshot did not report "
            "a value."
        )

    @staticmethod
    def _shown(value: float | int | None) -> str:
        if value is None:
            return "-"
        return str(value) if isinstance(value, int) else f"{value:.7g}"

    def _catalog_row_selected(self, _event: object) -> None:
        selected = self.catalog_tree.selection()
        if not selected:
            return
        planet = self.catalog_selection[int(selected[0])]
        spectral = planet.stellar_spectral_type or "not reported"
        temperature = self._shown(planet.stellar_temperature_k)
        self.catalog_detail_var.set(
            f"{planet.name} | host {planet.host_name} | system has {planet.system_planet_count} "
            f"reported planet(s) | stellar type {spectral} | stellar temperature {temperature} K"
        )

    def _open_galaxy_explorer(self) -> None:
        try:
            from aerognc.visualisation.galaxy_explorer import explore_exoplanet_catalog

            explore_exoplanet_catalog(self.catalog_selection)
        except Exception as error:
            messagebox.showerror("Cannot open catalog map", str(error))

    def _open_advanced_designer(self) -> None:
        command = [
            sys.executable,
            "-m",
            "aerognc.cli",
            "mission-designer",
            "--catalog",
            str(self.paths.planetary_catalog),
            "--verified-config",
            str(self.paths.verified_interplanetary_configuration),
        ]
        try:
            subprocess.Popen(command, cwd=Path.cwd())
        except OSError as error:
            messagebox.showerror("Cannot open advanced designer", str(error))
        else:
            self.status_var.set("Advanced Mission Designer opened in a separate window.")


def launch_workbench(
    six_dof_configuration: str | Path = "configs/six_dof_nominal.yaml",
    orbit_tour_configuration: str | Path = "configs/orbit_assisted_tour.yaml",
    planetary_catalog: str | Path = "configs/fictional_planetary_system.yaml",
    verified_interplanetary_configuration: str | Path = (
        "configs/interplanetary_gravity_assist.yaml"
    ),
    exoplanet_csv: str | Path = "data/catalogs/nasa_confirmed_exoplanets.csv",
    exoplanet_metadata: str | Path = ("data/catalogs/nasa_confirmed_exoplanets.metadata.json"),
    milky_way_metadata: str | Path = "data/catalogs/milky_way_metadata.yaml",
    solar_system_planets: str | Path = "data/catalogs/solar_system_planets.csv",
    project_file: str | Path | None = "projects/portfolio_demo.aerognc.yaml",
    orbit_sandbox_configuration: str | Path = "configs/orbit_sandbox.yaml",
    aircraft_configuration: str | Path = "configs/aircraft_sandbox.yaml",
    aircraft_mesh: str | Path = "assets/models/aquila_x1.obj",
) -> None:
    """Load verified local resources and run the unified desktop event loop."""
    paths = WorkbenchPaths(
        six_dof_configuration=Path(six_dof_configuration).resolve(),
        orbit_tour_configuration=Path(orbit_tour_configuration).resolve(),
        planetary_catalog=Path(planetary_catalog).resolve(),
        verified_interplanetary_configuration=Path(verified_interplanetary_configuration).resolve(),
        exoplanet_csv=Path(exoplanet_csv).resolve(),
        exoplanet_metadata=Path(exoplanet_metadata).resolve(),
        milky_way_metadata=Path(milky_way_metadata).resolve(),
        solar_system_planets=Path(solar_system_planets).resolve(),
        project_file=None if project_file is None else Path(project_file).resolve(),
        orbit_sandbox_configuration=Path(orbit_sandbox_configuration).resolve(),
        aircraft_configuration=Path(aircraft_configuration).resolve(),
        aircraft_mesh=Path(aircraft_mesh).resolve(),
    )
    paths.validate()
    fictional_catalog = load_planetary_catalog(paths.planetary_catalog)
    catalog = load_workbench_catalog(paths.exoplanet_csv, paths.exoplanet_metadata)
    milky_way = load_milky_way_metadata(paths.milky_way_metadata)
    solar_planets = load_solar_system_planets(paths.solar_system_planets)
    root = tk.Tk()
    AeroGNCWorkbenchApp(
        root,
        paths,
        fictional_catalog,
        catalog,
        milky_way,
        solar_planets,
    )
    root.mainloop()
