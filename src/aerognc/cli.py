"""Command-line workflows for reproducible local simulations."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from aerognc.configuration import ConfigurationError, load_three_dof_configuration
from aerognc.simulation.logging import write_result_csv, write_summary_json
from aerognc.simulation.simulator import simulate_three_dof

LOGGER = logging.getLogger("aerognc")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerognc",
        description="Public-safe fictional research-rocket GNC verification workflows.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.8.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a configured 3-DOF ascent")
    run.add_argument("--config", type=Path, required=True, help="scenario YAML path")
    run.add_argument("--output", type=Path, help="override output directory")
    run.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    rotating_ascent = subparsers.add_parser(
        "rotating-ascent",
        help="run a geodetic ascent on a synthetic rotating oblate planet",
    )
    rotating_ascent.add_argument("--config", type=Path, required=True, help="scenario YAML path")
    rotating_ascent.add_argument("--output", type=Path, help="override output directory")
    rotating_ascent.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    rotating_six_dof = subparsers.add_parser(
        "rotating-six-dof",
        help="run quaternion 6-DOF flight on a synthetic rotating oblate planet",
    )
    rotating_six_dof.add_argument("--config", type=Path, required=True, help="scenario YAML path")
    rotating_six_dof.add_argument("--output", type=Path, help="override output directory")
    rotating_six_dof.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    multistage = subparsers.add_parser(
        "multistage-recovery",
        help="run a fictional vertical staging and deployable-recovery benchmark",
    )
    multistage.add_argument("--config", type=Path, required=True, help="scenario YAML path")
    multistage.add_argument("--output", type=Path, help="override output directory")
    multistage.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    aero_analysis = subparsers.add_parser(
        "aero-analysis",
        help="inspect a synthetic regular-grid aerodynamic database",
    )
    aero_analysis.add_argument("--config", type=Path, required=True, help="3-DOF scenario YAML")
    aero_analysis.add_argument("--output", type=Path, help="override output directory")
    aero_analysis.add_argument("--mach", type=float, default=0.8, help="nominal Mach number")
    aero_analysis.add_argument(
        "--alpha-deg", type=float, default=0.0, help="nominal angle of attack [deg]"
    )
    aero_analysis.add_argument(
        "--beta-deg", type=float, default=0.0, help="nominal sideslip angle [deg]"
    )
    aero_analysis.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    play = subparsers.add_parser("play", help="play an interactive 3-DOF flight animation")
    play.add_argument("--config", type=Path, required=True, help="scenario YAML path")
    play.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="initial playback speed in simulation seconds per real second (default: 4)",
    )
    play.add_argument("--fps", type=int, default=30, help="display/export frame rate")
    play.add_argument("--repeat", action="store_true", help="restart automatically at impact")
    play.add_argument("--save-gif", type=Path, help="optional animated GIF output path")
    play.add_argument(
        "--no-window",
        action="store_true",
        help="do not open a window; requires --save-gif",
    )
    play_3d = subparsers.add_parser(
        "play-3d", help="play an interactive quaternion 6-DOF flight in 3D"
    )
    play_3d.add_argument("--config", type=Path, required=True, help="6-DOF scenario YAML path")
    play_3d.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help="initial playback speed in simulation seconds per real second (default: 2)",
    )
    play_3d.add_argument("--fps", type=int, default=30, help="display/export frame rate")
    play_3d.add_argument("--repeat", action="store_true", help="restart at scenario completion")
    play_3d.add_argument(
        "--camera",
        choices=("orbit", "chase", "top", "side", "free"),
        default="orbit",
        help="initial camera mode (default: orbit)",
    )
    play_3d.add_argument("--save-gif", type=Path, help="optional animated GIF output path")
    play_3d.add_argument(
        "--no-window",
        action="store_true",
        help="do not open a window; requires --save-gif",
    )
    six_dof = subparsers.add_parser("six-dof", help="run a configured quaternion 6-DOF ascent")
    six_dof.add_argument("--config", type=Path, required=True, help="6-DOF scenario YAML path")
    six_dof.add_argument("--output", type=Path, help="override output directory")
    six_dof.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    interplanetary = subparsers.add_parser(
        "interplanetary",
        help="solve and play a fictional civilian gravity-assist mission",
    )
    interplanetary.add_argument(
        "--config", type=Path, required=True, help="interplanetary mission YAML path"
    )
    interplanetary.add_argument("--output", type=Path, help="override result directory")
    interplanetary.add_argument(
        "--speed-days-per-second",
        type=float,
        default=40.0,
        help="mission playback rate (default: 40 days/s)",
    )
    interplanetary.add_argument("--fps", type=int, default=24, help="display/export frame rate")
    interplanetary.add_argument("--repeat", action="store_true", help="loop the mission player")
    interplanetary.add_argument(
        "--camera",
        choices=("system", "spacecraft", "assist", "destination", "top", "free"),
        default="system",
        help="initial mission-control camera",
    )
    interplanetary.add_argument("--save-gif", type=Path, help="optional mission GIF path")
    interplanetary.add_argument("--save-snapshot", type=Path, help="optional dashboard PNG path")
    interplanetary.add_argument(
        "--no-window",
        action="store_true",
        help="solve and write outputs without opening mission control",
    )
    orbit_tour = subparsers.add_parser(
        "orbit-tour",
        help="simulate a fictional capture, parking-orbit dwell, and planetary departure",
    )
    orbit_tour.add_argument("--config", type=Path, required=True, help="orbit-tour YAML path")
    orbit_tour.add_argument("--output", type=Path, help="override result directory")
    orbit_tour.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    orbit_sandbox = subparsers.add_parser(
        "orbit-sandbox",
        help="simulate free, two-body, restricted/full N-body, or drag-decay satellite motion",
    )
    orbit_sandbox.add_argument("--config", type=Path, required=True, help="orbit sandbox YAML")
    orbit_sandbox.add_argument("--output", type=Path, help="override result directory")
    orbit_sandbox.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    orbit_sandbox.add_argument(
        "--play", action="store_true", help="open seekable 3D playback after propagation"
    )
    aircraft = subparsers.add_parser(
        "aircraft",
        help="run the coefficient-driven fictional fixed-wing batch scenario",
    )
    aircraft.add_argument("--config", type=Path, required=True, help="aircraft YAML path")
    aircraft.add_argument("--output", type=Path, help="override result directory")
    aircraft.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    fly_aircraft = subparsers.add_parser(
        "fly-aircraft",
        help="fly the nonlinear fictional aircraft with keyboard or XInput controller",
    )
    fly_aircraft.add_argument("--config", type=Path, required=True, help="aircraft YAML path")
    fly_aircraft.add_argument(
        "--mesh",
        type=Path,
        default=Path("assets/models/aquila_x1.obj"),
        help="visual-only OBJ/STL aircraft mesh",
    )
    fly_aircraft.add_argument(
        "--mesh-axes",
        choices=("body_frd", "x_forward_z_up", "y_forward_z_up"),
        default="body_frd",
        help="source mesh axis convention",
    )
    fly_aircraft.add_argument("--fps", type=int, default=30, help="display frame rate")
    fly_aircraft.add_argument(
        "--real-time-factor",
        type=float,
        default=1.0,
        help="simulation seconds per real second",
    )
    fly_aircraft.add_argument(
        "--camera",
        choices=("chase", "cockpit", "orbit", "top", "free"),
        default="chase",
        help="initial 3D camera",
    )
    fly_aircraft.add_argument(
        "--no-gamepad", action="store_true", help="disable optional Windows XInput polling"
    )
    fly_aircraft.add_argument(
        "--control-mode",
        choices=("stability_assisted", "direct"),
        help="override the selected pilot profile's assisted/direct mode",
    )
    fly_aircraft.add_argument(
        "--pilot-profile",
        type=Path,
        default=Path("configs/pilot_profiles/accessible.json"),
        help="versioned JSON response, sensitivity, binding, and assistance profile",
    )
    fly_aircraft.add_argument(
        "--trail", choices=("off", "fading", "full"), default="fading"
    )
    fly_aircraft.add_argument("--trail-duration", type=float, default=45.0)
    fly_aircraft.add_argument(
        "--trail-color",
        choices=("constant", "altitude", "airspeed"),
        default="constant",
    )
    fly_aircraft.add_argument(
        "--mesh-scale",
        choices=("enlarged_marker", "true_scale"),
        default="enlarged_marker",
    )
    fly_aircraft.add_argument(
        "--recording-directory",
        type=Path,
        default=Path("results/aircraft_live"),
        help="F9 recorder/GIF/screenshot output directory",
    )
    fly_aircraft.add_argument(
        "--demo",
        action="store_true",
        help="enable reproducible civilian demonstration commands after Space starts flight",
    )
    fly_aircraft.add_argument(
        "--training-task",
        choices=(
            "altitude_speed_hold",
            "coordinated_360_turn",
            "stall_recovery",
            "research_altitude_crossing",
        ),
        help="write an objective civilian exercise score whenever F9 saves",
    )
    replay_aircraft = subparsers.add_parser(
        "replay-aircraft",
        help="seek and play exact states from an aircraft live-recorder CSV",
    )
    replay_aircraft.add_argument("--config", type=Path, required=True)
    replay_aircraft.add_argument("--recording", type=Path, required=True)
    replay_aircraft.add_argument(
        "--mesh", type=Path, default=Path("assets/models/aquila_x1.obj")
    )
    replay_aircraft.add_argument("--playback-factor", type=float, default=1.0)
    aircraft_aero_compare = subparsers.add_parser(
        "aircraft-aero-compare",
        help="compare analytic and synthetic static-table aircraft coefficients",
    )
    aircraft_aero_compare.add_argument(
        "--config", type=Path, default=Path("configs/aircraft_sandbox.yaml")
    )
    aircraft_aero_compare.add_argument(
        "--table", type=Path, default=Path("data/aerodynamics/aquila_x1_static.csv")
    )
    aircraft_aero_compare.add_argument(
        "--output",
        type=Path,
        default=Path("results/aircraft_aerodynamics/backend_comparison.json"),
    )
    launch_window = subparsers.add_parser(
        "launch-window",
        help="run deterministic fictional launch-window grid and bounded refinement",
    )
    launch_window.add_argument("--config", type=Path, required=True, help="launch-window YAML path")
    launch_window.add_argument("--output", type=Path, help="override result directory")
    launch_window.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    catalog = subparsers.add_parser(
        "catalog",
        help="browse the provenance-tagged Milky Way and confirmed-exoplanet data layer",
    )
    catalog.add_argument(
        "--csv",
        type=Path,
        default=Path("data/catalogs/nasa_confirmed_exoplanets.csv"),
        help="NASA confirmed-exoplanet snapshot CSV",
    )
    catalog.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/catalogs/nasa_confirmed_exoplanets.metadata.json"),
        help="snapshot provenance JSON",
    )
    catalog.add_argument(
        "--galaxy-metadata",
        type=Path,
        default=Path("data/catalogs/milky_way_metadata.yaml"),
        help="Milky Way context YAML",
    )
    catalog.add_argument(
        "--solar-system",
        type=Path,
        default=Path("data/catalogs/solar_system_planets.csv"),
        help="eight-planet Solar System table",
    )
    catalog.add_argument("--query", default="", help="planet or host-name substring")
    catalog.add_argument("--method", help="exact discovery-method filter")
    catalog.add_argument("--max-distance-pc", type=float, help="maximum reported distance [pc]")
    catalog.add_argument("--min-year", type=int, help="minimum discovery year")
    catalog.add_argument("--max-year", type=int, help="maximum discovery year")
    catalog.add_argument("--limit", type=int, help="maximum rows in the filtered output")
    catalog.add_argument(
        "--output",
        type=Path,
        default=Path("results/galaxy_catalog"),
        help="catalog report directory",
    )
    catalog.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    workbench = subparsers.add_parser(
        "workbench",
        help="open the unified rocket, planetary-tour, and Milky Way desktop UI",
    )
    workbench.add_argument(
        "--six-dof-config",
        type=Path,
        default=Path("configs/six_dof_nominal.yaml"),
        help="verified editable rocket scenario YAML",
    )
    workbench.add_argument(
        "--orbit-tour-config",
        type=Path,
        default=Path("configs/orbit_assisted_tour.yaml"),
        help="verified capture/orbit/departure tour YAML",
    )
    workbench.add_argument(
        "--planetary-catalog",
        type=Path,
        default=Path("configs/fictional_planetary_system.yaml"),
        help="fictional executable planetary catalog YAML",
    )
    workbench.add_argument(
        "--verified-interplanetary-config",
        type=Path,
        default=Path("configs/interplanetary_gravity_assist.yaml"),
        help="advanced Mission Designer reference YAML",
    )
    workbench.add_argument(
        "--exoplanet-csv",
        type=Path,
        default=Path("data/catalogs/nasa_confirmed_exoplanets.csv"),
        help="checksummed NASA confirmed-exoplanet snapshot",
    )
    workbench.add_argument(
        "--exoplanet-metadata",
        type=Path,
        default=Path("data/catalogs/nasa_confirmed_exoplanets.metadata.json"),
        help="snapshot provenance and checksum JSON",
    )
    workbench.add_argument(
        "--milky-way-metadata",
        type=Path,
        default=Path("data/catalogs/milky_way_metadata.yaml"),
        help="sourced approximate Milky Way context YAML",
    )
    workbench.add_argument(
        "--solar-system-planets",
        type=Path,
        default=Path("data/catalogs/solar_system_planets.csv"),
        help="sourced eight-planet Solar System CSV",
    )
    workbench.add_argument(
        "--project-file",
        type=Path,
        default=Path("projects/portfolio_demo.aerognc.yaml"),
        help="engineering project opened in the Saved Runs tab",
    )
    software_loopback = subparsers.add_parser(
        "software-loopback",
        help="exercise future-HIL packets and watchdog in deterministic logical time",
    )
    software_loopback.add_argument("--samples", type=int, default=500)
    software_loopback.add_argument("--sample-period-ms", type=float, default=10.0)
    software_loopback.add_argument("--latency-ms", type=float, default=2.0)
    software_loopback.add_argument("--jitter-ms", type=float, default=0.5)
    software_loopback.add_argument("--loss-percent", type=float, default=0.0)
    software_loopback.add_argument("--duplicate-percent", type=float, default=0.0)
    software_loopback.add_argument("--deadline-ms", type=float, default=25.0)
    software_loopback.add_argument("--timeout-ms", type=float, default=40.0)
    software_loopback.add_argument("--seed", type=int, default=218)
    software_loopback.add_argument(
        "--output",
        type=Path,
        default=Path("results/software_loopback/software_loopback_report.json"),
    )
    udp_loopback = subparsers.add_parser(
        "udp-loopback",
        help="exercise the versioned HIL packet boundary over localhost UDP",
    )
    udp_loopback.add_argument("--samples", type=int, default=100)
    udp_loopback.add_argument("--sample-period-ms", type=float, default=10.0)
    udp_loopback.add_argument("--receive-timeout-ms", type=float, default=100.0)
    udp_loopback.add_argument("--watchdog-timeout-ms", type=float, default=30.0)
    udp_loopback.add_argument(
        "--output",
        type=Path,
        default=Path("results/software_loopback/udp_loopback_report.json"),
    )
    fmi_interface = subparsers.add_parser(
        "fmi-interface",
        help="write the non-executable FMI 3.0 attitude-controller interface contract",
    )
    fmi_interface.add_argument(
        "--output",
        type=Path,
        default=Path("fmi_validation/attitude_controller_interface"),
        help="contract output directory",
    )
    mission_designer = subparsers.add_parser(
        "mission-designer",
        help="open the guided civilian interplanetary mission-design desktop UI",
    )
    mission_designer.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/fictional_planetary_system.yaml"),
        help="fictional planetary catalog YAML path",
    )
    mission_designer.add_argument(
        "--verified-config",
        type=Path,
        default=Path("configs/interplanetary_gravity_assist.yaml"),
        help="verified restricted N-body example YAML path",
    )
    attitude = subparsers.add_parser("attitude", help="compare closed-loop attitude controllers")
    attitude.add_argument("--config", type=Path, required=True, help="attitude benchmark YAML path")
    attitude.add_argument("--output", type=Path, help="override output directory")
    attitude.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    navigation = subparsers.add_parser("navigation", help="run the synthetic navigation demo")
    navigation.add_argument("--config", type=Path, required=True, help="navigation YAML path")
    navigation.add_argument("--output", type=Path, help="override output directory")
    navigation.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    advanced_navigation = subparsers.add_parser(
        "advanced-navigation",
        help="run rotating strapdown, delayed ESKF, integrity, and consistency checks",
    )
    advanced_navigation.add_argument(
        "--config", type=Path, required=True, help="advanced-navigation YAML path"
    )
    advanced_navigation.add_argument("--output", type=Path, help="override output directory")
    advanced_navigation.add_argument(
        "--consistency-runs",
        type=int,
        help="override the configured seeded consistency-run count",
    )
    advanced_navigation.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    monte_carlo = subparsers.add_parser("monte-carlo", help="run coupled seeded dispersions")
    monte_carlo.add_argument("--config", type=Path, required=True, help="Monte Carlo YAML path")
    monte_carlo.add_argument("--samples", type=int, help="override sample count")
    monte_carlo.add_argument("--workers", type=int, help="override process count")
    monte_carlo.add_argument("--output", type=Path, help="override output directory")
    monte_carlo.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    flight_test = subparsers.add_parser("flight-test", help="run synthetic flight-data evaluation")
    flight_test.add_argument(
        "--config", type=Path, required=True, help="navigation/sensor YAML path"
    )
    flight_test.add_argument("--output", type=Path, help="override output directory")
    flight_test.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    flight_data_identification = subparsers.add_parser(
        "flight-data-identification",
        help="align asynchronous synthetic logs and identify a pitch plant",
    )
    flight_data_identification.add_argument(
        "--config", type=Path, required=True, help="flight-data identification YAML path"
    )
    flight_data_identification.add_argument("--output", type=Path, help="override output directory")
    flight_data_identification.add_argument(
        "--no-plots", action="store_true", help="skip PNG generation"
    )
    flight_analysis = subparsers.add_parser(
        "flight-analysis",
        help="run trim, linearisation, LQR, margins, gain schedule, and SIL timing",
    )
    flight_analysis.add_argument(
        "--config", type=Path, required=True, help="flight-control analysis YAML path"
    )
    flight_analysis.add_argument("--output", type=Path, help="override output directory")
    flight_analysis.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    flight_envelope = subparsers.add_parser(
        "flight-envelope",
        help="verify trim, modes, authority, and scheduled control across an ascent envelope",
    )
    flight_envelope.add_argument(
        "--config", type=Path, required=True, help="flight-envelope YAML path"
    )
    flight_envelope.add_argument("--output", type=Path, help="override output directory")
    flight_envelope.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    constrained_ascent = subparsers.add_parser(
        "constrained-ascent",
        help="optimize and verify a public-safe max-Q/load/AoA ascent reference",
    )
    constrained_ascent.add_argument(
        "--config", type=Path, required=True, help="constrained-ascent YAML path"
    )
    constrained_ascent.add_argument("--output", type=Path, help="override output directory")
    constrained_ascent.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    benchmark = subparsers.add_parser(
        "benchmark",
        help="measure a local 3-DOF run with explicit non-real-time resource budgets",
    )
    benchmark.add_argument("--config", type=Path, required=True, help="3-DOF scenario YAML path")
    benchmark.add_argument("--repetitions", type=int, default=3, help="measured repetition count")
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmarks/three_dof_benchmark.json"),
        help="benchmark JSON output path",
    )
    benchmark.add_argument("--max-wall-time-s", type=float)
    benchmark.add_argument("--max-cpu-time-s", type=float)
    benchmark.add_argument("--max-peak-memory-mb", type=float)
    benchmark.add_argument("--min-samples-per-second", type=float)
    benchmark.add_argument("--min-steps-per-second", type=float)
    diagnose = subparsers.add_parser(
        "diagnose",
        help="report runtime, package, data, output, and optional-tool readiness",
    )
    diagnose.add_argument(
        "--project-root", type=Path, default=Path("."), help="AeroGNC-Lab repository root"
    )
    diagnose.add_argument(
        "--result-directory",
        type=Path,
        default=Path("results"),
        help="result location to permission-check",
    )
    diagnose.add_argument("--output", type=Path, help="JSON report path")
    project = subparsers.add_parser(
        "project",
        help="create, validate, run, inspect, compare, and report engineering projects",
    )
    project_commands = project.add_subparsers(dest="project_command", required=True)
    project_init = project_commands.add_parser("init", help="create an empty project workspace")
    project_init.add_argument("directory", type=Path, help="new project directory")
    project_init.add_argument("--name", required=True, help="portable project name")
    project_validate = project_commands.add_parser(
        "validate", help="validate project schema, paths, and workflows"
    )
    project_validate.add_argument("project", type=Path, help="project YAML path")
    project_inspect = project_commands.add_parser(
        "inspect", help="show project scenarios and result-store location"
    )
    project_inspect.add_argument("project", type=Path, help="project YAML path")
    project_run = project_commands.add_parser("run", help="execute and store one scenario")
    project_run.add_argument("project", type=Path, help="project YAML path")
    project_run.add_argument("scenario", help="scenario name")
    project_list = project_commands.add_parser("list", help="list immutable stored runs")
    project_list.add_argument("project", type=Path, help="project YAML path")
    project_list.add_argument("--scenario", help="optional scenario-name filter")
    project_list.add_argument(
        "--status", choices=("completed", "failed", "cancelled"), help="terminal-status filter"
    )
    project_compare = project_commands.add_parser(
        "compare", help="compare two compatible stored runs"
    )
    project_compare.add_argument("project", type=Path, help="project YAML path")
    project_compare.add_argument("baseline_run", help="baseline run identifier")
    project_compare.add_argument("candidate_run", help="candidate run identifier")
    project_compare.add_argument(
        "--channels",
        help="comma-separated channel names; default is every baseline channel",
    )
    project_compare.add_argument("--samples", type=int, help="aligned comparison sample count")
    project_compare.add_argument("--output", type=Path, help="comparison JSON path")
    project_report = project_commands.add_parser(
        "report", help="regenerate a self-contained run report"
    )
    project_report.add_argument("project", type=Path, help="project YAML path")
    project_report.add_argument("run_id", help="stored run identifier")
    project_report.add_argument("--output", type=Path, help="HTML report path")

    mission = subparsers.add_parser(
        "mission", help="validate a waypoint mission file"
    )
    mission_commands = mission.add_subparsers(dest="mission_command", required=True)
    mission_validate = mission_commands.add_parser(
        "validate", help="load and validate a waypoint mission (schema + flight envelope)"
    )
    mission_validate.add_argument("mission", type=Path, help="mission YAML path")

    waypoint = subparsers.add_parser(
        "waypoint",
        help="fly a waypoint mission in the internal fixed-wing simulator",
    )
    waypoint.add_argument("--mission", type=Path, required=True, help="mission YAML path")
    waypoint.add_argument(
        "--guidance",
        choices=("direct_bearing", "line_of_sight", "l1_guidance", "vector_field"),
        default="vector_field",
        help="lateral guidance mode (default: vector_field)",
    )
    waypoint.add_argument(
        "--wind-north-mps", type=float, default=0.0, help="steady north wind [m/s]"
    )
    waypoint.add_argument(
        "--wind-east-mps", type=float, default=0.0, help="steady east wind [m/s]"
    )
    waypoint.add_argument("--dt-s", type=float, default=0.05, help="integration step [s]")
    waypoint.add_argument(
        "--max-time-s", type=float, default=900.0, help="simulation time limit [s]"
    )
    waypoint.add_argument("--output", type=Path, help="output directory for log + plot")
    waypoint.add_argument("--no-plots", action="store_true", help="skip PNG generation")

    mission_planner = subparsers.add_parser(
        "mission-planner",
        help="open the interactive map-based waypoint mission planner (Tk)",
    )
    mission_planner.add_argument(
        "--mission", type=Path, help="optional mission YAML to open on launch"
    )

    rpo = subparsers.add_parser(
        "rpo",
        help="plan a satellite rendezvous / proximity-operations approach (non-weapon)",
    )
    rpo.add_argument(
        "--altitude-km", type=float, default=500.0, help="target circular-orbit altitude [km]"
    )
    rpo.add_argument(
        "--start-behind-m", type=float, default=800.0,
        help="chaser initial along-track offset behind the target [m]",
    )
    rpo.add_argument(
        "--leg-time-s", type=float, default=500.0, help="coast time per approach leg [s]"
    )
    rpo.add_argument("--output", type=Path, help="output directory for plot + JSON")
    rpo.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    return parser


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _run_three_dof(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    configuration = load_three_dof_configuration(config_path)
    output = output_override or configuration.simulation.output_directory
    LOGGER.info("scenario=%s vehicle=%s", configuration.simulation.name, configuration.vehicle.name)
    result = simulate_three_dof(configuration)
    csv_path = write_result_csv(result, output / "trajectory.csv")
    summary_path = write_summary_json(result, output / "summary.json")
    if no_plots:
        figure_paths: tuple[Path, ...] = ()
    else:
        from aerognc.visualisation import plot_three_dof_results

        figure_paths = plot_three_dof_results(result, output)
    LOGGER.info(
        "completed samples=%d simulated_time_s=%.3f execution_time_s=%.3f",
        result.time_s.size,
        result.time_s[-1],
        result.execution_time_s,
    )
    for event in result.event_summary:
        LOGGER.info(
            "event=%s time_s=%.3f altitude_m=%.1f speed_mps=%.1f",
            event["name"],
            event["time_s"],
            event["altitude_m"],
            event["speed_mps"],
        )
    LOGGER.info("trajectory=%s", csv_path)
    LOGGER.info("summary=%s", summary_path)
    for path in figure_paths:
        LOGGER.info("figure=%s", path)
    print(json.dumps(result.maximum_summary, indent=2, sort_keys=True))
    return 0


def _run_rotating_ascent(
    config_path: Path,
    output_override: Path | None,
    no_plots: bool,
) -> int:
    from aerognc.configuration import load_rotating_ascent_configuration
    from aerognc.simulation.rotating_ascent import simulate_rotating_ascent

    configuration = load_rotating_ascent_configuration(config_path)
    output = output_override or configuration.output_directory
    result = simulate_rotating_ascent(configuration)
    csv_path = write_result_csv(result, output / "rotating_ascent_trajectory.csv")
    summary_path = write_summary_json(result, output / "rotating_ascent_summary.json")
    if no_plots:
        figure_path = None
    else:
        from aerognc.visualisation import plot_rotating_ascent

        figure_path = plot_rotating_ascent(result, output)
    LOGGER.info(
        "rotating ascent planet=%s samples=%d execution_time_s=%.3f",
        configuration.planet.name,
        result.time_s.size,
        result.execution_time_s,
    )
    for event in result.event_summary:
        LOGGER.info(
            "event=%s time_s=%.3f altitude_m=%.1f speed_mps=%.1f",
            event["name"],
            event["time_s"],
            event["altitude_m"],
            event["speed_mps"],
        )
    LOGGER.info("trajectory=%s summary=%s", csv_path, summary_path)
    if figure_path is not None:
        LOGGER.info("figure=%s", figure_path)
    print(json.dumps(result.maximum_summary, indent=2, sort_keys=True))
    return 0


def _run_rotating_six_dof(
    config_path: Path,
    output_override: Path | None,
    no_plots: bool,
) -> int:
    from aerognc.configuration import load_rotating_six_dof_configuration
    from aerognc.simulation.rotating_six_dof import simulate_rotating_six_dof

    configuration = load_rotating_six_dof_configuration(config_path)
    output = output_override or configuration.output_directory
    result = simulate_rotating_six_dof(configuration)
    csv_path = write_result_csv(result, output / "rotating_six_dof_trajectory.csv")
    summary_path = write_summary_json(result, output / "rotating_six_dof_summary.json")
    if no_plots:
        figure_path = None
    else:
        from aerognc.visualisation.six_dof import plot_six_dof_results

        figure_path = plot_six_dof_results(result, output)
    LOGGER.info(
        "rotating 6-DOF planet=%s samples=%d execution_time_s=%.3f",
        configuration.rotating_planet.planet.name,
        result.time_s.size,
        result.execution_time_s,
    )
    LOGGER.info("trajectory=%s summary=%s", csv_path, summary_path)
    if figure_path is not None:
        LOGGER.info("figure=%s", figure_path)
    print(json.dumps(result.maximum_summary, indent=2, sort_keys=True))
    return 0


def _run_multistage_recovery(
    config_path: Path,
    output_override: Path | None,
    no_plots: bool,
) -> int:
    from aerognc.configuration import load_multistage_recovery_configuration
    from aerognc.simulation.multistage_recovery import (
        simulate_configured_multistage_recovery,
    )

    configuration = load_multistage_recovery_configuration(config_path)
    output = output_override or configuration.output_directory
    result = simulate_configured_multistage_recovery(configuration)
    csv_path = write_result_csv(result, output / "multistage_recovery_trajectory.csv")
    summary_path = write_summary_json(result, output / "multistage_recovery_summary.json")
    if no_plots:
        figure_path = None
    else:
        from aerognc.visualisation import plot_multistage_recovery

        figure_path = plot_multistage_recovery(result, output)
    LOGGER.info(
        "multistage/recovery samples=%d events=%d execution_time_s=%.3f",
        result.time_s.size,
        len(result.events),
        result.execution_time_s,
    )
    LOGGER.info("trajectory=%s summary=%s", csv_path, summary_path)
    if figure_path is not None:
        LOGGER.info("figure=%s", figure_path)
    print(json.dumps(result.maximum_summary, indent=2, sort_keys=True))
    return 0


def _run_aero_analysis(
    config_path: Path,
    output_override: Path | None,
    mach: float,
    alpha_deg: float,
    beta_deg: float,
    no_plots: bool,
) -> int:
    import numpy as np

    from aerognc.vehicle.aero_database import (
        AerodynamicCondition,
        TabulatedAerodynamicDatabase,
    )
    from aerognc.verification.aero_database import (
        analyze_aerodynamic_database,
        write_aerodynamic_database_analysis,
    )

    configuration = load_three_dof_configuration(config_path)
    output = output_override or Path("results/aerodynamic_database_analysis")
    condition = AerodynamicCondition(
        mach=mach,
        alpha_rad=float(np.deg2rad(alpha_deg)),
        beta_rad=float(np.deg2rad(beta_deg)),
    )
    analysis = analyze_aerodynamic_database(configuration, condition)
    report_path = write_aerodynamic_database_analysis(analysis, output)
    provider = configuration.vehicle.aerodynamics.coefficient_provider
    if not isinstance(provider, TabulatedAerodynamicDatabase):
        raise ValueError("scenario does not use a tabulated aerodynamic database")
    if no_plots:
        figure_path = None
    else:
        from aerognc.visualisation.aero_database import plot_aerodynamic_database

        figure_path = plot_aerodynamic_database(provider, output, condition)
    LOGGER.info("aerodynamic database analysis=%s", report_path)
    if figure_path is not None:
        LOGGER.info("figure=%s", figure_path)
    print(json.dumps(analysis.as_dict(), indent=2, sort_keys=True))
    return 0


def _run_playback(
    config_path: Path,
    playback_speed: float,
    frames_per_second: int,
    repeat: bool,
    save_gif: Path | None,
    no_window: bool,
) -> int:
    if no_window and save_gif is None:
        raise ValueError("--no-window requires --save-gif")
    from aerognc.visualisation.playback import play_three_dof

    configuration = load_three_dof_configuration(config_path)
    result = simulate_three_dof(configuration)
    LOGGER.info(
        "playback scenario=%s duration_s=%.3f speed=%.2fx fps=%d",
        result.scenario_name,
        result.time_s[-1],
        playback_speed,
        frames_per_second,
    )
    LOGGER.info("controls: Space=play/pause R=restart Left/Right=seek Up/Down=speed")
    output = play_three_dof(
        result,
        playback_speed=playback_speed,
        frames_per_second=frames_per_second,
        repeat=repeat,
        save_gif=save_gif,
        show_window=not no_window,
    )
    if output is not None:
        LOGGER.info("animation=%s", output)
    return 0


def _run_playback_3d(
    config_path: Path,
    playback_speed: float,
    frames_per_second: int,
    repeat: bool,
    camera_mode: str,
    save_gif: Path | None,
    no_window: bool,
) -> int:
    if no_window and save_gif is None:
        raise ValueError("--no-window requires --save-gif")
    from aerognc.configuration import load_six_dof_configuration
    from aerognc.simulation.six_dof_simulator import simulate_six_dof
    from aerognc.visualisation.playback_3d import (
        CAMERA_MODES,
        play_six_dof_3d,
    )

    if camera_mode not in CAMERA_MODES:
        raise ValueError(f"camera mode must be one of {CAMERA_MODES}")
    configuration = load_six_dof_configuration(config_path)
    result = simulate_six_dof(configuration)
    LOGGER.info(
        "3D playback scenario=%s duration_s=%.3f speed=%.2fx fps=%d camera=%s",
        result.scenario_name,
        result.time_s[-1],
        playback_speed,
        frames_per_second,
        camera_mode,
    )
    LOGGER.info(
        "controls: Space=play/pause R=restart Left/Right=seek Up/Down=speed "
        "C=camera; drag mouse in free camera"
    )
    output = play_six_dof_3d(
        result,
        playback_speed=playback_speed,
        frames_per_second=frames_per_second,
        repeat=repeat,
        camera_mode=camera_mode,
        save_gif=save_gif,
        show_window=not no_window,
    )
    if output is not None:
        LOGGER.info("animation=%s", output)
    return 0


def _run_six_dof(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_six_dof_configuration
    from aerognc.simulation.six_dof_simulator import simulate_six_dof

    configuration = load_six_dof_configuration(config_path)
    output = output_override or configuration.output_directory
    result = simulate_six_dof(configuration)
    csv_path = write_result_csv(result, output / "six_dof_trajectory.csv")
    summary_path = write_summary_json(result, output / "six_dof_summary.json")
    if no_plots:
        figure_path = None
    else:
        from aerognc.visualisation.six_dof import plot_six_dof_results

        figure_path = plot_six_dof_results(result, output)
    LOGGER.info(
        "completed 6-DOF samples=%d execution_time_s=%.3f max_attitude_error_deg=%.3f",
        result.time_s.size,
        result.execution_time_s,
        result.maximum_summary["attitude_error"]["value"],
    )
    LOGGER.info("trajectory=%s summary=%s", csv_path, summary_path)
    if figure_path is not None:
        LOGGER.info("figure=%s", figure_path)
    print(json.dumps(result.maximum_summary, indent=2, sort_keys=True))
    return 0


def _run_interplanetary(
    config_path: Path,
    output_override: Path | None,
    playback_days_per_second: float,
    frames_per_second: int,
    repeat: bool,
    camera_mode: str,
    save_gif: Path | None,
    save_snapshot: Path | None,
    no_window: bool,
) -> int:
    from aerognc.configuration import load_interplanetary_configuration
    from aerognc.simulation.interplanetary import simulate_interplanetary
    from aerognc.visualisation.mission_control import MISSION_CAMERAS

    if camera_mode not in MISSION_CAMERAS:
        raise ValueError(f"mission camera must be one of {MISSION_CAMERAS}")
    configuration = load_interplanetary_configuration(config_path)
    output = output_override or configuration.output_directory
    mission = simulate_interplanetary(configuration)
    csv_path = write_result_csv(mission.result, output / "interplanetary_trajectory.csv")
    summary_path = write_summary_json(mission.result, output / "interplanetary_summary.json")
    LOGGER.info(
        "interplanetary scenario=%s samples=%d duration_days=%.1f execution_time_s=%.3f",
        mission.result.scenario_name,
        mission.result.time_s.size,
        mission.result.time_s[-1] / 86_400.0,
        mission.result.execution_time_s,
    )
    for event in mission.result.event_summary:
        LOGGER.info(
            "event=%s time_days=%.3f reference=%s distance_m=%.3e",
            event["name"],
            event["time_days"],
            event["reference_body"],
            event["distance_m"],
        )
    LOGGER.info("trajectory=%s summary=%s", csv_path, summary_path)
    if not no_window or save_gif is not None or save_snapshot is not None:
        from aerognc.visualisation.mission_control import play_interplanetary_mission

        snapshot_path, gif_path = play_interplanetary_mission(
            mission,
            playback_days_per_second=playback_days_per_second,
            frames_per_second=frames_per_second,
            repeat=repeat,
            camera_mode=camera_mode,
            save_gif=save_gif,
            save_snapshot=save_snapshot,
            show_window=not no_window,
        )
        if snapshot_path is not None:
            LOGGER.info("snapshot=%s", snapshot_path)
        if gif_path is not None:
            LOGGER.info("animation=%s", gif_path)
    print(json.dumps(mission.result.maximum_summary, indent=2, sort_keys=True))
    return 0


def _run_orbit_tour(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_orbit_tour_configuration
    from aerognc.simulation.orbit_assisted_tour import (
        orbit_tour_payload,
        simulate_orbit_assisted_tour,
        write_orbit_tour_results,
    )

    configuration = load_orbit_tour_configuration(config_path)
    output = output_override or configuration.output_directory
    simulation = simulate_orbit_assisted_tour(configuration)
    csv_path, report_path = write_orbit_tour_results(simulation, output)
    from aerognc.verification.astrodynamics_interop import (
        write_astrodynamics_interoperability,
    )

    oem_path, gmat_script_path, external_status_path = write_astrodynamics_interoperability(
        simulation, output
    )
    if not no_plots:
        from aerognc.visualisation import plot_orbit_assisted_tour

        figure_path = plot_orbit_assisted_tour(simulation, output)
        LOGGER.info("figure=%s", figure_path)
    LOGGER.info(
        "orbit-tour route=%s-%s-%s delta_v_kmps=%.3f final_mass_kg=%.1f pass=%s",
        configuration.departure_body,
        configuration.assist_body,
        configuration.destination_body,
        simulation.tour.total_delta_v_mps / 1_000.0,
        simulation.tour.final_mass_kg,
        simulation.assessment.all_pass,
    )
    LOGGER.info(
        "trajectory=%s report=%s oem=%s gmat_interface=%s external_status=%s",
        csv_path,
        report_path,
        oem_path,
        gmat_script_path,
        external_status_path,
    )
    print(json.dumps(orbit_tour_payload(simulation)["requirements"], indent=2, sort_keys=True))
    return 0 if simulation.assessment.all_pass else 1


def _run_orbit_sandbox(
    config_path: Path,
    output_override: Path | None,
    no_plots: bool,
    play: bool,
) -> int:
    from aerognc.configuration import load_orbit_sandbox_configuration
    from aerognc.simulation.orbit_sandbox import (
        orbit_sandbox_payload,
        simulate_orbit_sandbox,
        write_orbit_sandbox_results,
    )

    configuration = load_orbit_sandbox_configuration(config_path)
    output = output_override or configuration.output_directory
    simulation = simulate_orbit_sandbox(configuration)
    csv_path, summary_path, report_path = write_orbit_sandbox_results(simulation, output)
    if not no_plots:
        from aerognc.visualisation.orbit_sandbox import plot_orbit_sandbox

        figure_paths = plot_orbit_sandbox(simulation, output)
        LOGGER.info("orbit figures=%s", figure_paths)
    LOGGER.info(
        "orbit-sandbox model=%s samples=%d duration_days=%.4f reentered=%s escaped=%s",
        configuration.model,
        simulation.result.time_s.size,
        simulation.result.time_s[-1] / 86_400.0,
        simulation.reentered,
        simulation.escaped,
    )
    LOGGER.info("trajectory=%s summary=%s report=%s", csv_path, summary_path, report_path)
    if play:
        from aerognc.visualisation.orbit_sandbox import play_orbit_sandbox

        play_orbit_sandbox(simulation)
    print(json.dumps(orbit_sandbox_payload(simulation), indent=2, sort_keys=True))
    return 0


def _run_aircraft(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_aircraft_configuration
    from aerognc.simulation.aircraft_sandbox import (
        aircraft_sandbox_payload,
        simulate_aircraft,
        write_aircraft_results,
    )

    configuration = load_aircraft_configuration(config_path)
    output = output_override or configuration.output_directory
    simulation = simulate_aircraft(configuration)
    csv_path, summary_path, report_path = write_aircraft_results(simulation, output)
    if not no_plots:
        from aerognc.visualisation.aircraft_sandbox import plot_aircraft_sandbox

        figure_paths = plot_aircraft_sandbox(simulation, output)
        LOGGER.info("aircraft figures=%s", figure_paths)
    LOGGER.info(
        "aircraft samples=%d duration_s=%.2f reached_space=%s impacted_ground=%s",
        simulation.result.time_s.size,
        simulation.result.time_s[-1],
        simulation.reached_space,
        simulation.impacted_ground,
    )
    LOGGER.info("trajectory=%s summary=%s report=%s", csv_path, summary_path, report_path)
    print(json.dumps(aircraft_sandbox_payload(simulation), indent=2, sort_keys=True))
    return 0


def _run_live_aircraft(
    config_path: Path,
    mesh_path: Path,
    mesh_axes: str,
    frames_per_second: int,
    real_time_factor: float,
    camera_mode: str,
    no_gamepad: bool,
    control_mode: str | None,
    pilot_profile_path: Path,
    trail_mode: str,
    trail_duration_s: float,
    trail_color: str,
    mesh_scale_mode: str,
    recording_directory: Path,
    scripted_demo: bool,
    training_task: str | None,
) -> int:
    from aerognc.configuration import load_aircraft_configuration
    from aerognc.simulation.aircraft_training import TrainingTask
    from aerognc.visualisation.aircraft_controls import (
        AIRCRAFT_CONTROL_MODES,
        load_pilot_profile,
    )
    from aerognc.visualisation.aircraft_experience import (
        TRAIL_COLOR_SOURCES,
        TRAIL_MODES,
        TrailSettings,
    )
    from aerognc.visualisation.aircraft_live import (
        LIVE_CAMERA_MODES,
        MeshScaleMode,
        play_aircraft_live,
    )
    from aerognc.visualisation.mesh import MESH_AXIS_CONVENTIONS

    if mesh_axes not in MESH_AXIS_CONVENTIONS:
        raise ValueError(f"mesh axes must be one of {MESH_AXIS_CONVENTIONS}")
    if camera_mode not in LIVE_CAMERA_MODES:
        raise ValueError(f"live camera must be one of {LIVE_CAMERA_MODES}")
    if control_mode is not None and control_mode not in AIRCRAFT_CONTROL_MODES:
        raise ValueError(f"aircraft control mode must be one of {AIRCRAFT_CONTROL_MODES}")
    if trail_mode not in TRAIL_MODES or trail_color not in TRAIL_COLOR_SOURCES:
        raise ValueError("aircraft trail mode or colour source is unsupported")
    if mesh_scale_mode not in {"enlarged_marker", "true_scale"}:
        raise ValueError("aircraft mesh scale mode is unsupported")
    configuration = load_aircraft_configuration(config_path)
    profile = load_pilot_profile(pilot_profile_path)
    if control_mode is not None:
        from dataclasses import replace

        profile = replace(profile, control_mode=control_mode)
    LOGGER.info(
        "live-aircraft vehicle=%s mesh=%s fps=%d real_time_factor=%.2f",
        configuration.name,
        mesh_path,
        frames_per_second,
        real_time_factor,
    )
    play_aircraft_live(
        configuration,
        mesh_path,
        axis_convention=mesh_axes,
        frames_per_second=frames_per_second,
        real_time_factor=real_time_factor,
        camera_mode=camera_mode,
        enable_gamepad=not no_gamepad,
        control_profile=profile,
        trail_settings=TrailSettings(
            mode=trail_mode,
            fading_duration_s=trail_duration_s,
            color_source=trail_color,
        ),
        mesh_scale_mode=cast("MeshScaleMode", mesh_scale_mode),
        recorder_directory=recording_directory,
        scripted_demo=scripted_demo,
        training_task=cast(TrainingTask, training_task),
    )
    return 0


def _run_aircraft_replay(
    config_path: Path,
    recording_path: Path,
    mesh_path: Path,
    playback_factor: float,
) -> int:
    from aerognc.configuration import load_aircraft_configuration
    from aerognc.visualisation.aircraft_replay import play_aircraft_recording

    configuration = load_aircraft_configuration(config_path)
    play_aircraft_recording(
        configuration,
        recording_path,
        mesh_path,
        playback_factor=playback_factor,
    )
    return 0


def _run_aircraft_aero_compare(config_path: Path, table_path: Path, output: Path) -> int:
    from aerognc.configuration import load_aircraft_configuration
    from aerognc.verification.aircraft_aerodynamics import (
        compare_aircraft_aerodynamic_backends,
        write_aircraft_aerodynamic_comparison,
    )

    comparison = compare_aircraft_aerodynamic_backends(
        load_aircraft_configuration(config_path), table_path
    )
    report_path = write_aircraft_aerodynamic_comparison(comparison, output)
    LOGGER.info("aircraft aerodynamic backend comparison=%s", report_path)
    print(json.dumps(comparison.summary(), indent=2, sort_keys=True))
    return 0


def _run_launch_window(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_launch_window_configuration
    from aerognc.verification.launch_window import (
        launch_window_payload,
        run_launch_window_optimization,
        write_launch_window_report,
    )

    configuration = load_launch_window_configuration(config_path)
    output = output_override or configuration.output_directory
    run = run_launch_window_optimization(configuration)
    report_path = write_launch_window_report(run, output)
    if not no_plots:
        from aerognc.visualisation import plot_launch_window_optimization

        figure_path = plot_launch_window_optimization(run, output)
        LOGGER.info("figure=%s", figure_path)
    optimum = run.optimization.optimum
    LOGGER.info(
        "launch-window departure_day=%.3f arrival_day=%.3f delta_v_kmps=%.3f pass=%s",
        optimum.departure_time_s / 86_400.0,
        optimum.arrival_time_s / 86_400.0,
        optimum.total_delta_v_mps / 1_000.0,
        run.assessment.all_pass,
    )
    LOGGER.info("report=%s", report_path)
    print(json.dumps(launch_window_payload(run)["requirements"], indent=2, sort_keys=True))
    return 0 if run.assessment.all_pass else 1


def _run_catalog(
    csv_path: Path,
    metadata_path: Path,
    galaxy_metadata_path: Path,
    solar_system_path: Path,
    query: str,
    method: str | None,
    maximum_distance_pc: float | None,
    minimum_year: int | None,
    maximum_year: int | None,
    limit: int | None,
    output: Path,
    no_plots: bool,
) -> int:
    from aerognc.catalogs import (
        load_exoplanet_catalog,
        load_milky_way_metadata,
        load_solar_system_planets,
    )
    from aerognc.verification.galaxy_catalog import write_galaxy_catalog_outputs

    catalog = load_exoplanet_catalog(csv_path, metadata_path)
    galaxy = load_milky_way_metadata(galaxy_metadata_path)
    solar_system = load_solar_system_planets(solar_system_path)
    selection = catalog.search(
        text=query,
        maximum_distance_pc=maximum_distance_pc,
        discovery_method=method,
        minimum_discovery_year=minimum_year,
        maximum_discovery_year=maximum_year,
        limit=limit,
    )
    report_path, selection_path = write_galaxy_catalog_outputs(
        catalog,
        selection,
        galaxy,
        solar_system,
        output,
    )
    if not no_plots:
        from aerognc.visualisation import plot_milky_way_catalog

        figure_path = plot_milky_way_catalog(catalog, selection, output)
        LOGGER.info("figure=%s", figure_path)
    summary = catalog.summary(selection)
    LOGGER.info(
        "catalog snapshot=%d selected=%d hosts=%d positioned=%d retrieved=%s",
        catalog.provenance.row_count,
        summary.planet_count,
        summary.host_count,
        summary.positioned_planet_count,
        catalog.provenance.retrieved_utc,
    )
    LOGGER.info("report=%s selection=%s", report_path, selection_path)
    print(
        json.dumps(
            {
                "scope_note": catalog.provenance.scope_note,
                "selection": {
                    "planet_count": summary.planet_count,
                    "host_count": summary.host_count,
                    "positioned_planet_count": summary.positioned_planet_count,
                    "planets_with_orbital_period": summary.planets_with_orbital_period,
                    "planets_with_mass": summary.planets_with_mass,
                    "planets_with_radius": summary.planets_with_radius,
                    "discovery_year_min": summary.discovery_year_min,
                    "discovery_year_max": summary.discovery_year_max,
                    "nearest_reported_distance_pc": summary.nearest_reported_distance_pc,
                    "farthest_reported_distance_pc": summary.farthest_reported_distance_pc,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_software_loopback(
    *,
    samples: int,
    sample_period_ms: float,
    latency_ms: float,
    jitter_ms: float,
    loss_percent: float,
    duplicate_percent: float,
    deadline_ms: float,
    timeout_ms: float,
    seed: int,
    output: Path,
) -> int:
    from aerognc.simulation.hil import LinkImpairmentConfiguration
    from aerognc.simulation.software_loopback import (
        SoftwareLoopbackConfiguration,
        run_software_loopback_demo,
        write_software_loopback_report,
    )

    if not 0.0 <= loss_percent <= 100.0:
        raise ValueError("loss percent must be in [0, 100]")
    if not 0.0 <= duplicate_percent <= 100.0:
        raise ValueError("duplicate percent must be in [0, 100]")
    common = {
        "latency_s": latency_ms * 1.0e-3,
        "jitter_standard_deviation_s": jitter_ms * 1.0e-3,
        "loss_probability": loss_percent * 0.01,
        "duplicate_probability": duplicate_percent * 0.01,
    }
    configuration = SoftwareLoopbackConfiguration(
        sample_period_s=sample_period_ms * 1.0e-3,
        command_deadline_s=deadline_ms * 1.0e-3,
        command_timeout_s=timeout_ms * 1.0e-3,
        state_link=LinkImpairmentConfiguration(**common, random_seed=seed),
        command_link=LinkImpairmentConfiguration(**common, random_seed=seed + 1),
    )
    result = run_software_loopback_demo(configuration, sample_count=samples)
    report_path = write_software_loopback_report(result, output)
    LOGGER.info(
        "software-loopback samples=%d state_rx=%d command_rx=%d deadline_misses=%d "
        "watchdog=%d report=%s",
        result.sample_count,
        result.state_packets_accepted,
        result.command_packets_accepted,
        result.command_deadline_misses,
        result.watchdog_activations,
        report_path,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def _run_udp_loopback(
    *,
    samples: int,
    sample_period_ms: float,
    receive_timeout_ms: float,
    watchdog_timeout_ms: float,
    output: Path,
) -> int:
    from aerognc.simulation.udp_transport import (
        run_udp_loopback_demo,
        write_udp_loopback_report,
    )

    result = run_udp_loopback_demo(
        sample_count=samples,
        sample_period_s=sample_period_ms * 1.0e-3,
        receive_timeout_s=receive_timeout_ms * 1.0e-3,
        watchdog_timeout_s=watchdog_timeout_ms * 1.0e-3,
    )
    report_path = write_udp_loopback_report(result, output)
    LOGGER.info(
        "udp-loopback samples=%d state_rx=%d command_rx=%d watchdog=%d report=%s",
        result.sample_count,
        result.state_endpoint.packets_accepted,
        result.command_endpoint.packets_accepted,
        result.watchdog_activations,
        report_path,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def _run_attitude(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_attitude_control_configuration
    from aerognc.simulation.attitude_control import (
        compare_attitude_controllers,
        write_attitude_control_outputs,
    )

    configuration = load_attitude_control_configuration(config_path)
    output = output_override or configuration.output_directory
    results = compare_attitude_controllers(configuration)
    csv_path, metrics_path = write_attitude_control_outputs(results, output)
    if not no_plots:
        from aerognc.visualisation.control import plot_attitude_control_comparison

        figure_path = plot_attitude_control_comparison(results, configuration, output)
        LOGGER.info("figure=%s", figure_path)
    LOGGER.info("signals=%s metrics=%s", csv_path, metrics_path)
    payload = {result.controller_name: result.metrics.as_dict() for result in results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_navigation(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_navigation_demo_configuration
    from aerognc.simulation.navigation_demo import run_navigation_demo, write_navigation_demo

    configuration = load_navigation_demo_configuration(config_path)
    output = output_override or configuration.output_directory
    result = run_navigation_demo(configuration)
    csv_path, summary_path = write_navigation_demo(result, output)
    if not no_plots:
        from aerognc.visualisation.navigation import plot_navigation_demo

        figure_path = plot_navigation_demo(result, output)
        LOGGER.info("figure=%s", figure_path)
    payload = {
        "raw_barometer_rms_m": result.raw_barometer_rms_m,
        "estimated_altitude_rms_m": result.estimated_altitude_rms_m,
        "improvement_percent": 100.0
        * (1.0 - result.estimated_altitude_rms_m / result.raw_barometer_rms_m),
    }
    LOGGER.info("signals=%s summary=%s", csv_path, summary_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_advanced_navigation(
    config_path: Path,
    output_override: Path | None,
    consistency_runs: int | None,
    no_plots: bool,
) -> int:
    from aerognc.configuration import load_advanced_navigation_configuration
    from aerognc.simulation.advanced_navigation import (
        run_navigation_consistency,
        simulate_advanced_navigation,
    )
    from aerognc.verification.advanced_navigation import (
        advanced_navigation_payload,
        assess_advanced_navigation,
        write_advanced_navigation_results,
    )

    configuration = load_advanced_navigation_configuration(config_path)
    output = output_override or configuration.output_directory
    LOGGER.info(
        "advanced navigation scenario=%s duration_s=%.1f imu_rate_hz=%.1f",
        configuration.name,
        configuration.duration_s,
        configuration.imu_sample_rate_hz,
    )
    result = simulate_advanced_navigation(configuration)
    consistency = run_navigation_consistency(configuration, run_count=consistency_runs)
    assessment = assess_advanced_navigation(result, consistency)
    trajectory_path, aiding_path, report_path = write_advanced_navigation_results(
        result,
        consistency,
        output,
    )
    if not no_plots:
        from aerognc.visualisation import plot_advanced_navigation

        figure_path = plot_advanced_navigation(result, consistency, output)
        LOGGER.info("figure=%s", figure_path)
    LOGGER.info(
        "position_rms_m=%.3f velocity_rms_mps=%.3f attitude_rms_deg=%.3f "
        "observability_rank=%d replay_steps=%d requirements_pass=%s",
        result.position_rms_m,
        result.velocity_rms_mps,
        result.attitude_rms_deg,
        result.observability_rank,
        result.maximum_replayed_step_count,
        assessment.all_pass,
    )
    LOGGER.info(
        "trajectory=%s aiding_audit=%s report=%s",
        trajectory_path,
        aiding_path,
        report_path,
    )
    payload = advanced_navigation_payload(result, consistency)
    print(json.dumps(payload["requirements"], indent=2, sort_keys=True))
    return 0 if assessment.all_pass else 1


def _run_monte_carlo(
    config_path: Path,
    sample_count: int | None,
    workers: int | None,
    output_override: Path | None,
    no_plots: bool,
) -> int:
    from aerognc.configuration import load_monte_carlo_configuration
    from aerognc.simulation.monte_carlo import run_monte_carlo, write_monte_carlo_outputs

    configuration = load_monte_carlo_configuration(config_path)
    output = output_override or configuration.output_directory
    summary = run_monte_carlo(configuration, sample_count=sample_count, workers=workers)
    csv_path, summary_path = write_monte_carlo_outputs(summary, output)
    if not no_plots:
        from aerognc.visualisation.monte_carlo import (
            plot_monte_carlo_sensitivity,
            plot_monte_carlo_summary,
        )

        figure_path = plot_monte_carlo_summary(summary, configuration.requirements, output)
        LOGGER.info("figure=%s", figure_path)
        sensitivity_path = plot_monte_carlo_sensitivity(summary, output)
        LOGGER.info("figure=%s", sensitivity_path)
    payload = {
        "sample_count": len(summary.runs),
        "successful_count": summary.successful_count,
        "failed_count": summary.failed_count,
        "requirement_pass_rates": summary.requirement_pass_rates,
        "worst_case_runs": summary.worst_case_runs,
    }
    LOGGER.info("runs=%s summary=%s", csv_path, summary_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_flight_test(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_navigation_demo_configuration
    from aerognc.verification.flight_test import run_synthetic_flight_test_workflow

    configuration = load_navigation_demo_configuration(config_path)
    output = output_override or Path("results/flight_test")
    workflow = run_synthetic_flight_test_workflow(configuration, output)
    if not no_plots:
        from aerognc.visualisation.flight_test import plot_flight_test_summary

        figure_path = plot_flight_test_summary(workflow, output)
        LOGGER.info("figure=%s", figure_path)
    payload = {
        "event_time_errors_s": workflow.event_time_errors_s,
        "apogee_error_m": workflow.apogee_error_m,
    }
    LOGGER.info("measurements=%s summary=%s", workflow.measurement_csv, workflow.summary_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_flight_data_identification(
    config_path: Path,
    output_override: Path | None,
    no_plots: bool,
) -> int:
    from aerognc.configuration import load_flight_data_identification_configuration
    from aerognc.verification.flight_data_identification import (
        assess_flight_data_identification,
        flight_data_identification_payload,
        run_flight_data_identification_workflow,
    )

    configuration = load_flight_data_identification_configuration(config_path)
    output = output_override or configuration.output_directory
    workflow = run_flight_data_identification_workflow(configuration, output)
    assessment = assess_flight_data_identification(workflow.result)
    if not no_plots:
        from aerognc.visualisation import plot_flight_data_identification

        figure_path = plot_flight_data_identification(workflow.result, output)
        LOGGER.info("figure=%s", figure_path)
    LOGGER.info(
        "clock_offset_s=%.6f clock_drift_ppm=%.2f R2=%.5f "
        "validation_pitch_rms_deg=%.3f requirements_pass=%s",
        workflow.result.clock_alignment.offset_s,
        workflow.result.clock_alignment.drift_ppm,
        workflow.result.identification_r_squared,
        workflow.result.validation_pitch_rms_deg,
        assessment.all_pass,
    )
    LOGGER.info(
        "command_log=%s sensor_log=%s aligned=%s report=%s",
        workflow.logs.command_log,
        workflow.logs.sensor_log,
        workflow.aligned_csv,
        workflow.report_json,
    )
    payload = flight_data_identification_payload(workflow.result)
    print(json.dumps(payload["requirements"], indent=2, sort_keys=True))
    return 0 if assessment.all_pass else 1


def _run_flight_analysis(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration.analysis_loader import (
        load_flight_control_analysis_configuration,
    )
    from aerognc.verification.flight_control_analysis import (
        run_flight_control_analysis,
        write_flight_control_analysis,
    )

    configuration = load_flight_control_analysis_configuration(config_path)
    output = output_override or configuration.output_directory
    result = run_flight_control_analysis(configuration)
    report_path = write_flight_control_analysis(result, output)
    if not no_plots:
        from aerognc.visualisation.flight_control_analysis import (
            plot_flight_control_analysis,
        )

        figure_path = plot_flight_control_analysis(result, output)
        LOGGER.info("figure=%s", figure_path)
    LOGGER.info("analysis=%s", report_path)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def _run_flight_envelope(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_flight_envelope_configuration
    from aerognc.gnc.flight_envelope import analyze_flight_envelope
    from aerognc.verification.flight_envelope import (
        assess_flight_envelope,
        flight_envelope_payload,
        write_flight_envelope_results,
    )

    configuration = load_flight_envelope_configuration(config_path)
    output = output_override or configuration.output_directory
    result = analyze_flight_envelope(configuration)
    report_path, csv_path = write_flight_envelope_results(result, output)
    if not no_plots:
        from aerognc.visualisation import plot_flight_envelope

        figure_path = plot_flight_envelope(result, output)
        LOGGER.info("figure=%s", figure_path)
    assessment = assess_flight_envelope(result)
    LOGGER.info(
        "flight envelope points=%d robust_samples=%d requirements_pass=%s",
        len(result.analyses),
        result.robustness_verification.sample_count,
        assessment.all_pass,
    )
    LOGGER.info("report=%s points=%s", report_path, csv_path)
    payload = flight_envelope_payload(result)
    print(
        json.dumps(
            {"summary": payload["summary"], "requirements": payload["requirements"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if assessment.all_pass else 1


def _run_constrained_ascent(config_path: Path, output_override: Path | None, no_plots: bool) -> int:
    from aerognc.configuration import load_ascent_guidance_configuration
    from aerognc.simulation.guided_ascent import optimize_ascent_guidance
    from aerognc.verification.ascent_guidance import (
        ascent_guidance_payload,
        assess_ascent_guidance,
        write_ascent_guidance_results,
    )

    configuration = load_ascent_guidance_configuration(config_path)
    output = output_override or configuration.output_directory
    optimization = optimize_ascent_guidance(configuration)
    reference_path, optimized_path, history_path, report_path = write_ascent_guidance_results(
        optimization, output
    )
    if not no_plots:
        from aerognc.visualisation import plot_ascent_guidance

        figure_path = plot_ascent_guidance(optimization, output)
        LOGGER.info("figure=%s", figure_path)
    assessment = assess_ascent_guidance(optimization)
    LOGGER.info(
        "constrained ascent evaluations=%d apogee_m=%.2f requirements_pass=%s",
        len(optimization.evaluations),
        optimization.optimized_run.apogee_m,
        assessment.all_pass,
    )
    LOGGER.info(
        "reference=%s optimized=%s history=%s report=%s",
        reference_path,
        optimized_path,
        history_path,
        report_path,
    )
    payload = ascent_guidance_payload(optimization)
    print(
        json.dumps(
            {
                "requirements": payload["requirements"],
                "optimized_run": payload["optimized_run"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if assessment.all_pass else 1


def _run_benchmark_command(
    config_path: Path,
    repetitions: int,
    output_path: Path,
    maximum_wall_time_s: float | None,
    maximum_cpu_time_s: float | None,
    maximum_peak_memory_mb: float | None,
    minimum_samples_per_second: float | None,
    minimum_steps_per_second: float | None,
) -> int:
    from aerognc.verification.benchmark import (
        BenchmarkBudget,
        benchmark_payload,
        run_benchmark,
        write_benchmark_report,
    )

    configuration = load_three_dof_configuration(config_path)
    preflight = simulate_three_dof(configuration)
    sample_count = int(preflight.time_s.size)
    result = run_benchmark(
        f"three-dof:{configuration.simulation.name}",
        lambda: simulate_three_dof(configuration),
        sample_count=sample_count,
        step_count=sample_count - 1,
        repetitions=repetitions,
        warmup=False,
        budget=BenchmarkBudget(
            maximum_wall_time_s,
            maximum_cpu_time_s,
            maximum_peak_memory_mb,
            minimum_samples_per_second,
            minimum_steps_per_second,
        ),
    )
    report_path = write_benchmark_report(result, output_path)
    LOGGER.info(
        "benchmark wall_s=%.6f cpu_s=%.6f samples_per_s=%.1f pass=%s report=%s",
        result.wall_time_s,
        result.cpu_time_s,
        result.samples_per_second,
        result.passed,
        report_path,
    )
    print(json.dumps(benchmark_payload(result), indent=2, sort_keys=True))
    return 0 if result.passed else 1


def _run_diagnostic_command(
    project_root: Path,
    result_directory: Path,
    output: Path | None,
) -> int:
    from aerognc.diagnostics import (
        format_diagnostic_report,
        run_diagnostics,
        write_diagnostic_report,
    )

    report = run_diagnostics(
        project_root=project_root,
        result_directory=result_directory,
    )
    output_path = output or result_directory / "diagnostics" / "health.json"
    report_path = write_diagnostic_report(report, output_path)
    print(format_diagnostic_report(report))
    LOGGER.info("diagnostic report=%s", report_path)
    return 0 if report.passed else 2


def _run_mission_command(arguments: argparse.Namespace) -> int:
    from aerognc.mission import load_mission

    if arguments.mission_command == "validate":
        mission = load_mission(arguments.mission).validate()
        LOGGER.info(
            "mission %r valid: %d waypoints, home (%.5f, %.5f)",
            mission.name,
            len(mission.waypoints),
            mission.home.latitude_deg,
            mission.home.longitude_deg,
        )
        return 0
    return 2


def _run_waypoint_command(arguments: argparse.Namespace) -> int:
    from aerognc.gnc.waypoint_guidance import GuidanceMode
    from aerognc.mission import load_mission
    from aerognc.simulation.waypoint_mission import (
        WaypointMissionConfig,
        run_waypoint_mission,
    )

    mission = load_mission(arguments.mission)
    config = WaypointMissionConfig(
        dt_s=arguments.dt_s,
        max_time_s=arguments.max_time_s,
        guidance_mode=GuidanceMode(arguments.guidance),
        wind_ned_mps=(arguments.wind_north_mps, arguments.wind_east_mps, 0.0),
    )
    result = run_waypoint_mission(mission, config)
    summary = result.summary()
    LOGGER.info("waypoint mission outcome: %s", summary)

    output_directory = arguments.output or Path("results") / "waypoint_gnc"
    output_directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_directory / "mission_log.csv")
    result.to_json(output_directory / "mission_log.json")
    if not arguments.no_plots:
        from aerognc.visualisation.waypoint_mission import plot_waypoint_mission

        plot_waypoint_mission(result, output_directory / "mission_dashboard.png")
    LOGGER.info("wrote mission log and artifacts to %s", output_directory)
    return 0 if result.completed else 1


def _run_rpo_command(arguments: argparse.Namespace) -> int:
    import json

    import numpy as np

    from aerognc.astrodynamics.relative_motion import ClohessyWiltshireModel, simulate_rendezvous

    radius_m = 6_378_137.0 + arguments.altitude_km * 1000.0
    model = ClohessyWiltshireModel.from_orbit(radius_m)
    start = float(arguments.start_behind_m)
    initial_state = np.array([0.3 * start, -start, 0.0, 0.0, 0.0, 0.0])
    # A safe stepped V-bar approach toward the target (station-keep at each hold).
    hold_points = [
        np.array([0.0, -0.5 * start, 0.0]),
        np.array([0.0, -0.15 * start, 0.0]),
        np.array([0.0, -30.0, 0.0]),
    ]
    trajectory = simulate_rendezvous(
        model, initial_state, hold_points, leg_time_s=arguments.leg_time_s
    )
    summary = {
        "target_altitude_km": arguments.altitude_km,
        "total_delta_v_mps": round(trajectory.total_delta_v_mps, 4),
        "closest_approach_m": round(trajectory.closest_approach_m, 3),
        "closest_approach_time_s": round(trajectory.closest_approach_time_s, 1),
        "final_hold_point_m": [0.0, -30.0, 0.0],
    }
    LOGGER.info("rendezvous (approach / station-keep, non-weapon): %s", summary)

    output_directory = arguments.output or Path("results") / "rpo"
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "rendezvous.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if not arguments.no_plots:
        from aerognc.visualisation.rpo import plot_rendezvous

        plot_rendezvous(trajectory, output_directory / "rendezvous.png")
    LOGGER.info("wrote rendezvous artifacts to %s", output_directory)
    return 0


def _run_mission_planner_command(arguments: argparse.Namespace) -> int:  # pragma: no cover - UI
    from aerognc.visualisation.mission_planner_map import launch_mission_planner

    mission_path = str(arguments.mission) if arguments.mission else None
    launch_mission_planner(mission_path)
    return 0


def _run_project_command(arguments: argparse.Namespace) -> int:
    from aerognc.project import create_empty_project, load_project
    from aerognc.project.comparison import compare_datasets, write_comparison_json
    from aerognc.project.report import write_engineering_report
    from aerognc.project.result_store import ResultStore
    from aerognc.project.runner import ProjectRunService, scenario_seed

    if arguments.project_command == "init":
        project = create_empty_project(arguments.directory, arguments.name)
        print(
            json.dumps(
                {
                    "project": str(project.source_path),
                    "workspace_root": str(project.workspace_root),
                    "result_root": str(project.result_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    project = load_project(arguments.project)
    service = ProjectRunService()
    if arguments.project_command in {"validate", "inspect"}:
        issues = service.validate_workflows(project)
        payload = {
            "schema_version": project.schema_version,
            "name": project.name,
            "description": project.description,
            "safety_scope": project.safety_scope,
            "project_file": str(project.source_path),
            "workspace_root": str(project.workspace_root),
            "result_root": str(project.result_root),
            "available_workflows": list(service.registry.names()),
            "plugin_issues": [
                {"entry_point": item.entry_point, "reason": item.reason}
                for item in service.registry.plugin_issues
            ],
            "issues": list(issues),
            "scenarios": [
                {
                    "name": scenario.name,
                    "workflow": scenario.workflow,
                    "configuration": scenario.configuration.as_posix(),
                    "enabled": scenario.enabled,
                    "seed": scenario_seed(project, scenario),
                    "tags": list(scenario.tags),
                }
                for scenario in project.scenarios
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if not issues else 1

    store = ResultStore(project.result_root)
    if arguments.project_command == "run":
        stored = service.run(project, arguments.scenario)
        print(
            json.dumps(
                {
                    "run_id": stored.manifest.run_id,
                    "status": stored.manifest.status,
                    "input_fingerprint": stored.manifest.input_fingerprint,
                    "run_directory": str(stored.directory),
                    "report": str(stored.directory / "report.html"),
                    "requirements_pass": all(item.passed for item in stored.manifest.requirements),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if all(item.passed for item in stored.manifest.requirements) else 1
    if arguments.project_command == "list":
        records = store.list_runs(
            project_name=project.name,
            scenario_name=arguments.scenario,
            status=arguments.status,
        )
        print(
            json.dumps(
                [
                    {
                        "run_id": item.run_id,
                        "created_utc": item.created_utc,
                        "scenario": item.scenario_name,
                        "workflow": item.workflow,
                        "status": item.status,
                        "input_fingerprint": item.input_fingerprint,
                    }
                    for item in records
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.project_command == "compare":
        baseline = store.load(arguments.baseline_run)
        candidate = store.load(arguments.candidate_run)
        if baseline.dataset is None or candidate.dataset is None:
            raise ValueError("only completed runs with trajectories can be compared")
        channels = None
        if arguments.channels is not None:
            channels = [item.strip() for item in arguments.channels.split(",") if item.strip()]
            if not channels:
                raise ValueError("--channels must contain at least one channel name")
        comparison = compare_datasets(
            baseline.dataset,
            candidate.dataset,
            channels=channels,
            sample_count=arguments.samples,
        )
        output = arguments.output or (
            project.result_root
            / "comparisons"
            / f"{arguments.baseline_run}__{arguments.candidate_run}.json"
        )
        json_path = write_comparison_json(comparison, output)
        html_path = write_engineering_report(
            candidate,
            output.with_suffix(".html"),
            comparison=comparison,
        )
        print(
            json.dumps(
                {
                    "comparison": str(json_path),
                    "report": str(html_path),
                    "channels": [item.channel for item in comparison.channels],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.project_command == "report":
        stored = store.load(arguments.run_id)
        path = write_engineering_report(stored, arguments.output)
        print(json.dumps({"report": str(path)}, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported project command: {arguments.project_command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute the selected workflow."""
    _configure_logging()
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            return _run_three_dof(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "rotating-ascent":
            return _run_rotating_ascent(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "rotating-six-dof":
            return _run_rotating_six_dof(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "multistage-recovery":
            return _run_multistage_recovery(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "aero-analysis":
            return _run_aero_analysis(
                arguments.config,
                arguments.output,
                arguments.mach,
                arguments.alpha_deg,
                arguments.beta_deg,
                arguments.no_plots,
            )
        if arguments.command == "play":
            return _run_playback(
                arguments.config,
                arguments.speed,
                arguments.fps,
                arguments.repeat,
                arguments.save_gif,
                arguments.no_window,
            )
        if arguments.command == "play-3d":
            return _run_playback_3d(
                arguments.config,
                arguments.speed,
                arguments.fps,
                arguments.repeat,
                arguments.camera,
                arguments.save_gif,
                arguments.no_window,
            )
        if arguments.command == "six-dof":
            return _run_six_dof(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "interplanetary":
            return _run_interplanetary(
                arguments.config,
                arguments.output,
                arguments.speed_days_per_second,
                arguments.fps,
                arguments.repeat,
                arguments.camera,
                arguments.save_gif,
                arguments.save_snapshot,
                arguments.no_window,
            )
        if arguments.command == "orbit-tour":
            return _run_orbit_tour(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "orbit-sandbox":
            return _run_orbit_sandbox(
                arguments.config,
                arguments.output,
                arguments.no_plots,
                arguments.play,
            )
        if arguments.command == "aircraft":
            return _run_aircraft(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "fly-aircraft":
            return _run_live_aircraft(
                arguments.config,
                arguments.mesh,
                arguments.mesh_axes,
                arguments.fps,
                arguments.real_time_factor,
                arguments.camera,
                arguments.no_gamepad,
                arguments.control_mode,
                arguments.pilot_profile,
                arguments.trail,
                arguments.trail_duration,
                arguments.trail_color,
                arguments.mesh_scale,
                arguments.recording_directory,
                arguments.demo,
                arguments.training_task,
            )
        if arguments.command == "replay-aircraft":
            return _run_aircraft_replay(
                arguments.config,
                arguments.recording,
                arguments.mesh,
                arguments.playback_factor,
            )
        if arguments.command == "aircraft-aero-compare":
            return _run_aircraft_aero_compare(
                arguments.config,
                arguments.table,
                arguments.output,
            )
        if arguments.command == "launch-window":
            return _run_launch_window(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "catalog":
            return _run_catalog(
                arguments.csv,
                arguments.metadata,
                arguments.galaxy_metadata,
                arguments.solar_system,
                arguments.query,
                arguments.method,
                arguments.max_distance_pc,
                arguments.min_year,
                arguments.max_year,
                arguments.limit,
                arguments.output,
                arguments.no_plots,
            )
        if arguments.command == "workbench":
            from aerognc.visualisation.workbench import launch_workbench

            launch_workbench(
                arguments.six_dof_config,
                arguments.orbit_tour_config,
                arguments.planetary_catalog,
                arguments.verified_interplanetary_config,
                arguments.exoplanet_csv,
                arguments.exoplanet_metadata,
                arguments.milky_way_metadata,
                arguments.solar_system_planets,
                arguments.project_file,
            )
            return 0
        if arguments.command == "software-loopback":
            return _run_software_loopback(
                samples=arguments.samples,
                sample_period_ms=arguments.sample_period_ms,
                latency_ms=arguments.latency_ms,
                jitter_ms=arguments.jitter_ms,
                loss_percent=arguments.loss_percent,
                duplicate_percent=arguments.duplicate_percent,
                deadline_ms=arguments.deadline_ms,
                timeout_ms=arguments.timeout_ms,
                seed=arguments.seed,
                output=arguments.output,
            )
        if arguments.command == "udp-loopback":
            return _run_udp_loopback(
                samples=arguments.samples,
                sample_period_ms=arguments.sample_period_ms,
                receive_timeout_ms=arguments.receive_timeout_ms,
                watchdog_timeout_ms=arguments.watchdog_timeout_ms,
                output=arguments.output,
            )
        if arguments.command == "fmi-interface":
            from aerognc.interoperability import write_fmi_controller_interface

            description_path, status_path = write_fmi_controller_interface(arguments.output)
            LOGGER.info("FMI interface=%s status=%s", description_path, status_path)
            return 0
        if arguments.command == "mission-designer":
            from aerognc.visualisation.mission_designer import launch_mission_designer

            launch_mission_designer(arguments.catalog, arguments.verified_config)
            return 0
        if arguments.command == "attitude":
            return _run_attitude(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "navigation":
            return _run_navigation(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "advanced-navigation":
            return _run_advanced_navigation(
                arguments.config,
                arguments.output,
                arguments.consistency_runs,
                arguments.no_plots,
            )
        if arguments.command == "monte-carlo":
            return _run_monte_carlo(
                arguments.config,
                arguments.samples,
                arguments.workers,
                arguments.output,
                arguments.no_plots,
            )
        if arguments.command == "flight-test":
            return _run_flight_test(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "flight-data-identification":
            return _run_flight_data_identification(
                arguments.config,
                arguments.output,
                arguments.no_plots,
            )
        if arguments.command == "flight-analysis":
            return _run_flight_analysis(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "flight-envelope":
            return _run_flight_envelope(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "constrained-ascent":
            return _run_constrained_ascent(arguments.config, arguments.output, arguments.no_plots)
        if arguments.command == "benchmark":
            return _run_benchmark_command(
                arguments.config,
                arguments.repetitions,
                arguments.output,
                arguments.max_wall_time_s,
                arguments.max_cpu_time_s,
                arguments.max_peak_memory_mb,
                arguments.min_samples_per_second,
                arguments.min_steps_per_second,
            )
        if arguments.command == "diagnose":
            return _run_diagnostic_command(
                arguments.project_root,
                arguments.result_directory,
                arguments.output,
            )
        if arguments.command == "project":
            return _run_project_command(arguments)
        if arguments.command == "mission":
            return _run_mission_command(arguments)
        if arguments.command == "waypoint":
            return _run_waypoint_command(arguments)
        if arguments.command == "mission-planner":
            return _run_mission_planner_command(arguments)
        if arguments.command == "rpo":
            return _run_rpo_command(arguments)
    except (ConfigurationError, ValueError, FloatingPointError, OSError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 2
    LOGGER.error("unsupported command: %s", arguments.command)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
