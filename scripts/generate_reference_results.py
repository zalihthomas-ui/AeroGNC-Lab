"""Regenerate compact README figures and summaries from nominal configurations."""

import json
from dataclasses import replace
from pathlib import Path

from aerognc.catalogs import (
    load_exoplanet_catalog,
    load_milky_way_metadata,
    load_solar_system_planets,
)
from aerognc.configuration import (
    load_advanced_navigation_configuration,
    load_aircraft_configuration,
    load_ascent_guidance_configuration,
    load_attitude_control_configuration,
    load_flight_control_analysis_configuration,
    load_flight_data_identification_configuration,
    load_flight_envelope_configuration,
    load_interplanetary_configuration,
    load_launch_window_configuration,
    load_monte_carlo_configuration,
    load_multistage_recovery_configuration,
    load_navigation_demo_configuration,
    load_orbit_sandbox_configuration,
    load_orbit_tour_configuration,
    load_rotating_ascent_configuration,
    load_rotating_six_dof_configuration,
    load_six_dof_configuration,
    load_three_dof_configuration,
)
from aerognc.gnc.flight_envelope import analyze_flight_envelope
from aerognc.simulation.advanced_navigation import (
    run_navigation_consistency,
    simulate_advanced_navigation,
)
from aerognc.simulation.aircraft_sandbox import aircraft_sandbox_payload, simulate_aircraft
from aerognc.simulation.attitude_control import compare_attitude_controllers
from aerognc.simulation.guided_ascent import optimize_ascent_guidance
from aerognc.simulation.hil import LinkImpairmentConfiguration
from aerognc.simulation.interplanetary import simulate_interplanetary
from aerognc.simulation.logging import write_summary_json
from aerognc.simulation.monte_carlo import run_monte_carlo
from aerognc.simulation.multistage_recovery import simulate_configured_multistage_recovery
from aerognc.simulation.navigation_demo import run_navigation_demo
from aerognc.simulation.orbit_assisted_tour import (
    orbit_tour_payload,
    simulate_orbit_assisted_tour,
)
from aerognc.simulation.orbit_sandbox import orbit_sandbox_payload, simulate_orbit_sandbox
from aerognc.simulation.rotating_ascent import simulate_rotating_ascent
from aerognc.simulation.rotating_six_dof import simulate_rotating_six_dof
from aerognc.simulation.simulator import simulate_three_dof
from aerognc.simulation.six_dof_simulator import simulate_six_dof
from aerognc.simulation.software_loopback import (
    SoftwareLoopbackConfiguration,
    run_software_loopback_demo,
    write_software_loopback_report,
)
from aerognc.simulation.udp_transport import run_udp_loopback_demo, write_udp_loopback_report
from aerognc.vehicle.aero_database import TabulatedAerodynamicDatabase
from aerognc.verification.advanced_navigation import advanced_navigation_payload
from aerognc.verification.aero_database import (
    analyze_aerodynamic_database,
    write_aerodynamic_database_analysis,
)
from aerognc.verification.ascent_guidance import ascent_guidance_payload
from aerognc.verification.flight_control_analysis import (
    run_flight_control_analysis,
    write_flight_control_analysis,
)
from aerognc.verification.flight_data_identification import (
    flight_data_identification_payload,
    run_flight_data_identification_workflow,
)
from aerognc.verification.flight_envelope import flight_envelope_payload
from aerognc.verification.flight_test import run_synthetic_flight_test_workflow
from aerognc.verification.galaxy_catalog import galaxy_catalog_payload
from aerognc.verification.launch_window import (
    launch_window_payload,
    run_launch_window_optimization,
)
from aerognc.visualisation import plot_three_dof_results
from aerognc.visualisation.advanced_navigation import plot_advanced_navigation
from aerognc.visualisation.aero_database import plot_aerodynamic_database
from aerognc.visualisation.aircraft_sandbox import plot_aircraft_sandbox
from aerognc.visualisation.ascent_guidance import plot_ascent_guidance
from aerognc.visualisation.control import plot_attitude_control_comparison
from aerognc.visualisation.flight_control_analysis import plot_flight_control_analysis
from aerognc.visualisation.flight_data_identification import (
    plot_flight_data_identification,
)
from aerognc.visualisation.flight_envelope import plot_flight_envelope
from aerognc.visualisation.flight_test import plot_flight_test_summary
from aerognc.visualisation.galaxy_catalog import plot_milky_way_catalog
from aerognc.visualisation.launch_window import plot_launch_window_optimization
from aerognc.visualisation.mission_control import (
    InterplanetaryMissionControl,
    MissionPlaybackConfiguration,
)
from aerognc.visualisation.monte_carlo import (
    plot_monte_carlo_sensitivity,
    plot_monte_carlo_summary,
)
from aerognc.visualisation.multistage_recovery import plot_multistage_recovery
from aerognc.visualisation.navigation import plot_navigation_demo
from aerognc.visualisation.orbit_assisted_tour import plot_orbit_assisted_tour
from aerognc.visualisation.orbit_sandbox import plot_orbit_sandbox
from aerognc.visualisation.playback import PlaybackConfiguration, ThreeDofPlayback
from aerognc.visualisation.playback_3d import SixDofPlayback3D
from aerognc.visualisation.rotating_ascent import plot_rotating_ascent
from aerognc.visualisation.six_dof import plot_six_dof_results


def main() -> int:
    """Generate only compact reference artefacts; omit the full trajectory CSV."""
    root = Path(__file__).resolve().parents[1]
    configuration = load_three_dof_configuration(root / "configs" / "three_dof_nominal.yaml")
    result = simulate_three_dof(configuration)
    output = root / "results" / "reference"
    orbit_sandbox_configuration = load_orbit_sandbox_configuration(
        root / "configs" / "orbit_sandbox.yaml"
    )
    orbit_sandbox = simulate_orbit_sandbox(orbit_sandbox_configuration)
    plot_orbit_sandbox(orbit_sandbox, output)
    (output / "orbit_sandbox_report.json").write_text(
        json.dumps(orbit_sandbox_payload(orbit_sandbox), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aircraft_configuration = load_aircraft_configuration(root / "configs" / "aircraft_sandbox.yaml")
    aircraft = simulate_aircraft(aircraft_configuration)
    plot_aircraft_sandbox(aircraft, output)
    (output / "aircraft_model_report.json").write_text(
        json.dumps(aircraft_sandbox_payload(aircraft), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_three_dof_results(result, output)
    write_summary_json(result, output / "three_dof_summary.json")
    playback = ThreeDofPlayback(
        result,
        PlaybackConfiguration(frames_per_second=30, initial_speed=4.0),
    )
    playback.save_snapshot(output / "three_dof_playback.png", time_s=8.0)
    playback.close()

    rotating_configuration = load_rotating_ascent_configuration(
        root / "configs" / "rotating_planet_ascent.yaml"
    )
    rotating_result = simulate_rotating_ascent(rotating_configuration)
    plot_rotating_ascent(rotating_result, output)
    write_summary_json(rotating_result, output / "rotating_planet_summary.json")

    aerodynamic_configuration = load_three_dof_configuration(
        root / "configs" / "three_dof_aero_database.yaml"
    )
    aerodynamic_analysis = analyze_aerodynamic_database(aerodynamic_configuration)
    aerodynamic_analysis = replace(
        aerodynamic_analysis,
        source_path="configs/aero_database_synthetic.csv",
    )
    write_aerodynamic_database_analysis(aerodynamic_analysis, output)
    provider = aerodynamic_configuration.vehicle.aerodynamics.coefficient_provider
    if not isinstance(provider, TabulatedAerodynamicDatabase):
        raise RuntimeError("reference aerodynamic scenario did not load its database")
    plot_aerodynamic_database(provider, output)

    envelope_configuration = load_flight_envelope_configuration(
        root / "configs" / "flight_envelope.yaml"
    )
    envelope_result = analyze_flight_envelope(envelope_configuration)
    plot_flight_envelope(envelope_result, output)
    envelope_payload = flight_envelope_payload(envelope_result)
    (output / "flight_envelope_summary.json").write_text(
        json.dumps(
            {
                "grid": envelope_payload["grid"],
                "summary": envelope_payload["summary"],
                "schedule_verification": envelope_payload["schedule_verification"],
                "robustness_verification": envelope_payload["robustness_verification"],
                "requirements": envelope_payload["requirements"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ascent_configuration = load_ascent_guidance_configuration(
        root / "configs" / "constrained_ascent_guidance.yaml"
    )
    ascent_result = optimize_ascent_guidance(ascent_configuration)
    plot_ascent_guidance(ascent_result, output)
    (output / "ascent_guidance_report.json").write_text(
        json.dumps(ascent_guidance_payload(ascent_result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    six_dof_configuration = load_six_dof_configuration(root / "configs" / "six_dof_nominal.yaml")
    six_dof_result = simulate_six_dof(six_dof_configuration)
    plot_six_dof_results(six_dof_result, output)
    write_summary_json(six_dof_result, output / "six_dof_summary.json")
    playback_3d = SixDofPlayback3D(
        six_dof_result,
        PlaybackConfiguration(frames_per_second=30, initial_speed=2.0),
    )
    playback_3d.save_snapshot(output / "six_dof_playback_3d.png", time_s=4.5)
    playback_3d.close()

    rotating_six_dof_configuration = load_rotating_six_dof_configuration(
        root / "configs" / "rotating_six_dof.yaml"
    )
    rotating_six_dof_result = simulate_rotating_six_dof(rotating_six_dof_configuration)
    plot_six_dof_results(
        rotating_six_dof_result,
        output,
        filename="rotating_six_dof_ascent.png",
    )
    write_summary_json(
        rotating_six_dof_result,
        output / "rotating_six_dof_summary.json",
    )

    multistage_configuration = load_multistage_recovery_configuration(
        root / "configs" / "multistage_recovery.yaml"
    )
    multistage_result = simulate_configured_multistage_recovery(multistage_configuration)
    plot_multistage_recovery(multistage_result, output)
    write_summary_json(multistage_result, output / "multistage_recovery_summary.json")

    interplanetary_configuration = load_interplanetary_configuration(
        root / "configs" / "interplanetary_gravity_assist.yaml"
    )
    interplanetary_mission = simulate_interplanetary(interplanetary_configuration)
    write_summary_json(
        interplanetary_mission.result,
        output / "interplanetary_summary.json",
    )
    mission_control = InterplanetaryMissionControl(
        interplanetary_mission,
        MissionPlaybackConfiguration(
            frames_per_second=24,
            playback_days_per_second=40.0,
        ),
    )
    mission_control.save_snapshot(
        output / "interplanetary_mission_control.png",
        time_days=interplanetary_configuration.snapshot_time_s / 86_400.0,
    )
    mission_control.close()

    orbit_tour_configuration = load_orbit_tour_configuration(
        root / "configs" / "orbit_assisted_tour.yaml"
    )
    orbit_tour = simulate_orbit_assisted_tour(orbit_tour_configuration)
    plot_orbit_assisted_tour(orbit_tour, output)
    (output / "orbit_assisted_tour_report.json").write_text(
        json.dumps(orbit_tour_payload(orbit_tour), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    launch_window_configuration = load_launch_window_configuration(
        root / "configs" / "launch_window_optimization.yaml"
    )
    launch_window = run_launch_window_optimization(launch_window_configuration)
    plot_launch_window_optimization(launch_window, output)
    (output / "launch_window_optimization_report.json").write_text(
        json.dumps(launch_window_payload(launch_window), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    catalog_directory = root / "data" / "catalogs"
    exoplanet_catalog = load_exoplanet_catalog(
        catalog_directory / "nasa_confirmed_exoplanets.csv",
        catalog_directory / "nasa_confirmed_exoplanets.metadata.json",
    )
    galaxy_metadata = load_milky_way_metadata(catalog_directory / "milky_way_metadata.yaml")
    solar_system = load_solar_system_planets(catalog_directory / "solar_system_planets.csv")
    catalog_selection = exoplanet_catalog.planets
    plot_milky_way_catalog(exoplanet_catalog, catalog_selection, output)
    (output / "galaxy_catalog_summary.json").write_text(
        json.dumps(
            galaxy_catalog_payload(
                exoplanet_catalog,
                catalog_selection,
                galaxy_metadata,
                solar_system,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    control_configuration = load_attitude_control_configuration(
        root / "configs" / "attitude_control.yaml"
    )
    control_results = compare_attitude_controllers(control_configuration)
    plot_attitude_control_comparison(control_results, control_configuration, output)
    metrics: dict[str, dict[str, float]] = {}
    for control_result in control_results:
        deterministic_metrics = control_result.metrics.as_dict()
        deterministic_metrics.pop("execution_time_s")
        metrics[control_result.controller_name] = deterministic_metrics
    (output / "attitude_control_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    flight_analysis_configuration = load_flight_control_analysis_configuration(
        root / "configs" / "flight_control_analysis.yaml"
    )
    flight_analysis_result = run_flight_control_analysis(flight_analysis_configuration)
    plot_flight_control_analysis(flight_analysis_result, output)
    write_flight_control_analysis(
        flight_analysis_result,
        output,
        include_timing=False,
    )

    navigation_configuration = load_navigation_demo_configuration(
        root / "configs" / "navigation_demo.yaml"
    )
    navigation_result = run_navigation_demo(navigation_configuration)
    plot_navigation_demo(navigation_result, output)
    (output / "navigation_summary.json").write_text(
        json.dumps(
            {
                "estimated_altitude_rms_m": navigation_result.estimated_altitude_rms_m,
                "raw_barometer_rms_m": navigation_result.raw_barometer_rms_m,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    advanced_navigation_configuration = load_advanced_navigation_configuration(
        root / "configs" / "advanced_navigation.yaml"
    )
    advanced_navigation_result = simulate_advanced_navigation(advanced_navigation_configuration)
    navigation_consistency = run_navigation_consistency(
        advanced_navigation_configuration,
        run_count=advanced_navigation_configuration.consistency_runs,
    )
    plot_advanced_navigation(advanced_navigation_result, navigation_consistency, output)
    (output / "advanced_navigation_report.json").write_text(
        json.dumps(
            advanced_navigation_payload(
                advanced_navigation_result,
                navigation_consistency,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    flight_data_configuration = load_flight_data_identification_configuration(
        root / "configs" / "flight_data_identification.yaml"
    )
    flight_data_workflow = run_flight_data_identification_workflow(
        flight_data_configuration,
        root / "results" / "flight_data_identification",
    )
    plot_flight_data_identification(flight_data_workflow.result, output)
    (output / "flight_data_identification_report.json").write_text(
        json.dumps(
            flight_data_identification_payload(flight_data_workflow.result),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monte_carlo_configuration = load_monte_carlo_configuration(
        root / "configs" / "monte_carlo.yaml"
    )
    monte_carlo_summary = run_monte_carlo(monte_carlo_configuration)
    plot_monte_carlo_summary(monte_carlo_summary, monte_carlo_configuration.requirements, output)
    plot_monte_carlo_sensitivity(monte_carlo_summary, output)
    (output / "monte_carlo_summary.json").write_text(
        json.dumps(
            {
                "sample_count": len(monte_carlo_summary.runs),
                "successful_count": monte_carlo_summary.successful_count,
                "failed_count": monte_carlo_summary.failed_count,
                "statistics": monte_carlo_summary.statistics,
                "requirement_pass_rates": monte_carlo_summary.requirement_pass_rates,
                "worst_case_runs": monte_carlo_summary.worst_case_runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    flight_test_workflow = run_synthetic_flight_test_workflow(
        navigation_configuration, root / "results" / "flight_test"
    )
    plot_flight_test_summary(flight_test_workflow, output)
    (output / "flight_test_summary.json").write_text(
        json.dumps(
            {
                "event_time_errors_s": flight_test_workflow.event_time_errors_s,
                "apogee_error_m": flight_test_workflow.apogee_error_m,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    loopback_configuration = SoftwareLoopbackConfiguration(
        sample_period_s=0.01,
        command_deadline_s=0.025,
        command_timeout_s=0.04,
        state_link=LinkImpairmentConfiguration(
            latency_s=0.002,
            jitter_standard_deviation_s=0.0005,
            random_seed=218,
        ),
        command_link=LinkImpairmentConfiguration(
            latency_s=0.002,
            jitter_standard_deviation_s=0.0005,
            random_seed=219,
        ),
    )
    write_software_loopback_report(
        run_software_loopback_demo(loopback_configuration, sample_count=500),
        output / "software_loopback_report.json",
    )
    write_udp_loopback_report(
        run_udp_loopback_demo(sample_count=100),
        output / "udp_loopback_report.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
