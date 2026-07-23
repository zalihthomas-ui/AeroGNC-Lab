import json
from pathlib import Path


def test_compact_reference_artifacts_and_regeneration_entry_point_are_present() -> None:
    reference = Path("results/reference")
    image_names = {
        "three_dof_kinematics.png",
        "three_dof_loads.png",
        "three_dof_trajectory.png",
        "three_dof_playback.png",
        "six_dof_ascent.png",
        "six_dof_playback_3d.png",
        "rotating_six_dof_ascent.png",
        "multistage_recovery.png",
        "attitude_control_comparison.png",
        "navigation_filter.png",
        "monte_carlo_summary.png",
        "monte_carlo_sensitivity.png",
        "flight_test_summary.png",
        "interplanetary_mission_control.png",
        "flight_control_analysis.png",
        "rotating_planet_ascent.png",
        "aerodynamic_database_analysis.png",
        "flight_envelope.png",
        "constrained_ascent_guidance.png",
        "advanced_navigation.png",
        "flight_data_identification.png",
        "orbit_assisted_tour.png",
        "launch_window_optimization.png",
        "milky_way_confirmed_exoplanet_catalog.png",
        "orbit_trajectory_3d.png",
        "orbit_decay_diagnostics.png",
        "aircraft_trajectory_3d.png",
        "aircraft_flight_diagnostics.png",
    }
    summary_names = {
        "three_dof_summary.json",
        "six_dof_summary.json",
        "attitude_control_metrics.json",
        "navigation_summary.json",
        "monte_carlo_summary.json",
        "flight_test_summary.json",
        "interplanetary_summary.json",
        "flight_control_analysis.json",
        "rotating_planet_summary.json",
        "aerodynamic_database_analysis.json",
        "flight_envelope_summary.json",
        "ascent_guidance_report.json",
        "advanced_navigation_report.json",
        "flight_data_identification_report.json",
        "orbit_assisted_tour_report.json",
        "launch_window_optimization_report.json",
        "galaxy_catalog_summary.json",
        "software_loopback_report.json",
        "rotating_six_dof_summary.json",
        "multistage_recovery_summary.json",
        "udp_loopback_report.json",
        "orbit_sandbox_report.json",
        "aircraft_model_report.json",
        "waypoint_backend_comparison.json",
    }
    for name in image_names:
        data = (reference / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 10_000
    for name in summary_names:
        payload = json.loads((reference / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and payload
    waypoint_comparison = json.loads(
        (reference / "waypoint_backend_comparison.json").read_text(encoding="utf-8")
    )
    assert waypoint_comparison["passed"] is True

    assert Path("scripts/generate_reference_results.py").is_file()
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    assert "results/**" in ignored
    assert "!results/reference/**" in ignored
