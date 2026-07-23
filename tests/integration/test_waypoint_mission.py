"""Integration and scenario tests for the end-to-end waypoint GNC loop."""

import json
from pathlib import Path

import numpy as np
import pytest

from aerognc.configuration.waypoint_loader import load_waypoint_runtime_configuration
from aerognc.gnc.waypoint_guidance import GuidanceMode
from aerognc.mission import (
    HomePosition,
    Mission,
    MissionDefaults,
    Waypoint,
    WaypointAction,
    load_mission,
)
from aerognc.mission.mission_manager import MissionState
from aerognc.navigation.providers import NoisyStateProvider
from aerognc.simulation.waypoint_mission import (
    WaypointMissionConfig,
    run_waypoint_mission,
)
from aerognc.vehicle.control_surfaces import SurfaceFailureMode
from aerognc.verification.waypoint_backends import (
    compare_waypoint_vehicle_models,
    write_waypoint_cross_model_comparison,
)
from aerognc.verification.waypoint_control import (
    run_waypoint_control_campaign,
    write_waypoint_control_campaign,
)
from aerognc.verification.waypoint_navigation import (
    run_waypoint_navigation_campaign,
    write_waypoint_navigation_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _straight_mission(action_last: WaypointAction = WaypointAction.FLY_THROUGH) -> Mission:
    return Mission(
        name="scenario",
        home=HomePosition(0.0, 0.0, 0.0),
        defaults=MissionDefaults(airspeed_mps=20.0, acceptance_radius_m=40.0),
        waypoints=(
            Waypoint(id=1, name="A", latitude_deg=0.006, longitude_deg=0.0, altitude_m=120.0),
            Waypoint(id=2, name="B", latitude_deg=0.012, longitude_deg=0.004, altitude_m=160.0),
            Waypoint(
                id=3,
                name="C",
                latitude_deg=0.012,
                longitude_deg=0.010,
                altitude_m=160.0,
                action=action_last,
            ),
        ),
    )


def _no_nans(result) -> None:
    for sample in result.samples:
        assert np.isfinite(sample.north_m)
        assert np.isfinite(sample.altitude_m)
        assert np.isfinite(sample.airspeed_mps)


# --- nominal ----------------------------------------------------------------


def test_nominal_multi_waypoint_completes() -> None:
    result = run_waypoint_mission(_straight_mission())
    assert result.completed
    assert result.final_state is MissionState.MISSION_COMPLETE
    _no_nans(result)


def test_bundled_demo_mission_completes() -> None:
    from aerognc.mission import load_mission

    mission = load_mission(REPO_ROOT / "missions" / "waypoint_demo.mission.yaml")
    result = run_waypoint_mission(mission)
    assert result.completed
    summary = result.summary()
    assert summary["min_airspeed_mps"] > 10.0  # never stalled to a standstill
    assert summary["max_abs_cross_track_m"] < 400.0


def test_coefficient_backend_completes_and_passes_cross_model_bounds(tmp_path: Path) -> None:
    from aerognc.mission import load_mission

    reduced_runtime = load_waypoint_runtime_configuration(
        REPO_ROOT / "configs" / "waypoint_gnc.yaml"
    )
    coefficient_runtime = load_waypoint_runtime_configuration(
        REPO_ROOT / "configs" / "waypoint_gnc_coefficient.yaml"
    )
    comparison = compare_waypoint_vehicle_models(
        load_mission(coefficient_runtime.mission_path),
        reduced_runtime.build_mission_config(),
        coefficient_runtime.build_mission_config(),
    )
    result = comparison.coefficient

    assert comparison.passed
    assert result.completed
    assert result.metadata["safety_events"] == []
    assert result.planned_path_ned_m[0, 2] == pytest.approx(-100.0)
    summary = result.summary()
    assert summary["max_abs_cross_track_m"] < 120.0
    assert summary["min_altitude_m"] > 90.0
    assert summary["min_airspeed_mps"] > 15.0
    backend_details = result.metadata["vehicle_backend_details"]
    assert isinstance(backend_details, dict)
    assert backend_details["model"] == "coefficient_driven_18_state"
    assert len(str(backend_details["aircraft_configuration_sha256"])) == 64
    report = write_waypoint_cross_model_comparison(comparison, tmp_path / "comparison.json")
    assert '"passed": true' in report.read_text(encoding="utf-8")


def test_estimated_navigation_completes_outage_and_recovers_with_bounded_error(
    tmp_path: Path,
) -> None:
    runtime = load_waypoint_runtime_configuration(
        REPO_ROOT / "configs" / "waypoint_gnc_estimated.yaml"
    )
    parameters = runtime.navigation.estimated_parameters
    assert parameters is not None
    campaign = run_waypoint_navigation_campaign(parameters)
    assert campaign.passed
    assert campaign.outage_maximum_position_error_m < 10.0
    assert campaign.recovery_position_rms_m < 0.5
    report = write_waypoint_navigation_campaign(campaign, tmp_path / "navigation.json")
    assert '"passed": true' in report.read_text(encoding="utf-8")

    result = run_waypoint_mission(
        load_mission(runtime.mission_path),
        runtime.build_mission_config(),
    )
    assert result.completed
    assert result.metadata["safety_events"] == []
    assert result.summary()["max_abs_cross_track_m"] < 120.0
    details = result.metadata["navigation_provider_details"]
    diagnostics = result.metadata["navigation_diagnostics"]
    assert isinstance(details, dict) and details["mode"] == "estimated"
    assert isinstance(diagnostics, dict)
    assert diagnostics["maximum_gnss_age_s"] == pytest.approx(20.45)
    assert diagnostics["imu_held_step_count"] == 0
    assert (
        "truth"
        not in json.dumps(
            {"details": details, "diagnostics": diagnostics},
            allow_nan=False,
        ).lower()
    )


def test_trim_tecs_and_geometric_transitions_pass_on_both_internal_backends(
    tmp_path: Path,
) -> None:
    runtime = load_waypoint_runtime_configuration(REPO_ROOT / "configs" / "waypoint_gnc_tecs.yaml")
    campaign = run_waypoint_control_campaign(
        load_mission(runtime.mission_path),
        runtime.build_mission_config(),
    )
    assert campaign.passed
    assert campaign.scenario.horizontal_wind_mps == pytest.approx(1.0)
    assert campaign.coefficient.maximum_cross_track_m < 11.0
    assert campaign.reduced.maximum_cross_track_m < 21.0
    assert campaign.coefficient.actuator_saturation_samples == 0
    assert campaign.reduced.actuator_saturation_samples == 0
    assert campaign.terminal_separation_m < 1.5
    assert campaign.coefficient.maximum_course_command_step_rad <= np.deg2rad(3.0) + 1.0e-12
    assert campaign.reduced.maximum_course_command_step_rad <= np.deg2rad(3.0) + 1.0e-12
    report = write_waypoint_control_campaign(campaign, tmp_path / "control.json")
    assert '"passed": true' in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", list(GuidanceMode))
def test_all_guidance_modes_complete(mode: GuidanceMode) -> None:
    result = run_waypoint_mission(_straight_mission(), WaypointMissionConfig(guidance_mode=mode))
    assert result.completed


# --- environment / disturbances ---------------------------------------------


def test_crosswind_mission_completes_and_tracks() -> None:
    config = WaypointMissionConfig(wind_ned_mps=(0.0, 6.0, 0.0))  # 6 m/s from the west
    result = run_waypoint_mission(_straight_mission(), config)
    assert result.completed
    # After settling, cross-track stays bounded despite the steady crosswind.
    settled = [abs(s.cross_track_error_m) for s in result.samples[len(result.samples) // 2 :]]
    assert max(settled) < 120.0


def test_gps_dropout_mission_still_completes() -> None:
    config = WaypointMissionConfig(
        provider=NoisyStateProvider(seed=3, gps_dropout_window_s=(20.0, 30.0))
    )
    result = run_waypoint_mission(_straight_mission(), config)
    # It should either complete or safely return home / abort - never crash or NaN.
    assert result.outcome in {"complete", "abort"}
    _no_nans(result)


# --- return home & safety ---------------------------------------------------


def test_return_home_waypoint_completes_at_home() -> None:
    result = run_waypoint_mission(_straight_mission(WaypointAction.RETURN_HOME))
    assert result.completed
    last = result.samples[-1]
    assert np.hypot(last.north_m, last.east_m) < 100.0  # ended near home horizontally


def test_geofence_breach_triggers_return_or_abort() -> None:
    from aerognc.mission.safety import SafetyLimits

    # A tight geofence the mission legs exceed -> safety should intervene.
    config = WaypointMissionConfig(safety_limits=SafetyLimits(geofence_radius_m=300.0))
    result = run_waypoint_mission(_straight_mission(), config)
    triggered = [e for e in result.metadata["safety_events"]]  # type: ignore[index]
    assert any(event[1] == "geofence" for event in triggered)


# --- actuator failure -------------------------------------------------------


def test_elevator_failure_does_not_crash_the_simulation() -> None:
    config = WaypointMissionConfig(
        surface_failures={"elevator": SurfaceFailureMode.STUCK}, max_time_s=300.0
    )
    result = run_waypoint_mission(_straight_mission(), config)
    # With a stuck elevator the mission likely will not complete, but the loop
    # must terminate cleanly with finite state and a definite outcome.
    assert result.outcome in {"complete", "abort", "emergency", "timeout"}
    _no_nans(result)


# --- logging / export -------------------------------------------------------


def test_csv_and_json_export(tmp_path: Path) -> None:
    result = run_waypoint_mission(_straight_mission())
    csv_path = result.to_csv(tmp_path / "log.csv")
    json_path = result.to_json(tmp_path / "log.json")
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert json_path.is_file() and json_path.stat().st_size > 0
