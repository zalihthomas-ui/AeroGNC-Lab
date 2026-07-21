from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from aerognc.configuration import load_orbit_tour_configuration
from aerognc.simulation.orbit_assisted_tour import simulate_orbit_assisted_tour
from aerognc.simulation.orbit_tour_playback import orbit_tour_playback_mission
from aerognc.visualisation.mission_control import (
    InterplanetaryMissionControl,
    MissionPlaybackConfiguration,
    mission_phase,
)

TOUR = simulate_orbit_assisted_tour(
    load_orbit_tour_configuration("configs/orbit_assisted_tour.yaml")
)
MISSION = orbit_tour_playback_mission(TOUR)


def test_orbit_tour_adapter_preserves_evidence_and_exposes_playback_energy() -> None:
    assert MISSION.result.time_s[0] == 0.0
    assert MISSION.result.time_s[-1] > 2_000.0 * 86_400.0
    assert "central_specific_energy_jkg" in MISSION.result.columns
    np.testing.assert_array_equal(
        MISSION.result.columns["position_x_m"],
        TOUR.result.columns["position_x_m"],
    )
    assert MISSION.configuration.body_with_role("assist").name == "Neria"
    assert MISSION.configuration.assist_encounter_radius_m == (
        5.0 * TOUR.tour.assist_parking_radius_m
    )


def test_orbit_tour_phases_name_capture_dwell_and_powered_departure() -> None:
    event_days = {event.name: event.time_s / 86_400.0 for event in MISSION.result.events}

    assert mission_phase(MISSION, 0.0) == "DEPARTURE PARKING-ORBIT INJECTION"
    assert mission_phase(MISSION, event_days["assist_orbit_capture"] - 1.0) == (
        "TRANSFER TO ORBIT CAPTURE"
    )
    assert "CAPTURED PARKING ORBIT" in mission_phase(
        MISSION,
        0.5 * (event_days["assist_orbit_capture"] + event_days["assist_periapsis_departure"]),
    )
    assert (
        mission_phase(
            MISSION,
            event_days["assist_periapsis_departure"] + 1.0,
        )
        == "POWERED DEPARTURE / OUTBOUND TRANSFER"
    )
    assert mission_phase(MISSION, MISSION.result.time_s[-1] / 86_400.0) == (
        "DESTINATION ORBIT CAPTURE"
    )


def test_orbit_tour_uses_shared_3d_controls_and_snapshot(tmp_path: Path) -> None:
    player = InterplanetaryMissionControl(
        MISSION,
        MissionPlaybackConfiguration(
            frames_per_second=5,
            playback_days_per_second=1_000.0,
            export_dpi=60,
        ),
        camera_mode="assist",
    )
    try:
        assert player.camera_mode == "assist"
        player.seek_next_event()
        assert player.current_day > 0.0
        player.cycle_camera()
        assert player.camera_mode == "destination"
        snapshot = player.save_snapshot(tmp_path / "orbit_tour_3d.png", time_days=240.03)
        assert snapshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert snapshot.stat().st_size > 25_000
    finally:
        player.close()
