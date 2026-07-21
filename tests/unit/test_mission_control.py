from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from aerognc.configuration import load_interplanetary_configuration
from aerognc.simulation.interplanetary import simulate_interplanetary
from aerognc.visualisation.mission_control import (
    MAXIMUM_DAYS_PER_SECOND,
    MINIMUM_DAYS_PER_SECOND,
    InterplanetaryMissionControl,
    MissionPlaybackConfiguration,
    mission_phase,
)

MISSION = simulate_interplanetary(
    load_interplanetary_configuration("configs/interplanetary_gravity_assist.yaml")
)


def test_mission_phases_and_player_configuration_validation() -> None:
    event_days = {event.name: event.time_s / 86_400.0 for event in MISSION.result.events}
    assert mission_phase(MISSION, 0.0) == "DEPARTURE INJECTION"
    assert mission_phase(MISSION, event_days["assist_entry"] - 1.0).startswith("CRUISE")
    assert "GRAVITY ASSIST" in mission_phase(MISSION, event_days["assist_closest_approach"])
    assert mission_phase(MISSION, event_days["assist_exit"] + 1.0) == "OUTBOUND TRANSFER"
    assert mission_phase(MISSION, event_days["destination_arrival"] + 1.0) == (
        "DESTINATION ENCOUNTER"
    )
    with pytest.raises(ValueError, match="camera_mode"):
        InterplanetaryMissionControl(MISSION, camera_mode="bad")  # type: ignore[arg-type]


def test_mission_control_seek_speed_event_camera_pause_and_repeat() -> None:
    player = InterplanetaryMissionControl(
        MISSION,
        MissionPlaybackConfiguration(frames_per_second=20, playback_days_per_second=40.0),
    )
    try:
        player.advance_one_frame()
        assert player.current_day == pytest.approx(2.0)
        player.cycle_camera()
        assert player.camera_mode == "spacecraft"
        player.set_camera_mode("free")
        player.seek(0.0)
        player.seek_next_event()
        assert player.current_day == pytest.approx(
            next(
                event.time_s / 86_400.0
                for event in MISSION.result.events
                if event.name == "assist_entry"
            )
        )
        player.toggle_pause()
        paused_day = player.current_day
        player.advance_one_frame()
        assert player.current_day == paused_day
        player.set_speed(1.0e9)
        assert player.playback_days_per_second == MAXIMUM_DAYS_PER_SECOND
        player.set_speed(1.0e-9)
        assert player.playback_days_per_second == MINIMUM_DAYS_PER_SECOND
    finally:
        player.close()

    repeating = InterplanetaryMissionControl(
        MISSION,
        MissionPlaybackConfiguration(
            frames_per_second=5,
            playback_days_per_second=1_000.0,
            repeat=True,
        ),
    )
    try:
        repeating.seek(repeating.end_day - 0.01)
        repeating.advance_one_frame()
        assert repeating.current_day == 0.0
        assert repeating.is_playing
    finally:
        repeating.close()


def test_mission_control_snapshot_and_short_gif_are_non_mutating(tmp_path: Path) -> None:
    original_time = MISSION.result.time_s.copy()
    original_x = MISSION.result.columns["position_x_m"].copy()
    player = InterplanetaryMissionControl(
        MISSION,
        MissionPlaybackConfiguration(
            frames_per_second=5,
            playback_days_per_second=1_000.0,
            export_dpi=60,
        ),
    )
    try:
        snapshot = player.save_snapshot(tmp_path / "mission.png", time_days=1_342.0)
        animation = player.save_gif(tmp_path / "mission.gif")
        assert snapshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert animation.read_bytes().startswith((b"GIF87a", b"GIF89a"))
        assert animation.stat().st_size > 20_000
        with pytest.raises(ValueError, match=r"end in \.gif"):
            player.save_gif(tmp_path / "mission.mp4")
    finally:
        player.close()
    np.testing.assert_array_equal(MISSION.result.time_s, original_time)
    np.testing.assert_array_equal(MISSION.result.columns["position_x_m"], original_x)
