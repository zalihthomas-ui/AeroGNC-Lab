from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from aerognc.configuration import load_three_dof_configuration
from aerognc.simulation.simulator import simulate_three_dof
from aerognc.visualisation.playback import (
    MAXIMUM_PLAYBACK_SPEED,
    MINIMUM_PLAYBACK_SPEED,
    PlaybackConfiguration,
    ThreeDofPlayback,
    flight_phase,
)

RESULT = simulate_three_dof(load_three_dof_configuration("configs/three_dof_nominal.yaml"))


def test_playback_configuration_and_flight_phases() -> None:
    with pytest.raises(ValueError, match="frames_per_second"):
        PlaybackConfiguration(frames_per_second=4)
    event_times = {event.name: event.time_s for event in RESULT.events}

    assert flight_phase(RESULT, 0.0) == "POWERED ASCENT"
    assert flight_phase(RESULT, event_times["burnout"] + 0.1) == "COAST ASCENT"
    assert flight_phase(RESULT, event_times["apogee"] + 0.1) == "DESCENT"
    assert flight_phase(RESULT, event_times["ground_impact"]) == "COMPLETE"


def test_playback_controls_seek_speed_pause_and_repeat() -> None:
    player = ThreeDofPlayback(
        RESULT, PlaybackConfiguration(frames_per_second=20, initial_speed=4.0)
    )
    try:
        player.advance_one_frame()
        assert player.current_time_s == pytest.approx(0.2)
        player.toggle_pause()
        paused_time_s = player.current_time_s
        player.advance_one_frame()
        assert player.current_time_s == paused_time_s

        player.set_speed(1_000.0)
        assert player.playback_speed == MAXIMUM_PLAYBACK_SPEED
        player.set_speed(0.001)
        assert player.playback_speed == MINIMUM_PLAYBACK_SPEED
        player.seek(1_000.0)
        assert player.current_time_s == player.end_time_s
        player.toggle_pause()
        assert player.is_playing
        assert player.current_time_s == 0.0
    finally:
        player.close()

    repeating = ThreeDofPlayback(
        RESULT,
        PlaybackConfiguration(frames_per_second=5, initial_speed=64.0, repeat=True),
    )
    try:
        repeating.seek(repeating.end_time_s - 0.01)
        repeating.advance_one_frame()
        assert repeating.current_time_s == 0.0
        assert repeating.is_playing
    finally:
        repeating.close()


def test_playback_snapshot_and_short_gif_export(tmp_path: Path) -> None:
    original_time = RESULT.time_s.copy()
    original_altitude = RESULT.columns["altitude_m"].copy()
    player = ThreeDofPlayback(
        RESULT,
        PlaybackConfiguration(
            frames_per_second=5,
            initial_speed=64.0,
            export_dpi=60,
        ),
    )
    try:
        snapshot = player.save_snapshot(tmp_path / "frame.png", time_s=8.0)
        animation = player.save_gif(tmp_path / "flight.gif")
        assert snapshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert animation.read_bytes().startswith((b"GIF87a", b"GIF89a"))
        assert animation.stat().st_size > 10_000
        with pytest.raises(ValueError, match=r"end in \.gif"):
            player.save_gif(tmp_path / "flight.mp4")
    finally:
        player.close()
    np.testing.assert_array_equal(RESULT.time_s, original_time)
    np.testing.assert_array_equal(RESULT.columns["altitude_m"], original_altitude)
