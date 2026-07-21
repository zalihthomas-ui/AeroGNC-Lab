from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from aerognc.configuration import load_six_dof_configuration
from aerognc.mathematics.quaternion import normalize_quaternion, quaternion_to_dcm
from aerognc.simulation.six_dof_simulator import simulate_six_dof
from aerognc.visualisation.playback import (
    MAXIMUM_PLAYBACK_SPEED,
    MINIMUM_PLAYBACK_SPEED,
    PlaybackConfiguration,
)
from aerognc.visualisation.playback_3d import SixDofPlayback3D, six_dof_flight_phase

RESULT = simulate_six_dof(load_six_dof_configuration("configs/six_dof_nominal.yaml"))


def test_six_dof_flight_phases_and_camera_validation() -> None:
    burnout_time_s = next(event.time_s for event in RESULT.events if event.name == "burnout")

    assert six_dof_flight_phase(RESULT, 0.0) == "POWERED ASCENT"
    assert six_dof_flight_phase(RESULT, burnout_time_s + 0.1) == "COAST ASCENT"
    assert six_dof_flight_phase(RESULT, RESULT.time_s[-1]) == "SCENARIO COMPLETE"
    with pytest.raises(ValueError, match="finite"):
        six_dof_flight_phase(RESULT, np.nan)
    with pytest.raises(ValueError, match="camera_mode"):
        SixDofPlayback3D(RESULT, camera_mode="invalid")  # type: ignore[arg-type]


def test_3d_playback_controls_camera_speed_pause_and_repeat() -> None:
    player = SixDofPlayback3D(
        RESULT,
        PlaybackConfiguration(frames_per_second=20, initial_speed=4.0),
    )
    try:
        line_east, line_north, line_altitude = player.vehicle_body.get_data_3d()
        displayed_forward = np.array(
            [
                line_east[1] - line_east[0],
                line_north[1] - line_north[0],
                line_altitude[1] - line_altitude[0],
            ]
        )
        displayed_forward /= np.linalg.norm(displayed_forward)
        initial_quaternion = normalize_quaternion(
            np.array([RESULT.columns[f"quaternion_q{index}"][0] for index in range(4)])
        )
        navigation_forward = quaternion_to_dcm(initial_quaternion)[:, 0]
        expected_forward = np.array(
            [navigation_forward[1], navigation_forward[0], -navigation_forward[2]]
        )
        np.testing.assert_allclose(displayed_forward, expected_forward, atol=1.0e-12)

        player.advance_one_frame()
        assert player.current_time_s == pytest.approx(0.2)
        player.cycle_camera()
        assert player.camera_mode == "chase"
        player.set_camera_mode("free")
        assert player.camera_mode == "free"

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

    repeating = SixDofPlayback3D(
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


def test_3d_playback_snapshot_and_short_gif_export(tmp_path: Path) -> None:
    original_time = RESULT.time_s.copy()
    original_quaternion = RESULT.columns["quaternion_q0"].copy()
    player = SixDofPlayback3D(
        RESULT,
        PlaybackConfiguration(
            frames_per_second=5,
            initial_speed=64.0,
            export_dpi=60,
        ),
    )
    try:
        snapshot = player.save_snapshot(tmp_path / "frame_3d.png", time_s=4.5)
        animation = player.save_gif(tmp_path / "flight_3d.gif")
        assert snapshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert animation.read_bytes().startswith((b"GIF87a", b"GIF89a"))
        assert animation.stat().st_size > 10_000
        with pytest.raises(ValueError, match=r"end in \.gif"):
            player.save_gif(tmp_path / "flight_3d.mp4")
    finally:
        player.close()
    np.testing.assert_array_equal(RESULT.time_s, original_time)
    np.testing.assert_array_equal(RESULT.columns["quaternion_q0"], original_quaternion)
