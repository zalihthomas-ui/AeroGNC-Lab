import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerognc.configuration import load_aircraft_configuration
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    FixedWingFlightModel,
    aircraft_initial_state,
    longitudinal_trim_command,
)
from aerognc.visualisation.aircraft_experience import RealtimeSimulationClock
from aerognc.visualisation.aircraft_live import AircraftLivePlayer, advance_live_aircraft
from aerognc.visualisation.mesh import load_triangle_mesh
from aerognc.visualisation.pilot_input import GamepadSnapshot


def test_live_advance_is_deterministic_and_control_changes_state() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    initial = aircraft_initial_state(configuration)
    trim = longitudinal_trim_command(configuration)

    first = advance_live_aircraft(model, initial, 0.0, 0.1, trim)
    repeated = advance_live_aircraft(model, initial, 0.0, 0.1, trim)
    roll_command = AircraftControlCommand(roll=0.5, pitch=trim.pitch, throttle=trim.throttle)
    controlled = advance_live_aircraft(model, initial, 0.0, 0.1, roll_command)

    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(controlled[10:13], first[10:13])
    assert controlled[14] > first[14]


class _FakeClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def __call__(self) -> float:
        return self.time_s


def test_live_player_maps_progressive_controls_and_advances_fixed_hidden_frame() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    mesh = load_triangle_mesh("assets/models/aquila_x1.obj")
    source = _FakeClock()
    clock = RealtimeSimulationClock(
        configuration.integration_step_s,
        time_source=source,
    )
    player = AircraftLivePlayer(
        configuration,
        mesh,
        enable_gamepad=False,
        simulation_clock=clock,
    )
    try:
        assert player.is_paused
        player._set_paused(False)
        player.pressed_keys.update(("right", "up", "d", "r"))
        player._update_pilot_inputs(0.5)
        command = player._current_command(GamepadSnapshot(False))
        assert 0.0 < command.roll <= 1.0
        assert 0.0 < command.pitch <= 1.0
        assert 0.0 < command.yaw <= 1.0
        assert command.throttle == 0.28
        assert command.rocket_assist

        start_time = player.time_s
        source.time_s += 0.1
        artists = player._animation_frame(0)
        assert player.time_s == pytest.approx(start_time + 0.1)
        assert len(artists) == 20
        assert player._snapshot(command).altitude_m > 0.0
        player._set_paused(True)
        assert not player.pressed_keys
        assert player.virtual_stick.roll == 0.0
    finally:
        plt.close(player.figure)


def test_live_physics_state_is_invariant_to_render_callback_rate() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    mesh = load_triangle_mesh("assets/models/aquila_x1.obj")

    def run(callback_step_s: float, callback_count: int) -> np.ndarray:
        source = _FakeClock()
        clock = RealtimeSimulationClock(
            configuration.integration_step_s,
            time_source=source,
        )
        player = AircraftLivePlayer(
            configuration,
            mesh,
            enable_gamepad=False,
            simulation_clock=clock,
        )
        try:
            player._set_paused(False)
            player._update_artists = lambda _command: ()
            for frame in range(callback_count):
                source.time_s += callback_step_s
                player._animation_frame(frame)
            assert player.time_s == pytest.approx(1.0)
            return player.state.copy()
        finally:
            plt.close(player.figure)

    slow_render = run(0.1, 10)
    fast_render = run(0.02, 50)
    np.testing.assert_allclose(slow_render, fast_render, rtol=0.0, atol=1.0e-12)


def test_live_one_hour_limit_caps_both_state_and_timestamp() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    mesh = load_triangle_mesh("assets/models/aquila_x1.obj")
    player = AircraftLivePlayer(configuration, mesh, enable_gamepad=False)
    try:
        command = player._current_command(GamepadSnapshot(False))
        player.time_s = 3_599.99
        initial_state = player.state.copy()
        expected_state = advance_live_aircraft(
            player.model,
            initial_state,
            player.time_s,
            3_600.0 - player.time_s,
            command,
        )

        player._advance_one_step(command, 0.02)

        assert player.time_s == 3_600.0
        np.testing.assert_allclose(player.state, expected_state, rtol=0.0, atol=1.0e-10)
        assert player.is_paused
        assert "ONE-HOUR" in player.finished_reason
    finally:
        plt.close(player.figure)
