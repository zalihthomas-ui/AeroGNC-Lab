import matplotlib.pyplot as plt
import numpy as np

from aerognc.configuration import load_aircraft_configuration
from aerognc.simulation.aircraft_telemetry import aircraft_telemetry
from aerognc.vehicle.fixed_wing import (
    FixedWingFlightModel,
    aircraft_initial_state,
    longitudinal_trim_command,
)
from aerognc.visualisation.aircraft_experience import FlightRecorder, load_recorded_flight
from aerognc.visualisation.aircraft_live import advance_live_aircraft
from aerognc.visualisation.aircraft_replay import AircraftReplayPlayer
from aerognc.visualisation.mesh import load_triangle_mesh


def test_aircraft_replay_seeks_recorded_states_without_evaluating_plant(
    tmp_path, monkeypatch
) -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    command = longitudinal_trim_command(configuration)
    initial = aircraft_initial_state(configuration)
    following = advance_live_aircraft(model, initial, 0.0, 0.1, command)
    recorder = FlightRecorder(maximum_samples=10)
    recorder.append(
        initial,
        command,
        aircraft_telemetry(model, 0.0, initial, command, initial_position_inertial_m=initial[:3]),
    )
    recorder.append(
        following,
        command,
        aircraft_telemetry(model, 0.1, following, command, initial_position_inertial_m=initial[:3]),
    )
    artifacts = recorder.write(tmp_path)
    recording = load_recorded_flight(artifacts.csv_path)
    monkeypatch.setattr(
        FixedWingFlightModel,
        "loads",
        lambda *_args: (_ for _ in ()).throw(AssertionError("replay evaluated plant loads")),
    )
    player = AircraftReplayPlayer(
        configuration,
        load_triangle_mesh("assets/models/aquila_x1.obj"),
        recording,
    )
    try:
        player.set_time(0.1)
        np.testing.assert_allclose(player.state, following, atol=1.0e-12)
        assert player.telemetry.altitude_m == recording.sample_telemetry(0.1).altitude_m
        player.set_time(0.05)
        assert player.time_s == 0.05
        assert np.linalg.norm(player.state[6:10]) == 1.0
        assert "State + telemetry are interpolated" in player.telemetry_text.get_text()
        gif_path = player.export_gif(
            tmp_path / "replay.gif",
            maximum_duration_s=0.1,
            frames_per_second=2,
            width_px=320,
            height_px=180,
        )
        assert gif_path.stat().st_size > 1_000
    finally:
        plt.close(player.figure)
