"""Unit tests for navigation providers and internal fixed-wing backends."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from aerognc.configuration import AircraftSandboxConfiguration, load_aircraft_configuration
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.navigation.providers import NoisyStateProvider, PerfectStateProvider
from aerognc.navigation.state import FlightEnvironment, NavigationState
from aerognc.simulation.waypoint_backends import (
    CommandLevel,
    InternalCoefficientFixedWingBackend,
    InternalFixedWingBackend,
)
from aerognc.vehicle.control_surfaces import SurfaceDeflections
from aerognc.vehicle.fixed_wing import longitudinal_trim_elevator_rad
from aerognc.verification.waypoint_backends import WaypointCrossModelTolerances

CALM = FlightEnvironment.calm()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _truth() -> NavigationState:
    return NavigationState(
        position_ned_m=np.array([10.0, 20.0, -100.0]),
        velocity_ned_mps=np.array([20.0, 0.0, 0.0]),
        quaternion_nb=euler321_to_quaternion(0.0, 0.0, 0.0),
        angular_rate_body_radps=np.zeros(3),
        airspeed_mps=20.0,
    )


# --- providers ---------------------------------------------------------------


def test_perfect_provider_returns_truth() -> None:
    truth = _truth()
    assert PerfectStateProvider().update(truth, 0.1) is truth


def test_noisy_provider_is_reproducible() -> None:
    a = NoisyStateProvider(seed=7).update(_truth(), 0.1)
    b = NoisyStateProvider(seed=7).update(_truth(), 0.1)
    assert np.allclose(a.position_ned_m, b.position_ned_m)


def test_noisy_provider_perturbs_but_stays_close() -> None:
    estimate = NoisyStateProvider(seed=1, position_sigma_m=1.0).update(_truth(), 0.1)
    assert not np.allclose(estimate.position_ned_m, _truth().position_ned_m)
    assert np.linalg.norm(estimate.position_ned_m - _truth().position_ned_m) < 20.0


def test_gps_dropout_marks_state_invalid() -> None:
    provider = NoisyStateProvider(seed=1, gps_dropout_window_s=(0.2, 0.5))
    assert provider.update(_truth(), 0.1).valid  # t=0.1 before window
    assert not provider.update(_truth(), 0.2).valid  # t=0.3 inside window
    assert provider.update(_truth(), 0.3).valid  # t=0.6 after window


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_sigma_m": float("nan")},
        {"gps_dropout_window_s": (-1.0, 2.0)},
        {"gps_dropout_window_s": (2.0, 1.0)},
    ],
)
def test_noisy_provider_rejects_invalid_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        NoisyStateProvider(**kwargs)


@pytest.mark.parametrize("dt_s", [0.0, -0.1, float("nan")])
def test_noisy_provider_rejects_invalid_time_step(dt_s: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        NoisyStateProvider().update(_truth(), dt_s)


# --- internal backend --------------------------------------------------------


def _backend(heading=0.0, airspeed=20.0) -> InternalFixedWingBackend:
    backend = InternalFixedWingBackend()
    backend.initialize(position_ned_m=np.zeros(3), heading_rad=heading, airspeed_mps=airspeed)
    return backend


def _run(backend, deflections, seconds=5.0, dt=0.02):
    backend.send_actuator_commands(deflections)
    for _ in range(int(seconds / dt)):
        backend.step(dt, CALM)
    return backend.read_state()


def test_backend_command_level() -> None:
    assert InternalFixedWingBackend().command_level is CommandLevel.RAW_ACTUATOR


def test_wings_level_flies_straight_and_level() -> None:
    state = _run(_backend(), SurfaceDeflections(0.0, 0.0, 0.0, 0.5, saturated=False), seconds=10.0)
    assert state.course_rad == pytest.approx(0.0, abs=1e-3)  # heading unchanged
    assert abs(state.altitude_m) < 1.0  # stays near initial altitude
    assert state.position_ned_m[0] > 100.0  # travelled north


def test_positive_aileron_rolls_right() -> None:
    state = _run(_backend(), SurfaceDeflections(0.2, 0.0, 0.0, 0.5, saturated=False), seconds=1.0)
    assert state.roll_rad > 0.0


def test_positive_elevator_pitches_up_and_climbs() -> None:
    state = _run(_backend(), SurfaceDeflections(0.0, 0.1, 0.0, 0.6, saturated=False), seconds=2.0)
    assert state.pitch_rad > 0.0
    assert state.climb_rate_mps > 0.0


def test_full_throttle_accelerates() -> None:
    state = _run(_backend(airspeed=18.0), SurfaceDeflections(0.0, 0.0, 0.0, 1.0, saturated=False))
    assert state.airspeed_mps > 18.0


def test_banked_turn_changes_heading() -> None:
    # Establish a small right bank with a brief aileron input, then neutralise
    # (roll rate decays, the bank holds) -> heading turns right (yaw increases).
    backend = _backend()
    backend.send_actuator_commands(SurfaceDeflections(0.02, 0.0, 0.0, 0.5, saturated=False))
    for _ in range(25):  # 0.5 s of roll input
        backend.step(0.02, CALM)
    backend.send_actuator_commands(SurfaceDeflections(0.0, 0.0, 0.0, 0.5, saturated=False))
    for _ in range(150):  # 3 s holding the bank
        backend.step(0.02, CALM)
    state = backend.read_state()
    assert state.roll_rad > 0.0  # still banked right
    assert 0.0 < state.yaw_rad < np.pi  # turned right, not wrapped


def test_step_rejects_bad_dt() -> None:
    backend = _backend()
    with pytest.raises(ValueError):
        backend.step(0.0, CALM)


# --- coefficient-driven internal backend -----------------------------------


def _coefficient_backend(
    *,
    position_ned_m: np.ndarray | None = None,
    heading_rad: float = 0.0,
    airspeed_mps: float = 20.0,
) -> tuple[InternalCoefficientFixedWingBackend, AircraftSandboxConfiguration]:
    configuration = load_aircraft_configuration(
        REPO_ROOT / "configs" / "aircraft_waypoint_uav.yaml"
    )
    backend = InternalCoefficientFixedWingBackend(configuration, wind_horizon_s=20.0)
    backend.initialize(
        position_ned_m=(np.array([0.0, 0.0, -100.0]) if position_ned_m is None else position_ned_m),
        heading_rad=heading_rad,
        airspeed_mps=airspeed_mps,
    )
    return backend, configuration


def test_coefficient_backend_initial_state_matches_waypoint_contract() -> None:
    position = np.array([12.0, -7.0, -120.0])
    backend, _ = _coefficient_backend(position_ned_m=position, heading_rad=0.4)

    state = backend.read_state()

    np.testing.assert_allclose(state.position_ned_m, position, atol=1.0e-9)
    assert state.airspeed_mps == pytest.approx(20.0, abs=1.0e-9)
    assert state.yaw_rad == pytest.approx(0.4, abs=1.0e-9)
    assert backend.command_level is CommandLevel.RAW_ACTUATOR


def test_coefficient_backend_trimmed_flight_remains_finite_and_forward() -> None:
    backend, configuration = _coefficient_backend()
    trim_elevator = -longitudinal_trim_elevator_rad(
        configuration.initial.angle_of_attack_rad, configuration
    )
    backend.send_actuator_commands(
        SurfaceDeflections(0.0, trim_elevator, 0.0, 0.28, saturated=False)
    )

    for _ in range(250):
        backend.step(0.02, CALM)
    state = backend.read_state()

    assert np.all(np.isfinite(state.position_ned_m))
    assert state.position_ned_m[0] > 70.0
    assert abs(state.altitude_m - 100.0) < 20.0
    assert 16.0 < state.airspeed_mps < 24.0


def test_coefficient_backend_positive_aileron_rolls_right() -> None:
    backend, configuration = _coefficient_backend()
    trim_elevator = -longitudinal_trim_elevator_rad(
        configuration.initial.angle_of_attack_rad, configuration
    )
    backend.send_actuator_commands(
        SurfaceDeflections(np.deg2rad(2.0), trim_elevator, 0.0, 0.28, saturated=False)
    )

    for _ in range(25):
        backend.step(0.02, CALM)

    assert backend.read_state().roll_rad > 0.0


def test_coefficient_backend_positive_elevator_increases_pitch() -> None:
    baseline, configuration = _coefficient_backend()
    commanded, _ = _coefficient_backend()
    trim_elevator = -longitudinal_trim_elevator_rad(
        configuration.initial.angle_of_attack_rad, configuration
    )
    baseline.send_actuator_commands(
        SurfaceDeflections(0.0, trim_elevator, 0.0, 0.28, saturated=False)
    )
    commanded.send_actuator_commands(
        SurfaceDeflections(
            0.0,
            trim_elevator + np.deg2rad(1.0),
            0.0,
            0.28,
            saturated=False,
        )
    )

    for _ in range(15):
        baseline.step(0.02, CALM)
        commanded.step(0.02, CALM)

    assert commanded.read_state().pitch_rad > baseline.read_state().pitch_rad


def test_coefficient_backend_rejects_invalid_environment_and_commands() -> None:
    backend, _ = _coefficient_backend()
    crosswind = FlightEnvironment(np.array([0.0, 1.0, 0.0]), 9.80665)

    with pytest.raises(ValueError, match="environment wind differs"):
        backend.step(0.02, crosswind)
    with pytest.raises(ValueError, match="surface deflection exceeds"):
        backend.send_actuator_commands(SurfaceDeflections(1.0, 0.0, 0.0, 0.5, saturated=True))
    with pytest.raises(ValueError, match="throttle must lie"):
        backend.send_actuator_commands(SurfaceDeflections(0.0, 0.0, 0.0, 1.1, saturated=True))


def test_coefficient_backend_snapshots_configuration_provenance(tmp_path: Path) -> None:
    source = REPO_ROOT / "configs" / "aircraft_waypoint_uav.yaml"
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    expected_digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    configuration = load_aircraft_configuration(copied)
    backend = InternalCoefficientFixedWingBackend(configuration)
    copied.write_text(copied.read_text(encoding="utf-8") + "\n# changed after construction\n")

    provenance = backend.provenance()

    assert provenance["model"] == "coefficient_driven_18_state"
    assert provenance["aircraft_name"] == configuration.name
    assert provenance["aircraft_configuration_sha256"] == expected_digest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_cross_track_m": 0.0},
        {"maximum_duration_ratio": 0.5},
        {"maximum_final_altitude_difference_m": float("nan")},
    ],
)
def test_cross_model_tolerances_reject_invalid_limits(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        WaypointCrossModelTolerances(**kwargs)
