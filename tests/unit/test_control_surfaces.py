"""Unit tests for the control-surface actuator bank and failure modes."""

import numpy as np
import pytest

from aerognc.vehicle.control_surfaces import (
    ControlSurface,
    ControlSurfaceConfig,
    ControlSurfaceSet,
    SurfaceFailureMode,
)

CONFIG = ControlSurfaceConfig(max_deflection_rad=np.deg2rad(25.0), time_constant_s=0.05)


def _settle(surface: ControlSurface, command: float, steps: int = 400) -> float:
    value = 0.0
    for _ in range(steps):
        value = surface.update(command, 0.02)
    return value


def test_full_command_reaches_deflection_limit() -> None:
    surface = ControlSurface(CONFIG)
    assert _settle(surface, 1.0) == pytest.approx(np.deg2rad(25.0), abs=1e-3)
    surface.reset()
    assert _settle(surface, -1.0) == pytest.approx(-np.deg2rad(25.0), abs=1e-3)


def test_command_is_clipped_to_unit_range() -> None:
    surface = ControlSurface(CONFIG)
    assert _settle(surface, 5.0) == pytest.approx(np.deg2rad(25.0), abs=1e-3)


def test_rate_limit_slows_response() -> None:
    slow = ControlSurface(
        ControlSurfaceConfig(
            max_deflection_rad=np.deg2rad(25.0),
            time_constant_s=0.01,
            rate_limit_radps=np.deg2rad(10.0),
        )
    )
    # One 0.1 s step can move at most 1 deg at 10 deg/s.
    assert abs(slow.update(1.0, 0.1)) <= np.deg2rad(1.0) + 1e-6


def test_reversed_failure_flips_sign() -> None:
    surface = ControlSurface(CONFIG, failure=SurfaceFailureMode.REVERSED)
    assert _settle(surface, 1.0) < 0.0


def test_reduced_authority_limits_travel() -> None:
    surface = ControlSurface(
        CONFIG, failure=SurfaceFailureMode.REDUCED_AUTHORITY, reduced_authority_fraction=0.3
    )
    assert _settle(surface, 1.0) == pytest.approx(0.3 * np.deg2rad(25.0), abs=1e-3)


def test_stuck_failure_holds_position() -> None:
    surface = ControlSurface(CONFIG, failure=SurfaceFailureMode.STUCK)
    # Never moves from neutral regardless of command.
    for _ in range(50):
        assert surface.update(1.0, 0.02) == pytest.approx(0.0)


def test_loss_failure_centres_surface() -> None:
    surface = ControlSurface(CONFIG, failure=SurfaceFailureMode.LOSS)
    assert _settle(surface, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_oscillating_failure_moves_without_command() -> None:
    surface = ControlSurface(
        CONFIG, failure=SurfaceFailureMode.OSCILLATING, oscillation_amplitude_rad=np.deg2rad(5.0)
    )
    values = [surface.update(0.0, 0.02) for _ in range(50)]
    assert max(values) - min(values) > np.deg2rad(2.0)


def test_trim_offset_shifts_neutral() -> None:
    surface = ControlSurface(
        ControlSurfaceConfig(max_deflection_rad=np.deg2rad(25.0), trim_rad=np.deg2rad(2.0))
    )
    assert _settle(surface, 0.0) == pytest.approx(np.deg2rad(2.0), abs=1e-3)


def test_surface_can_start_and_reset_at_resolved_trim_position() -> None:
    surface = ControlSurface(CONFIG, initial_position_rad=0.12)
    assert surface.deflection_rad == pytest.approx(0.12)
    surface.update(-1.0, 0.1)
    surface.reset()
    assert surface.deflection_rad == pytest.approx(0.12)
    with pytest.raises(ValueError, match="initial surface position"):
        ControlSurface(CONFIG, initial_position_rad=99.0)


def test_surface_set_maps_all_channels() -> None:
    surfaces = ControlSurfaceSet.from_limits(
        aileron_limit_rad=np.deg2rad(22.0),
        elevator_limit_rad=np.deg2rad(25.0),
        rudder_limit_rad=np.deg2rad(28.0),
    )
    result = surfaces.update(0.0, 0.0, 0.0, 1.0, 0.5)
    assert result.aileron_rad == pytest.approx(0.0, abs=1e-6)
    # Throttle ramps toward the command through its first-order lag.
    assert 0.0 < result.throttle <= 1.0


def test_surface_set_throttle_settles() -> None:
    surfaces = ControlSurfaceSet.from_limits(
        aileron_limit_rad=0.3,
        elevator_limit_rad=0.3,
        rudder_limit_rad=0.3,
        initial_throttle=0.3,
    )
    throttle = surfaces.update(0.0, 0.0, 0.0, 0.3, 0.05).throttle
    assert throttle == pytest.approx(0.3)
    for _ in range(200):
        throttle = surfaces.update(0.0, 0.0, 0.0, 0.8, 0.05).throttle
    assert throttle == pytest.approx(0.8, abs=1e-2)
    surfaces.reset()
    assert surfaces.update(0.0, 0.0, 0.0, 0.3, 0.05).throttle == pytest.approx(0.3)
    with pytest.raises(ValueError, match="initial_throttle"):
        ControlSurfaceSet.from_limits(
            aileron_limit_rad=0.3,
            elevator_limit_rad=0.3,
            rudder_limit_rad=0.3,
            initial_throttle=1.1,
        )
