"""Unit tests for waypoint backend trim resolution and failure policy."""

from pathlib import Path

import numpy as np
import pytest

from aerognc.configuration.aircraft_loader import load_aircraft_configuration
from aerognc.simulation.waypoint_backends import ReducedFixedWingParams
from aerognc.simulation.waypoint_mission import WaypointMissionConfig, _resolve_trim
from aerognc.simulation.waypoint_trim import (
    TrimConvergenceError,
    TrimFailurePolicy,
    WaypointTrimOptions,
    configuration_with_resolved_trim,
    solve_coefficient_waypoint_trim,
    solve_reduced_waypoint_trim,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _coefficient_trim(options: WaypointTrimOptions):
    configuration = load_aircraft_configuration(
        REPO_ROOT / "configs" / "aircraft_waypoint_uav.yaml"
    )
    return configuration, solve_coefficient_waypoint_trim(
        configuration,
        altitude_m=100.0,
        airspeed_mps=20.0,
        heading_rad=0.0,
        steady_wind_ned_mps=(0.0, 0.0, 0.0),
        elevator_command_limit_rad=float(np.deg2rad(25.0)),
        options=options,
    )


def test_coefficient_trim_converges_and_updates_initial_condition() -> None:
    configuration, result = _coefficient_trim(WaypointTrimOptions(enabled=True))

    assert result.converged
    assert not result.used_fallback
    assert result.iterations <= 5
    assert result.residual_infinity_norm < 1.0e-8
    assert np.rad2deg(result.angle_of_attack_rad) == pytest.approx(2.69267039)
    assert np.rad2deg(result.elevator_deflection_rad) == pytest.approx(3.28780649)
    assert result.throttle == pytest.approx(0.2984337403)
    assert result.autopilot_trim.elevator_command == pytest.approx(
        result.elevator_deflection_rad / np.deg2rad(25.0)
    )

    resolved = configuration_with_resolved_trim(configuration, result)
    assert resolved.initial.angle_of_attack_rad == pytest.approx(result.angle_of_attack_rad)
    assert resolved.initial_throttle == pytest.approx(result.throttle)


def test_trim_failure_policy_rejects_or_falls_back_explicitly() -> None:
    strict = WaypointTrimOptions(enabled=True, tolerance=1.0e-14, maximum_iterations=1)
    with pytest.raises(TrimConvergenceError, match="failed"):
        _coefficient_trim(strict)

    fallback = WaypointTrimOptions(
        enabled=True,
        failure_policy=TrimFailurePolicy.FALLBACK_CONFIGURED,
        tolerance=1.0e-14,
        maximum_iterations=1,
    )
    _, result = _coefficient_trim(fallback)
    assert not result.converged
    assert result.used_fallback
    assert result.source == "configured_fallback_after_failure"


def test_reduced_trim_is_analytic_and_bounded() -> None:
    result = solve_reduced_waypoint_trim(ReducedFixedWingParams(), airspeed_mps=20.0)
    assert result.converged
    assert result.iterations == 0
    assert result.throttle == pytest.approx((20.0 - 12.0) / (45.0 - 12.0))
    assert result.residual_infinity_norm == pytest.approx(0.0)


def test_trim_options_reject_invalid_domains() -> None:
    with pytest.raises(ValueError, match="increasing"):
        WaypointTrimOptions(minimum_angle_of_attack_rad=0.2, maximum_angle_of_attack_rad=0.1)
    with pytest.raises(ValueError, match="positive integer"):
        WaypointTrimOptions(maximum_iterations=0)
    with pytest.raises(ValueError, match="boolean"):
        WaypointTrimOptions(enabled=1)  # type: ignore[arg-type]


def test_reduced_trim_fallback_metadata_matches_configured_commands() -> None:
    config = WaypointMissionConfig(
        initial_airspeed_mps=50.0,
        trim_options=WaypointTrimOptions(
            enabled=True,
            failure_policy=TrimFailurePolicy.FALLBACK_CONFIGURED,
        ),
    )
    trim, result, coefficient_configuration = _resolve_trim(config, heading_rad=0.0)

    assert result is not None
    assert coefficient_configuration is None
    assert not result.converged
    assert result.used_fallback
    assert result.source == "configured_reduced_fallback_after_failure"
    assert result.autopilot_trim == trim
    assert result.throttle == pytest.approx(config.autopilot_gains.throttle_trim)
    assert result.elevator_command == pytest.approx(config.autopilot_gains.elevator_trim)
    assert result.residual_infinity_norm > 0.0
