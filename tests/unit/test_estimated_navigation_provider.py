"""Unit tests for the truth-isolated waypoint estimated-navigation provider."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aerognc.configuration.waypoint_loader import load_waypoint_runtime_configuration
from aerognc.gnc.delayed_error_state_ekf import InnovationGateConfiguration
from aerognc.navigation.estimated_provider import (
    EstimatedNavigationParameters,
    EstimatedNavigationProvider,
)
from aerognc.navigation.state import NavigationState
from aerognc.vehicle.sensors import SensorErrorParameters

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "waypoint_gnc_estimated.yaml"


def _parameters() -> EstimatedNavigationParameters:
    runtime = load_waypoint_runtime_configuration(CONFIG)
    parameters = runtime.navigation.estimated_parameters
    assert parameters is not None
    return parameters


def _truth(time_s: float) -> NavigationState:
    return NavigationState(
        position_ned_m=np.array([20.0 * time_s, 0.0, -100.0]),
        velocity_ned_mps=np.array([20.0, 0.0, 0.0]),
        quaternion_nb=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_radps=np.zeros(3),
        airspeed_mps=20.0,
    )


def _sequence(provider: EstimatedNavigationProvider, count: int = 30) -> np.ndarray:
    step_s = provider.parameters.step_s
    return np.asarray(
        [provider.update(_truth(index * step_s), step_s).position_ned_m for index in range(count)]
    )


def test_estimated_provider_is_seeded_resettable_and_does_not_alias_truth() -> None:
    provider = EstimatedNavigationProvider(_parameters())
    truth = _truth(0.0)
    first = provider.update(truth, provider.parameters.step_s)

    assert not np.shares_memory(first.position_ned_m, truth.position_ned_m)
    assert not np.shares_memory(first.velocity_ned_mps, truth.velocity_ned_mps)
    assert not np.array_equal(first.position_ned_m, truth.position_ned_m)
    assert first.valid
    provider.reset()
    first_sequence = _sequence(provider)
    provider.reset()
    second_sequence = _sequence(provider)
    np.testing.assert_array_equal(first_sequence, second_sequence)

    metadata = json.dumps(
        {"provenance": provider.provenance(), "diagnostics": provider.diagnostics()},
        allow_nan=False,
    )
    assert "truth" not in metadata.lower()
    assert provider.provenance()["filter"] == "fixed_lag_rotating_ned_15_state_eskf"
    with pytest.raises(ValueError, match="configured step"):
        provider.update(_truth(2.0), 0.1)


def test_estimated_provider_gates_outliers_and_fails_invalid_gnss_health() -> None:
    parameters = _parameters()
    source = parameters.gnss
    biased_gnss = SensorErrorParameters(
        sample_rate_hz=source.sample_rate_hz,
        noise_std=source.noise_std,
        constant_bias=[1000.0, -1000.0, 800.0, 100.0, 100.0, -100.0],
        bias_drift_std_per_sqrt_s=source.bias_drift_std_per_sqrt_s,
        quantisation=source.quantisation,
        delay_s=source.delay_s,
        dropout_probability=0.0,
        dropout_intervals_s=(),
    )
    poisoned = replace(
        parameters,
        gnss=biased_gnss,
        innovation_gate=InnovationGateConfiguration(
            gnss_nis_threshold=1.0,
            barometer_nis_threshold=12.0,
            degraded_after_rejections=1,
            failed_after_rejections=2,
        ),
    )
    provider = EstimatedNavigationProvider(poisoned)
    estimate = _truth(0.0)
    for index in range(30):
        estimate = provider.update(_truth(index * poisoned.step_s), poisoned.step_s)

    diagnostics = provider.diagnostics()
    integrity = diagnostics["gnss_integrity"]
    assert isinstance(integrity, dict)
    assert integrity["rejected_count"] >= 2
    assert integrity["health"] == "failed"
    assert not estimate.valid


def test_estimated_parameters_reject_bad_cadence_and_report_unavailable_airdata() -> None:
    parameters = _parameters()
    with pytest.raises(ValueError, match="sample period"):
        replace(parameters, step_s=0.04)

    source = parameters.airspeed
    unavailable_airdata = SensorErrorParameters(
        sample_rate_hz=source.sample_rate_hz,
        noise_std=source.noise_std,
        constant_bias=source.constant_bias,
        bias_drift_std_per_sqrt_s=source.bias_drift_std_per_sqrt_s,
        quantisation=source.quantisation,
        delay_s=source.delay_s,
        dropout_probability=0.0,
        dropout_intervals_s=((0.0, 0.1),),
    )
    provider = EstimatedNavigationProvider(replace(parameters, airspeed=unavailable_airdata))
    estimate = provider.update(_truth(0.0), parameters.step_s)
    diagnostics = provider.diagnostics()
    assert not estimate.valid
    assert diagnostics["airspeed_age_s"] is None
    json.dumps(diagnostics, allow_nan=False)
