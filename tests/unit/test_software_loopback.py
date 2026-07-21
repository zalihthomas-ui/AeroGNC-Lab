import numpy as np
import pytest

from aerognc.simulation.hil import LinkImpairmentConfiguration
from aerognc.simulation.software_loopback import (
    SoftwareLoopbackConfiguration,
    run_software_loopback_demo,
    synthetic_loopback_states,
)


def test_zero_latency_loopback_delivers_each_command_without_watchdog() -> None:
    result = run_software_loopback_demo(SoftwareLoopbackConfiguration(), sample_count=80)

    assert result.state_packets_accepted == 80
    assert result.command_packets_accepted == 80
    assert result.command_deadline_misses == 0
    assert result.watchdog_activations == 0
    assert result.maximum_logical_latency_s == 0.0
    assert np.any(result.applied_commands != 0.0)


def test_impaired_loopback_is_reproducible_and_rejects_duplicates() -> None:
    configuration = SoftwareLoopbackConfiguration(
        sample_period_s=0.005,
        command_deadline_s=0.012,
        command_timeout_s=0.025,
        state_link=LinkImpairmentConfiguration(
            latency_s=0.004,
            jitter_standard_deviation_s=0.003,
            loss_probability=0.08,
            duplicate_probability=0.2,
            random_seed=218,
        ),
        command_link=LinkImpairmentConfiguration(
            latency_s=0.003,
            jitter_standard_deviation_s=0.002,
            loss_probability=0.05,
            duplicate_probability=0.15,
            random_seed=219,
        ),
    )
    first = run_software_loopback_demo(configuration, sample_count=300)
    second = run_software_loopback_demo(configuration, sample_count=300)

    assert first.as_dict() == second.as_dict()
    np.testing.assert_array_equal(first.applied_commands, second.applied_commands)
    assert first.state_packets_dropped > 0
    assert first.state_packets_duplicated > 0
    assert first.state_packets_stale > 0


def test_watchdog_applies_zero_until_delayed_commands_arrive() -> None:
    configuration = SoftwareLoopbackConfiguration(
        sample_period_s=0.01,
        command_deadline_s=0.01,
        command_timeout_s=0.015,
        state_link=LinkImpairmentConfiguration(latency_s=0.05),
    )
    result = run_software_loopback_demo(configuration, sample_count=20)

    assert result.watchdog_activations >= 5
    np.testing.assert_array_equal(result.applied_commands[:5], np.zeros((5, 3)))
    assert result.command_deadline_misses > 0


def test_software_loopback_input_validation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        synthetic_loopback_states(1, 0.01)
    with pytest.raises(ValueError, match="sample_period"):
        SoftwareLoopbackConfiguration(sample_period_s=0.0)
