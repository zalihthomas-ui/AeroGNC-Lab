from dataclasses import replace
from pathlib import Path

import numpy as np

from aerognc.configuration.advanced_navigation_loader import (
    load_advanced_navigation_configuration,
)
from aerognc.simulation.advanced_navigation import (
    run_navigation_consistency,
    simulate_advanced_navigation,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_short_advanced_navigation_run_and_consistency_are_reproducible() -> None:
    configured = load_advanced_navigation_configuration(
        PROJECT_ROOT / "configs" / "advanced_navigation.yaml"
    )
    short = replace(configured, duration_s=2.0, faults=(), consistency_runs=2)
    first = simulate_advanced_navigation(short)
    second = simulate_advanced_navigation(short)
    np.testing.assert_array_equal(first.estimated_position_ned_m, second.estimated_position_ned_m)
    np.testing.assert_array_equal(first.nees_15, second.nees_15)
    assert first.maximum_replayed_step_count == 18
    assert len(first.aiding_updates) > 20
    assert np.all(first.position_sigma_m > 0.0)
    assert first.uncompensated_position_error_m > 0.0

    consistency_first = run_navigation_consistency(short)
    consistency_second = run_navigation_consistency(short)
    np.testing.assert_array_equal(
        consistency_first.mean_nees_15,
        consistency_second.mean_nees_15,
    )
    assert consistency_first.seeds == consistency_second.seeds
