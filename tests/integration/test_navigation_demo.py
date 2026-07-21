from pathlib import Path

import numpy as np

from aerognc.configuration import load_navigation_demo_configuration
from aerognc.simulation.navigation_demo import run_navigation_demo

PROJECT_ROOT = Path(__file__).parents[2]


def test_navigation_demo_is_reproducible_and_improves_altitude_rms() -> None:
    configuration = load_navigation_demo_configuration(
        PROJECT_ROOT / "configs" / "navigation_demo.yaml"
    )
    first = run_navigation_demo(configuration)
    second = run_navigation_demo(configuration)
    np.testing.assert_array_equal(first.estimated_altitude_m, second.estimated_altitude_m)
    assert first.estimated_altitude_rms_m < first.raw_barometer_rms_m
    assert first.estimated_altitude_rms_m < 2.0
    assert np.any(~np.isfinite(first.measured_gnss_altitude_m))
    assert np.all(first.altitude_sigma_m > 0.0)
    altitude_normalised_error = (
        np.abs(first.estimated_altitude_m - first.true_altitude_m) / first.altitude_sigma_m
    )
    velocity_normalised_error = (
        np.abs(first.estimated_vertical_velocity_up_mps - first.true_vertical_velocity_up_mps)
        / first.velocity_sigma_mps
    )
    assert np.mean(altitude_normalised_error <= 2.0) > 0.90
    assert np.mean(velocity_normalised_error <= 2.0) > 0.90
