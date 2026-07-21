from pathlib import Path

import numpy as np

from aerognc.configuration import load_attitude_control_configuration
from aerognc.simulation.attitude_control import compare_attitude_controllers

PROJECT_ROOT = Path(__file__).parents[2]


def test_closed_loop_attitude_response_and_comparison_metrics() -> None:
    configuration = load_attitude_control_configuration(
        PROJECT_ROOT / "configs" / "attitude_control.yaml"
    )
    pid, state_feedback = compare_attitude_controllers(configuration)
    assert pid.metrics.settling_time_s < 3.0
    assert pid.metrics.overshoot_percent < 15.0
    assert pid.metrics.rms_tracking_error_rad < np.deg2rad(3.0)
    assert pid.metrics.maximum_tracking_error_rad <= configuration.reference_angle_rad
    assert pid.metrics.disturbance_recovery_s < 2.0
    assert np.all(np.isfinite(pid.angle_rad))
    assert np.all(np.isfinite(state_feedback.angle_rad))
    assert state_feedback.metrics.control_effort_nm2s > 0.0
    assert pid.metrics.execution_time_s > 0.0
