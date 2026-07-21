from pathlib import Path

from aerognc.configuration import load_navigation_demo_configuration
from aerognc.verification.flight_test import (
    analyse_synthetic_flight_csv,
    run_synthetic_flight_test_workflow,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_measurement_only_flight_record_is_reloaded_and_events_reconstructed(
    tmp_path: Path,
) -> None:
    configuration = load_navigation_demo_configuration(
        PROJECT_ROOT / "configs" / "navigation_demo.yaml"
    )
    workflow = run_synthetic_flight_test_workflow(configuration, tmp_path)
    assert workflow.measurement_csv.is_file()
    assert workflow.summary_json.is_file()
    standalone = analyse_synthetic_flight_csv(workflow.measurement_csv, configuration)
    assert standalone.burnout_time_s == workflow.analysis.burnout_time_s
    assert abs(workflow.event_time_errors_s["burnout"]) < 0.1
    assert abs(workflow.event_time_errors_s["apogee"]) < 0.5
    assert abs(workflow.event_time_errors_s["ground_impact"]) < 0.05
    assert abs(workflow.apogee_error_m) < 5.0
