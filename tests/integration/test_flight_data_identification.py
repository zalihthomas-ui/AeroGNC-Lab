import json
from pathlib import Path

import pytest

from aerognc.configuration.flight_data_loader import (
    load_flight_data_identification_configuration,
)
from aerognc.verification.flight_data_identification import (
    assess_flight_data_identification,
    run_flight_data_identification_workflow,
)
from aerognc.visualisation.flight_data_identification import (
    plot_flight_data_identification,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_flight_data_workflow_aligns_identifies_and_validates(tmp_path: Path) -> None:
    configuration = load_flight_data_identification_configuration(
        PROJECT_ROOT / "configs" / "flight_data_identification.yaml"
    )
    workflow = run_flight_data_identification_workflow(configuration, tmp_path)
    assessment = assess_flight_data_identification(workflow.result)
    figure_path = plot_flight_data_identification(workflow.result, tmp_path)

    assert assessment.all_pass
    assert workflow.result.clock_alignment.offset_s == pytest.approx(0.370, abs=0.001)
    assert workflow.result.detected_outlier_count >= len(configuration.outliers)
    assert workflow.result.identification_r_squared > 0.99
    assert workflow.result.validation_pitch_rms_deg < 0.10
    assert workflow.logs.command_log.stat().st_size > 10_000
    assert workflow.logs.sensor_log.stat().st_size > 10_000
    assert workflow.aligned_csv.stat().st_size > 10_000
    assert figure_path.stat().st_size > 25_000
    payload = json.loads(workflow.report_json.read_text(encoding="utf-8"))
    assert payload["requirements"]["all_pass"] is True

    repeat = run_flight_data_identification_workflow(configuration, tmp_path / "repeat")
    assert repeat.logs.command_log.read_bytes() == workflow.logs.command_log.read_bytes()
    assert repeat.logs.sensor_log.read_bytes() == workflow.logs.sensor_log.read_bytes()
    assert repeat.report_json.read_bytes() == workflow.report_json.read_bytes()
