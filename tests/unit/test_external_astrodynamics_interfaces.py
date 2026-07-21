from pathlib import Path

import numpy as np
import pytest

from aerognc.interoperability.external_tools import (
    compare_gmat_report,
    detect_external_astrodynamics_tools,
    write_gmat_two_body_script,
)


def test_external_detection_never_claims_implicit_execution() -> None:
    statuses = detect_external_astrodynamics_tools()
    assert {status.name for status in statuses} == {"GMAT", "SPICE/spiceypy"}
    assert all(status.executed is False for status in statuses)
    assert all(status.message for status in statuses)


def test_gmat_script_is_generated_and_report_comparison_converts_units(tmp_path: Path) -> None:
    script = write_gmat_two_body_script(
        tmp_path / "case.script", duration_s=120.0, report_step_s=60.0
    )
    text = script.read_text(encoding="utf-8")
    assert "EarthPointMass" in text
    assert "For index = 1:2" in text

    time_s = np.array([0.0, 60.0, 120.0])
    states_si = np.arange(18, dtype=float).reshape(3, 6) * 1_000.0
    report = np.column_stack((time_s, states_si / 1_000.0))
    report_path = tmp_path / "report.txt"
    np.savetxt(report_path, report)
    metrics = compare_gmat_report(report_path, time_s, states_si)
    assert metrics["maximum_position_error_m"] == pytest.approx(0.0)
    assert metrics["maximum_velocity_error_mps"] == pytest.approx(0.0)
