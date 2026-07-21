import json
from pathlib import Path

from aerognc.diagnostics import (
    format_diagnostic_report,
    run_diagnostics,
    write_diagnostic_report,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_repository_diagnostic_passes_required_checks_without_running_tools(
    tmp_path: Path,
) -> None:
    report = run_diagnostics(project_root=PROJECT_ROOT, result_directory=tmp_path)

    assert report.passed
    assert all(check.status == "pass" for check in report.checks if check.required)
    integrity = next(
        check for check in report.checks if check.name == "Exoplanet catalog integrity"
    )
    assert "6324 records" in integrity.detail
    optional = [check for check in report.checks if not check.required]
    assert optional
    assert all(
        "executed" not in check.detail or "not executed" in check.detail for check in optional
    )
    assert not tuple(tmp_path.glob(".aerognc-write-probe-*"))


def test_diagnostic_missing_project_reports_actionable_required_failures(tmp_path: Path) -> None:
    report = run_diagnostics(
        project_root=tmp_path / "missing-project",
        result_directory=tmp_path,
    )
    text = format_diagnostic_report(report)

    assert not report.passed
    assert "Overall readiness: NOT READY" in text
    assert "Next:" in text
    assert any(check.required and check.status == "fail" for check in report.checks)


def test_diagnostic_rejects_a_result_path_that_is_a_file(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    report = run_diagnostics(project_root=PROJECT_ROOT, result_directory=output_file)

    result_check = next(check for check in report.checks if check.name == "Result location")
    assert result_check.status == "fail"
    assert not report.passed


def test_diagnostic_json_records_scope_and_remediation(tmp_path: Path) -> None:
    report = run_diagnostics(project_root=PROJECT_ROOT, result_directory=tmp_path)
    path = write_diagnostic_report(report, tmp_path / "health.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["passed"] is True
    assert "no dependency installation" in payload["scope"]
    assert all("remediation" in check for check in payload["checks"])
