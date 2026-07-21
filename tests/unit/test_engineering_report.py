from datetime import UTC, datetime

import numpy as np

from aerognc.project.manifest import file_sha256, new_run_manifest
from aerognc.project.report import report_html, write_engineering_report
from aerognc.project.result_store import ResultDataset, ResultStore


def test_report_is_self_contained_escaped_and_print_ready(tmp_path) -> None:
    configuration = tmp_path / "case.yaml"
    configuration.write_text("case: true\n", encoding="utf-8")
    manifest = new_run_manifest(
        project_name="<script>alert('project')</script>",
        scenario_name="safe-case",
        workflow="three-dof",
        safety_scope="Fictional civilian synthetic case.",
        configuration_path="case.yaml",
        configuration_sha256=file_sha256(configuration),
        seed=4,
        solver_settings={"method": "rk4"},
        parameters={},
        status="completed",
        execution_time_s=0.1,
        warnings=("<b>synthetic warning</b>",),
        created=datetime(2026, 7, 20, tzinfo=UTC),
    )
    dataset = ResultDataset(
        "safe-case",
        np.linspace(0.0, 2.0, 5),
        {
            "altitude_m": np.array([0.0, 4.0, 7.0, 8.0, 7.0]),
            "mass_kg": np.array([10.0, 9.5, 9.0, 8.5, 8.0]),
        },
        {"altitude_m": "m", "mass_kg": "kg"},
        events=({"name": "apogee", "time_s": 1.5},),
        maxima={"altitude": {"value": 8.0, "unit": "m", "time_s": 1.5}},
    )
    stored = ResultStore(tmp_path / "runs").commit(manifest, dataset)

    html = report_html(stored)
    path = write_engineering_report(stored)

    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "&lt;b&gt;synthetic warning&lt;/b&gt;" in html
    assert "<svg" in html
    assert "@media print" in html
    assert "Requirement assessment" in html
    assert path.read_text(encoding="utf-8") == html


def test_failed_report_displays_terminal_reason(tmp_path) -> None:
    configuration = tmp_path / "case.yaml"
    configuration.write_text("case: true\n", encoding="utf-8")
    manifest = new_run_manifest(
        project_name="Project",
        scenario_name="failed-case",
        workflow="three-dof",
        safety_scope="Fictional civilian synthetic case.",
        configuration_path="case.yaml",
        configuration_sha256=file_sha256(configuration),
        seed=4,
        solver_settings={},
        parameters={},
        status="failed",
        execution_time_s=0.0,
        failure_reason="solver <failed>",
        created=datetime(2026, 7, 20, tzinfo=UTC),
    )
    stored = ResultStore(tmp_path / "runs").commit(manifest, None)
    assert "solver &lt;failed&gt;" in report_html(stored)
