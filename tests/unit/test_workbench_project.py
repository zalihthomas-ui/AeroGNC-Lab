from pathlib import Path

import numpy as np
import pytest
import yaml

from aerognc.project.registry import (
    CancellationToken,
    WorkflowContext,
    WorkflowDescriptor,
    WorkflowRegistry,
    WorkflowResult,
)
from aerognc.project.result_store import ResultDataset
from aerognc.project.runner import ProjectRunError, ProjectRunService
from aerognc.simulation.workbench_project import ProjectWorkbenchService


def _runner(context: WorkflowContext) -> WorkflowResult:
    context.cancellation.raise_if_cancelled()
    context.report_progress(0.5, "fixture midpoint")
    offset = float(context.seed % 5)
    return WorkflowResult(
        ResultDataset(
            context.scenario_name,
            np.array([0.0, 1.0, 2.0]),
            {"altitude_m": np.array([0.0, 10.0 + offset, 20.0 + offset])},
            {"altitude_m": "m"},
        ),
        {"method": "project workbench fixture"},
    )


def _project_file(tmp_path: Path, workflow: str = "fixture") -> Path:
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "case.yaml").write_text("fictional: true\n", encoding="utf-8")
    path = tmp_path / "project.aerognc.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "workspace_root": ".",
                "name": "Workbench-Project",
                "description": "Project-aware workbench fixture.",
                "safety_scope": "Fictional civilian study with synthetic inputs.",
                "settings": {
                    "result_directory": "results",
                    "default_seed": 12,
                    "max_workers": 2,
                },
                "scenarios": [
                    {
                        "name": "case-a",
                        "workflow": workflow,
                        "configuration": "configs/case.yaml",
                        "description": "Readable scenario description.",
                        "tags": ["test"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _service() -> ProjectWorkbenchService:
    registry = WorkflowRegistry()
    registry.register(WorkflowDescriptor("fixture", "test project workflow", _runner))
    return ProjectWorkbenchService(ProjectRunService(registry))


def test_project_workbench_opens_saves_validates_runs_compares_and_reports(tmp_path) -> None:
    service = _service()
    snapshot = service.open(_project_file(tmp_path))
    assert snapshot.project_name == "Workbench-Project"
    assert snapshot.validation_issues == ()
    assert snapshot.scenarios[0].seed == 12
    assert snapshot.scenarios[0].description == "Readable scenario description."

    saved = service.save(tmp_path / "saved.aerognc.yaml")
    assert saved.project_path.name == "saved.aerognc.yaml"
    progress: list[tuple[float, str]] = []
    first = service.run_scenario(
        "case-a", progress=lambda value, text: progress.append((value, text))
    )
    second = service.run_scenario("case-a")
    assert progress == [(0.5, "fixture midpoint")]
    history = service.run_history()
    assert {record.run_id for record in history} == {
        first.manifest.run_id,
        second.manifest.run_id,
    }
    comparison = service.compare_runs(first.manifest.run_id, second.manifest.run_id)
    assert comparison.comparison.channels[0].rms_difference == 0.0
    assert comparison.json_path.is_file()
    assert comparison.report_path.is_file()
    assert service.report_path(first.manifest.run_id).is_file()


def test_project_workbench_records_cooperative_cancellation_and_unknown_workflow(tmp_path) -> None:
    service = _service()
    service.open(_project_file(tmp_path))
    token = CancellationToken()
    token.cancel()
    with pytest.raises(ProjectRunError, match="cancelled"):
        service.run_scenario("case-a", cancellation=token)
    assert service.run_history()[0].status == "cancelled"

    other_root = tmp_path / "unknown"
    unknown = _service()
    snapshot = unknown.open(_project_file(other_root, workflow="unregistered"))
    assert "unknown workflow" in snapshot.validation_issues[0]
    with pytest.raises(ValueError, match="validation failed"):
        unknown.run_scenario("case-a")
