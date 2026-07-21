import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from aerognc.project import load_project
from aerognc.project.models import EngineeringProject, ProjectScenario, ProjectSettings
from aerognc.project.registry import (
    CancellationToken,
    WorkflowContext,
    WorkflowDescriptor,
    WorkflowRegistry,
)
from aerognc.project.result_store import ResultStore
from aerognc.project.runner import (
    ProjectRunError,
    ProjectRunService,
    built_in_workflow_registry,
)


def _project_fixture(tmp_path: Path) -> EngineeringProject:
    shutil.copytree(Path("configs"), tmp_path / "configs")
    path = tmp_path / "project.aerognc.yaml"
    data = {
        "schema_version": "1.0",
        "workspace_root": ".",
        "name": "Integration-Project",
        "description": "Project service integration fixture.",
        "safety_scope": "Fictional civilian research vehicles with synthetic parameters.",
        "settings": {
            "result_directory": "project-results",
            "default_seed": 27,
            "max_workers": 1,
        },
        "scenarios": [
            {
                "name": "point-mass",
                "workflow": "three-dof",
                "configuration": "configs/three_dof_nominal.yaml",
            },
            {
                "name": "rigid-body",
                "workflow": "six-dof",
                "configuration": "configs/six_dof_nominal.yaml",
            },
            {
                "name": "planetary-tour",
                "workflow": "orbit-tour",
                "configuration": "configs/orbit_assisted_tour.yaml",
            },
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_project(path)


def test_project_service_runs_and_reloads_all_built_in_workflows(tmp_path) -> None:
    project = _project_fixture(tmp_path)
    service = ProjectRunService()
    start = datetime(2026, 7, 20, 16, 0, tzinfo=UTC)
    updates = []

    stored_runs = tuple(
        service.run(
            project,
            scenario.name,
            created=start + timedelta(seconds=index),
            progress=lambda fraction, message: updates.append((fraction, message)),
        )
        for index, scenario in enumerate(project.scenarios)
    )

    assert service.validate_workflows(project) == ()
    assert built_in_workflow_registry().names() == (
        "multistage-recovery",
        "orbit-tour",
        "rotating-six-dof",
        "six-dof",
        "three-dof",
    )
    assert all(stored.manifest.status == "completed" for stored in stored_runs)
    assert all(stored.dataset is not None for stored in stored_runs)
    assert all((stored.directory / "report.html").is_file() for stored in stored_runs)
    assert all(all(item.passed for item in stored.manifest.requirements) for stored in stored_runs)
    assert updates[-1][0] == 1.0

    records = ResultStore(project.result_root).list_runs(project_name=project.name)
    assert len(records) == 3
    reloaded = ResultStore(project.result_root).load(stored_runs[0].manifest.run_id)
    assert reloaded.dataset is not None
    assert reloaded.dataset.scenario_name == "three_dof_nominal"


def _failing_runner(_context: WorkflowContext):
    raise RuntimeError("intentional fixture failure")


def test_project_service_records_failure_and_cancellation(tmp_path) -> None:
    configuration = tmp_path / "case.yaml"
    configuration.write_text("case: true\n", encoding="utf-8")
    project = EngineeringProject(
        name="Failure-Project",
        description="Failure recording fixture.",
        safety_scope="Fictional civilian vehicle with synthetic parameters.",
        scenarios=(ProjectScenario("case", "failure-case", Path("case.yaml")),),
        settings=ProjectSettings(Path("results"), 5, 1),
        source_path=tmp_path / "project.aerognc.yaml",
        workspace_root=tmp_path,
    )
    registry = WorkflowRegistry()
    registry.register(WorkflowDescriptor("failure-case", "failure fixture", _failing_runner))
    service = ProjectRunService(registry)
    created = datetime(2026, 7, 20, 17, 0, tzinfo=UTC)

    with pytest.raises(ProjectRunError, match="intentional fixture failure") as failure:
        service.run(project, "case", created=created)
    failed = ResultStore(project.result_root).load(failure.value.run_id)
    assert failed.manifest.status == "failed"
    assert failed.dataset is None
    assert (failed.directory / "report.html").is_file()

    cancelled_project = EngineeringProject(
        name="Cancel-Project",
        description="Cancellation recording fixture.",
        safety_scope=project.safety_scope,
        scenarios=(ProjectScenario("case", "three-dof", Path("case.yaml")),),
        settings=ProjectSettings(Path("cancelled-results"), 5, 1),
        source_path=project.source_path,
        workspace_root=tmp_path,
    )
    token = CancellationToken()
    token.cancel()
    with pytest.raises(ProjectRunError, match="cancelled") as cancellation:
        ProjectRunService().run(cancelled_project, "case", cancellation=token, created=created)
    cancelled = ResultStore(cancelled_project.result_root).load(cancellation.value.run_id)
    assert cancelled.manifest.status == "cancelled"


def test_project_service_reports_unknown_workflows(tmp_path) -> None:
    configuration = tmp_path / "case.yaml"
    configuration.write_text("case: true\n", encoding="utf-8")
    project = EngineeringProject(
        name="Unknown-Project",
        description="Unknown workflow fixture.",
        safety_scope="Fictional civilian vehicle with synthetic parameters.",
        scenarios=(ProjectScenario("case", "not-installed", Path("case.yaml")),),
        settings=ProjectSettings(),
        source_path=tmp_path / "project.aerognc.yaml",
        workspace_root=tmp_path,
    )
    assert "unknown workflow" in ProjectRunService().validate_workflows(project)[0]
