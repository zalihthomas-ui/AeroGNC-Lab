"""Testable project/run-history services used by the native desktop workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerognc.project.comparison import RunComparison, compare_datasets, write_comparison_json
from aerognc.project.models import EngineeringProject
from aerognc.project.registry import CancellationToken, ProgressCallback
from aerognc.project.report import write_engineering_report
from aerognc.project.result_store import ResultStore, RunRecord, StoredRun
from aerognc.project.runner import ProjectRunService, scenario_seed
from aerognc.project.workspace import load_project, save_project


@dataclass(frozen=True, slots=True)
class ProjectScenarioView:
    """Plain display record for one validated project scenario."""

    name: str
    workflow: str
    configuration: str
    description: str
    enabled: bool
    seed: int
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectWorkbenchSnapshot:
    """Project identity, validation issues, scenarios, and immutable run history."""

    project_name: str
    project_path: Path
    workspace_root: Path
    result_root: Path
    description: str
    safety_scope: str
    validation_issues: tuple[str, ...]
    scenarios: tuple[ProjectScenarioView, ...]
    runs: tuple[RunRecord, ...]


@dataclass(frozen=True, slots=True)
class ProjectRunComparison:
    """Compatible comparison plus its JSON and self-contained HTML evidence."""

    comparison: RunComparison
    json_path: Path
    report_path: Path
    baseline_run_id: str
    candidate_run_id: str


class ProjectWorkbenchService:
    """Stateful UI facade over the same validated service used by the CLI."""

    def __init__(self, run_service: ProjectRunService | None = None) -> None:
        self.run_service = ProjectRunService() if run_service is None else run_service
        self._project: EngineeringProject | None = None

    @property
    def project(self) -> EngineeringProject:
        """Return the open project or fail with a UI-readable message."""
        if self._project is None:
            raise RuntimeError("open an AeroGNC project before using project actions")
        return self._project

    def open(self, path: str | Path) -> ProjectWorkbenchSnapshot:
        """Load strict YAML and make it the active workbench project."""
        self._project = load_project(path)
        return self.snapshot()

    def save(self, path: str | Path | None = None) -> ProjectWorkbenchSnapshot:
        """Save the current project deterministically and reload the written file."""
        destination = save_project(self.project, path)
        self._project = load_project(destination)
        return self.snapshot()

    def validation_issues(self) -> tuple[str, ...]:
        """Return workflow compatibility issues after structural/path validation."""
        return self.run_service.validate_workflows(self.project)

    def scenario_views(self) -> tuple[ProjectScenarioView, ...]:
        """Return enabled state and resolved seed without exposing mutable UI models."""
        project = self.project
        return tuple(
            ProjectScenarioView(
                scenario.name,
                scenario.workflow,
                scenario.configuration.as_posix(),
                scenario.description,
                scenario.enabled,
                scenario_seed(project, scenario),
                scenario.tags,
            )
            for scenario in project.scenarios
        )

    def run_history(self) -> tuple[RunRecord, ...]:
        """Read newest-first history from the rebuildable local result index."""
        project = self.project
        return ResultStore(project.result_root).list_runs(project_name=project.name)

    def snapshot(self) -> ProjectWorkbenchSnapshot:
        """Refresh all project browser records without executing a scenario."""
        project = self.project
        return ProjectWorkbenchSnapshot(
            project.name,
            project.source_path,
            project.workspace_root,
            project.result_root,
            project.description,
            project.safety_scope,
            self.validation_issues(),
            self.scenario_views(),
            self.run_history(),
        )

    def run_scenario(
        self,
        scenario_name: str,
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> StoredRun:
        """Execute through the shared runner and persist terminal evidence."""
        issues = self.validation_issues()
        if issues:
            raise ValueError("project workflow validation failed: " + "; ".join(issues))
        return self.run_service.run(
            self.project,
            scenario_name,
            cancellation=cancellation,
            progress=progress,
        )

    def stored_run(self, run_id: str) -> StoredRun:
        """Load and integrity-check one selected run."""
        return ResultStore(self.project.result_root).load(run_id)

    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> ProjectRunComparison:
        """Compare every common same-unit channel and write JSON/HTML evidence."""
        if baseline_run_id == candidate_run_id:
            raise ValueError("select two different runs to compare")
        baseline = self.stored_run(baseline_run_id)
        candidate = self.stored_run(candidate_run_id)
        if baseline.dataset is None or candidate.dataset is None:
            raise ValueError("only completed runs with trajectories can be compared")
        common_channels = tuple(
            name
            for name in baseline.dataset.channels
            if name in candidate.dataset.channels
            and baseline.dataset.units[name] == candidate.dataset.units[name]
        )
        if not common_channels:
            raise ValueError("selected runs have no common channels with matching units")
        comparison = compare_datasets(
            baseline.dataset,
            candidate.dataset,
            channels=common_channels,
        )
        output = self.project.result_root / "comparisons"
        stem = f"{baseline_run_id}_vs_{candidate_run_id}"
        json_path = write_comparison_json(comparison, output / f"{stem}.json")
        report_path = write_engineering_report(
            candidate,
            output / f"{stem}.html",
            comparison=comparison,
        )
        return ProjectRunComparison(
            comparison,
            json_path,
            report_path,
            baseline_run_id,
            candidate_run_id,
        )

    def report_path(self, run_id: str) -> Path:
        """Return a regenerated self-contained report for one integrity-checked run."""
        return write_engineering_report(self.stored_run(run_id))
