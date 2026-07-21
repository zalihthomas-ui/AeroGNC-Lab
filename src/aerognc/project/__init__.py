"""Persistent engineering projects, run provenance, and result analysis."""

from aerognc.project.comparison import (
    ChannelComparison,
    RunComparison,
    compare_datasets,
    write_comparison_json,
)
from aerognc.project.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ArtifactRecord,
    RequirementOutcome,
    RunManifest,
    load_manifest,
)
from aerognc.project.models import (
    PROJECT_SCHEMA_VERSION,
    EngineeringProject,
    ProjectScenario,
    ProjectSettings,
)
from aerognc.project.registry import CancellationToken, WorkflowRegistry
from aerognc.project.report import report_html, write_engineering_report
from aerognc.project.result_store import (
    DATASET_SCHEMA_VERSION,
    ResultDataset,
    ResultIntegrityError,
    ResultStore,
    RunRecord,
    StoredRun,
)
from aerognc.project.runner import (
    ProjectRunError,
    ProjectRunService,
    built_in_workflow_registry,
)
from aerognc.project.workspace import (
    ProjectConfigurationError,
    create_empty_project,
    load_project,
    save_project,
)

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PROJECT_SCHEMA_VERSION",
    "ArtifactRecord",
    "CancellationToken",
    "ChannelComparison",
    "EngineeringProject",
    "ProjectConfigurationError",
    "ProjectRunError",
    "ProjectRunService",
    "ProjectScenario",
    "ProjectSettings",
    "RequirementOutcome",
    "ResultDataset",
    "ResultIntegrityError",
    "ResultStore",
    "RunComparison",
    "RunManifest",
    "RunRecord",
    "StoredRun",
    "WorkflowRegistry",
    "built_in_workflow_registry",
    "compare_datasets",
    "create_empty_project",
    "load_manifest",
    "load_project",
    "report_html",
    "save_project",
    "write_comparison_json",
    "write_engineering_report",
]
