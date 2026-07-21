"""Strict YAML loading and saving for portable AeroGNC engineering projects."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from aerognc.project.models import (
    PROJECT_SCHEMA_VERSION,
    EngineeringProject,
    ProjectScenario,
    ProjectSettings,
)


class ProjectConfigurationError(ValueError):
    """Raised when a project file is malformed or unsafe."""


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProjectConfigurationError(f"{context} must be a mapping with string keys")
    return cast(Mapping[str, Any], value)


def _keys(
    value: Mapping[str, Any],
    context: str,
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ProjectConfigurationError(f"{context} missing keys: {sorted(missing)}")
    if unknown:
        raise ProjectConfigurationError(f"{context} unknown keys: {sorted(unknown)}")


def _string(value: object, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "a string" if allow_empty else "a nonempty string"
        raise ProjectConfigurationError(f"{context} must be {suffix}")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectConfigurationError(f"{context} must be an integer")
    return value


def _relative(value: object, context: str) -> Path:
    path = Path(_string(value, context))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ProjectConfigurationError(f"{context} must be a normalized relative path")
    return path


def _resolve_workspace(project_path: Path, reference: object) -> Path:
    raw = Path(_string(reference, "project.workspace_root"))
    if raw.is_absolute():
        raise ProjectConfigurationError("project.workspace_root must be relative")
    root = (project_path.parent / raw).resolve()
    if not project_path.resolve().is_relative_to(root):
        raise ProjectConfigurationError("project file must be contained by workspace_root")
    return root


def load_project(path: str | Path) -> EngineeringProject:
    """Load and validate a project without executing any workflow."""
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"project file does not exist: {source_path}")
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ProjectConfigurationError(f"invalid project YAML: {error}") from error
    root = _mapping(loaded, "project")
    _keys(
        root,
        "project",
        required={
            "schema_version",
            "workspace_root",
            "name",
            "description",
            "safety_scope",
            "settings",
            "scenarios",
        },
    )
    schema_version = _string(root["schema_version"], "project.schema_version")
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ProjectConfigurationError(
            f"unsupported project schema {schema_version!r}; expected {PROJECT_SCHEMA_VERSION!r}"
        )
    workspace_root = _resolve_workspace(source_path, root["workspace_root"])

    settings_data = _mapping(root["settings"], "project.settings")
    _keys(
        settings_data,
        "project.settings",
        required={"result_directory", "default_seed", "max_workers"},
    )
    try:
        settings = ProjectSettings(
            result_directory=_relative(
                settings_data["result_directory"], "project.settings.result_directory"
            ),
            default_seed=_integer(settings_data["default_seed"], "project.settings.default_seed"),
            max_workers=_integer(settings_data["max_workers"], "project.settings.max_workers"),
        )
    except ValueError as error:
        raise ProjectConfigurationError(str(error)) from error

    raw_scenarios = root["scenarios"]
    if not isinstance(raw_scenarios, list):
        raise ProjectConfigurationError("project.scenarios must be a sequence")
    scenarios: list[ProjectScenario] = []
    for index, raw_scenario in enumerate(raw_scenarios):
        context = f"project.scenarios[{index}]"
        data = _mapping(raw_scenario, context)
        _keys(
            data,
            context,
            required={"name", "workflow", "configuration"},
            optional={"description", "enabled", "seed", "tags", "parameters"},
        )
        tags_value = data.get("tags", [])
        if not isinstance(tags_value, list) or not all(
            isinstance(item, str) for item in tags_value
        ):
            raise ProjectConfigurationError(f"{context}.tags must be a sequence of strings")
        parameters = _mapping(data.get("parameters", {}), f"{context}.parameters")
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ProjectConfigurationError(f"{context}.enabled must be boolean")
        seed_value = data.get("seed")
        seed = None if seed_value is None else _integer(seed_value, f"{context}.seed")
        try:
            scenario = ProjectScenario(
                name=_string(data["name"], f"{context}.name"),
                workflow=_string(data["workflow"], f"{context}.workflow"),
                configuration=_relative(data["configuration"], f"{context}.configuration"),
                description=_string(
                    data.get("description", ""), f"{context}.description", allow_empty=True
                ),
                enabled=enabled,
                seed=seed,
                tags=tuple(cast(list[str], tags_value)),
                parameters=parameters,
            )
        except ValueError as error:
            raise ProjectConfigurationError(f"{context}: {error}") from error
        scenarios.append(scenario)

    try:
        project = EngineeringProject(
            name=_string(root["name"], "project.name"),
            description=_string(root["description"], "project.description"),
            safety_scope=_string(root["safety_scope"], "project.safety_scope"),
            scenarios=tuple(scenarios),
            settings=settings,
            source_path=source_path,
            workspace_root=workspace_root,
            schema_version=schema_version,
        )
        for scenario in project.scenarios:
            project.configuration_path(scenario)
        _ = project.result_root
    except (FileNotFoundError, ValueError) as error:
        raise ProjectConfigurationError(str(error)) from error
    return project


def _relative_string(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start=start)).as_posix()


def project_as_mapping(project: EngineeringProject, destination: Path) -> dict[str, Any]:
    """Return a stable YAML-ready representation relative to ``destination``."""
    workspace_reference = _relative_string(project.workspace_root, destination.parent)
    return {
        "schema_version": project.schema_version,
        "workspace_root": workspace_reference,
        "name": project.name,
        "description": project.description,
        "safety_scope": project.safety_scope,
        "settings": {
            "result_directory": project.settings.result_directory.as_posix(),
            "default_seed": project.settings.default_seed,
            "max_workers": project.settings.max_workers,
        },
        "scenarios": [
            {
                "name": scenario.name,
                "workflow": scenario.workflow,
                "configuration": scenario.configuration.as_posix(),
                "description": scenario.description,
                "enabled": scenario.enabled,
                **({"seed": scenario.seed} if scenario.seed is not None else {}),
                "tags": list(scenario.tags),
                "parameters": dict(scenario.parameters),
            }
            for scenario in project.scenarios
        ],
    }


def save_project(project: EngineeringProject, path: str | Path | None = None) -> Path:
    """Write a project deterministically and verify that it can be reloaded."""
    destination = project.source_path if path is None else Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        project_as_mapping(project, destination),
        sort_keys=False,
        allow_unicode=True,
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    load_project(destination)
    return destination


def create_empty_project(directory: str | Path, name: str) -> EngineeringProject:
    """Create a portable empty project directory and validated project file."""
    project_directory = Path(directory).resolve()
    project_directory.mkdir(parents=True, exist_ok=True)
    (project_directory / "configs").mkdir(exist_ok=True)
    (project_directory / "results").mkdir(exist_ok=True)
    project_path = project_directory / "project.aerognc.yaml"
    project = EngineeringProject(
        name=name,
        description="AeroGNC-Lab engineering analysis project.",
        safety_scope=(
            "Fictional civilian research vehicle with synthetic parameters; no target or "
            "terminal-homing logic."
        ),
        scenarios=(),
        settings=ProjectSettings(),
        source_path=project_path,
        workspace_root=project_directory,
    )
    save_project(project)
    return load_project(project_path)
