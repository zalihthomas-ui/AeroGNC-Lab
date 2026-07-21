"""Typed, versioned engineering-project records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

PROJECT_SCHEMA_VERSION = "1.0"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _relative_path(value: Path, field_name: str) -> Path:
    if value.is_absolute():
        raise ValueError(f"{field_name} must be relative")
    if not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(f"{field_name} must be a normalized nonempty relative path")
    return value


def _name(value: str, field_name: str) -> str:
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must begin with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-' (maximum 64 characters)"
        )
    return value


def _json_scalar_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be nonempty strings")
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise ValueError(f"{field_name}.{key} must be a JSON scalar")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ProjectScenario:
    """One named workflow and its portable input reference."""

    name: str
    workflow: str
    configuration: Path
    description: str = ""
    enabled: bool = True
    seed: int | None = None
    tags: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.name, "scenario.name")
        _name(self.workflow, "scenario.workflow")
        _relative_path(self.configuration, "scenario.configuration")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("scenario.seed must be an unsigned 32-bit integer")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("scenario.tags must be unique")
        for tag in self.tags:
            _name(tag, "scenario tag")
        object.__setattr__(self, "parameters", _json_scalar_mapping(self.parameters, "parameters"))


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """Portable run defaults shared by project scenarios."""

    result_directory: Path = Path("results")
    default_seed: int = 0
    max_workers: int = 1

    def __post_init__(self) -> None:
        _relative_path(self.result_directory, "settings.result_directory")
        if (
            isinstance(self.default_seed, bool)
            or not isinstance(self.default_seed, int)
            or not 0 <= self.default_seed < 2**32
        ):
            raise ValueError("settings.default_seed must be an unsigned 32-bit integer")
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or not 1 <= self.max_workers <= 128
        ):
            raise ValueError("settings.max_workers must be an integer from 1 to 128")


@dataclass(frozen=True, slots=True)
class EngineeringProject:
    """Validated project with a resolved local workspace boundary."""

    name: str
    description: str
    safety_scope: str
    scenarios: tuple[ProjectScenario, ...]
    settings: ProjectSettings
    source_path: Path
    workspace_root: Path
    schema_version: str = PROJECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _name(self.name, "project.name")
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported project schema {self.schema_version!r}; "
                f"expected {PROJECT_SCHEMA_VERSION!r}"
            )
        if not self.description.strip():
            raise ValueError("project.description cannot be empty")
        folded = self.safety_scope.casefold()
        if not all(word in folded for word in ("fictional", "civilian", "synthetic")):
            raise ValueError("project.safety_scope must say fictional, civilian, and synthetic")
        names = [scenario.name for scenario in self.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("project scenario names must be unique")
        source = self.source_path.resolve()
        root = self.workspace_root.resolve()
        if not source.is_relative_to(root):
            raise ValueError("project file must be inside workspace_root")
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "workspace_root", root)

    @property
    def result_root(self) -> Path:
        """Return the validated absolute result-store directory."""
        return self.resolve_path(self.settings.result_directory, must_exist=False)

    def resolve_path(self, relative_path: Path, *, must_exist: bool) -> Path:
        """Resolve a portable project path without escaping the workspace root."""
        _relative_path(relative_path, "project path")
        resolved = (self.workspace_root / relative_path).resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError(f"project path escapes workspace root: {relative_path}")
        if must_exist and not resolved.is_file():
            raise FileNotFoundError(f"project input file does not exist: {relative_path}")
        return resolved

    def scenario(self, name: str) -> ProjectScenario:
        """Return one scenario by exact name."""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise KeyError(f"unknown project scenario: {name}")

    def configuration_path(self, scenario: ProjectScenario) -> Path:
        """Return a scenario configuration constrained to the workspace."""
        return self.resolve_path(scenario.configuration, must_exist=True)
