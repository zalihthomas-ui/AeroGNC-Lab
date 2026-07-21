"""Versioned workflow registry shared by CLI, desktop, and optional plugins."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from aerognc.project.manifest import RequirementOutcome
from aerognc.project.result_store import ResultDataset

WORKFLOW_API_VERSION = "1.0"
_WORKFLOW_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
ProgressCallback = Callable[[float, str], None]


class WorkflowCancellationError(RuntimeError):
    """Raised cooperatively when a project run is cancelled."""


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation; repeated calls are harmless."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Stop a cooperative workflow at a safe boundary."""
        if self.cancelled:
            raise WorkflowCancellationError("run cancelled by user")


@dataclass(frozen=True, slots=True)
class WorkflowContext:
    """Validated, UI-independent request passed to a workflow runner."""

    scenario_name: str
    configuration_path: Path
    workspace_root: Path
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    progress: ProgressCallback | None = None

    def __post_init__(self) -> None:
        if not self.configuration_path.is_file():
            raise FileNotFoundError(f"workflow configuration is missing: {self.configuration_path}")
        if not self.configuration_path.resolve().is_relative_to(self.workspace_root.resolve()):
            raise ValueError("workflow configuration escapes workspace root")
        if not 0 <= self.seed < 2**32:
            raise ValueError("workflow seed must be an unsigned 32-bit integer")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def report_progress(self, fraction: float, message: str) -> None:
        """Validate and emit one optional progress update."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("progress fraction must be within [0, 1]")
        if self.progress is not None:
            self.progress(fraction, message)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Common successful result returned by every registered workflow."""

    dataset: ResultDataset
    solver_settings: Mapping[str, Any]
    requirements: tuple[RequirementOutcome, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "solver_settings", MappingProxyType(dict(self.solver_settings)))


@runtime_checkable
class WorkflowRunner(Protocol):
    """Callable protocol implemented by built-in and optional workflows."""

    def __call__(self, context: WorkflowContext) -> WorkflowResult:
        """Execute one validated request and return an immutable result."""
        ...


@dataclass(frozen=True, slots=True)
class WorkflowDescriptor:
    """One registry entry and its API compatibility declaration."""

    name: str
    description: str
    runner: WorkflowRunner
    api_version: str = WORKFLOW_API_VERSION
    provider: str = "aerognc-lab"

    def __post_init__(self) -> None:
        if not _WORKFLOW_PATTERN.fullmatch(self.name):
            raise ValueError("workflow name must use lower-case letters, digits, and hyphens")
        if not self.description.strip() or not self.provider.strip():
            raise ValueError("workflow description and provider cannot be empty")
        if self.api_version != WORKFLOW_API_VERSION:
            raise ValueError(
                f"workflow {self.name!r} API {self.api_version!r} is incompatible with "
                f"{WORKFLOW_API_VERSION!r}"
            )
        if not isinstance(self.runner, WorkflowRunner):
            raise TypeError("workflow runner must be callable")


@dataclass(frozen=True, slots=True)
class PluginIssue:
    """Isolated optional-plugin discovery failure."""

    entry_point: str
    reason: str


class WorkflowRegistry:
    """Deterministically ordered built-ins plus isolated optional entry points."""

    def __init__(self) -> None:
        self._descriptors: dict[str, WorkflowDescriptor] = {}
        self._plugin_issues: list[PluginIssue] = []

    def register(self, descriptor: WorkflowDescriptor) -> None:
        """Register one unique API-compatible workflow."""
        if descriptor.name in self._descriptors:
            raise ValueError(f"workflow is already registered: {descriptor.name}")
        self._descriptors[descriptor.name] = descriptor

    def get(self, name: str) -> WorkflowDescriptor:
        """Return a workflow or raise a contextual lookup error."""
        try:
            return self._descriptors[name]
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown workflow {name!r}; available: {available}") from error

    def names(self) -> tuple[str, ...]:
        """Return deterministic workflow names."""
        return tuple(sorted(self._descriptors))

    def descriptors(self) -> tuple[WorkflowDescriptor, ...]:
        """Return deterministic immutable descriptor order."""
        return tuple(self._descriptors[name] for name in self.names())

    @property
    def plugin_issues(self) -> tuple[PluginIssue, ...]:
        """Return optional discovery failures without disabling built-ins."""
        return tuple(self._plugin_issues)

    def discover_plugins(self, group: str = "aerognc.workflows") -> tuple[PluginIssue, ...]:
        """Load optional descriptors and isolate all entry-point failures."""
        try:
            entry_points = metadata.entry_points(group=group)
        except Exception as error:  # pragma: no cover - environment metadata failure
            issue = PluginIssue(group, f"entry-point query failed: {type(error).__name__}: {error}")
            self._plugin_issues.append(issue)
            return (issue,)
        new_issues: list[PluginIssue] = []
        for entry_point in sorted(entry_points, key=lambda item: (item.name, item.value)):
            label = f"{entry_point.name}={entry_point.value}"
            try:
                loaded = entry_point.load()
                descriptor = (
                    loaded()
                    if callable(loaded) and not isinstance(loaded, WorkflowDescriptor)
                    else loaded
                )
                if not isinstance(descriptor, WorkflowDescriptor):
                    raise TypeError(
                        "entry point must provide WorkflowDescriptor or a zero-argument factory"
                    )
                self.register(descriptor)
            except Exception as error:
                issue = PluginIssue(label, f"{type(error).__name__}: {error}")
                self._plugin_issues.append(issue)
                new_issues.append(issue)
        return tuple(new_issues)
