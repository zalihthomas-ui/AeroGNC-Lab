from pathlib import Path

import numpy as np
import pytest

from aerognc.project.registry import (
    WORKFLOW_API_VERSION,
    CancellationToken,
    WorkflowCancellationError,
    WorkflowContext,
    WorkflowDescriptor,
    WorkflowRegistry,
    WorkflowResult,
)
from aerognc.project.result_store import ResultDataset


def _runner(context: WorkflowContext) -> WorkflowResult:
    context.report_progress(1.0, "complete")
    return WorkflowResult(
        ResultDataset(
            context.scenario_name,
            np.array([0.0, 1.0]),
            {"altitude_m": np.array([0.0, 1.0])},
            {"altitude_m": "m"},
        ),
        {"method": "fixture"},
    )


def test_registry_orders_workflows_and_rejects_duplicates_or_versions() -> None:
    registry = WorkflowRegistry()
    registry.register(WorkflowDescriptor("z-case", "Z fixture", _runner))
    registry.register(WorkflowDescriptor("a-case", "A fixture", _runner))

    assert registry.names() == ("a-case", "z-case")
    assert registry.get("a-case").api_version == WORKFLOW_API_VERSION
    with pytest.raises(ValueError, match="already registered"):
        registry.register(WorkflowDescriptor("a-case", "duplicate", _runner))
    with pytest.raises(ValueError, match="incompatible"):
        WorkflowDescriptor("old-case", "old", _runner, api_version="0")
    with pytest.raises(KeyError, match="available"):
        registry.get("missing")


def test_workflow_context_progress_and_cooperative_cancellation(tmp_path: Path) -> None:
    configuration = tmp_path / "case.yaml"
    configuration.write_text("case: true\n", encoding="utf-8")
    updates = []
    token = CancellationToken()
    context = WorkflowContext(
        "case",
        configuration,
        tmp_path,
        5,
        progress=lambda fraction, message: updates.append((fraction, message)),
        cancellation=token,
    )

    _runner(context)
    assert updates == [(1.0, "complete")]
    token.cancel()
    with pytest.raises(WorkflowCancellationError, match="cancelled"):
        token.raise_if_cancelled()
    with pytest.raises(ValueError, match="progress"):
        context.report_progress(2.0, "bad")


class _FakeEntryPoint:
    def __init__(self, name, value, loaded):
        self.name = name
        self.value = value
        self._loaded = loaded

    def load(self):
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


def test_plugin_discovery_isolates_failures(monkeypatch) -> None:
    valid = WorkflowDescriptor("plugin-case", "plugin fixture", _runner, provider="test")
    entries = [
        _FakeEntryPoint("broken", "broken:plugin", RuntimeError("boom")),
        _FakeEntryPoint("valid", "valid:plugin", lambda: valid),
    ]
    monkeypatch.setattr(
        "aerognc.project.registry.metadata.entry_points",
        lambda **_kwargs: entries,
    )
    registry = WorkflowRegistry()

    issues = registry.discover_plugins()

    assert registry.names() == ("plugin-case",)
    assert len(issues) == 1
    assert "boom" in issues[0].reason
