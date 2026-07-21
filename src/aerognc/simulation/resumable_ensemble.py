"""Atomic, manifest-checked, resumable deterministic engineering ensembles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

EnsembleEvaluator = Callable[[Mapping[str, float]], Mapping[str, float]]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class EnsembleDefinition:
    """Immutable experiment identity, ordered samples, and evaluator declaration."""

    name: str
    evaluator_id: str
    samples: tuple[Mapping[str, float], ...]
    metadata: Mapping[str, str | float | int | bool | None]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported ensemble schema_version; expected '1.0'")
        if not self.name.strip() or not self.evaluator_id.strip() or not self.samples:
            raise ValueError("ensemble name, evaluator_id, and samples cannot be empty")
        parameter_names = tuple(self.samples[0])
        if not parameter_names or any(not name.strip() for name in parameter_names):
            raise ValueError("ensemble parameter names cannot be empty")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("ensemble parameter names must be unique")
        copied_samples: list[Mapping[str, float]] = []
        for index, sample in enumerate(self.samples):
            if tuple(sample) != parameter_names:
                raise ValueError(
                    f"ensemble sample {index} parameter order/names differ from sample zero"
                )
            values = np.asarray(tuple(sample.values()), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"ensemble sample {index} contains nonfinite values")
            copied_samples.append(MappingProxyType({name: float(sample[name]) for name in sample}))
        metadata = dict(self.metadata)
        try:
            _canonical_json(metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("ensemble metadata must contain finite JSON values") from error
        object.__setattr__(self, "samples", tuple(copied_samples))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def payload(self) -> dict[str, Any]:
        """Return canonical manifest content excluding its hash."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "evaluator_id": self.evaluator_id,
            "samples": [dict(sample) for sample in self.samples],
            "metadata": dict(self.metadata),
        }

    @property
    def sha256(self) -> str:
        """Hash all semantics that make persisted members compatible."""
        return _sha256(self.payload())


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    """One persisted successful or gracefully failed evaluation."""

    index: int
    parameters: Mapping[str, float]
    success: bool
    metrics: Mapping[str, float]
    error: str | None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("ensemble member index must be nonnegative")
        if self.success == (self.error is not None):
            raise ValueError("successful member must omit error; failed member must include it")
        if self.success and not self.metrics:
            raise ValueError("successful ensemble member must contain metrics")
        if not self.success and self.metrics:
            raise ValueError("failed ensemble member cannot contain metrics")
        if not np.all(np.isfinite(tuple(self.parameters.values()))):
            raise ValueError("ensemble member parameters must be finite")
        if self.metrics and not np.all(np.isfinite(tuple(self.metrics.values()))):
            raise ValueError("ensemble member metrics must be finite")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class ResumableEnsembleSummary:
    """Ordered persisted members and schedule-independent aggregate evidence."""

    definition_sha256: str
    members: tuple[EnsembleMember, ...]
    complete: bool
    reused_member_count: int
    newly_executed_count: int
    metric_statistics: Mapping[str, Mapping[str, float]]
    correlations: Mapping[str, Mapping[str, float]]
    worst_case_runs: Mapping[str, int]

    @property
    def successful_count(self) -> int:
        """Return number of successful persisted members."""
        return sum(member.success for member in self.members)

    @property
    def failed_count(self) -> int:
        """Return number of failed persisted members."""
        return len(self.members) - self.successful_count


def _manifest_payload(definition: EnsembleDefinition) -> dict[str, Any]:
    return {**definition.payload(), "definition_sha256": definition.sha256}


def _prepare_manifest(definition: EnsembleDefinition, root: Path) -> None:
    path = root / "ensemble_manifest.json"
    expected = _manifest_payload(definition)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing ensemble manifest is unreadable") from error
        if existing != expected:
            raise ValueError(
                "existing ensemble manifest is incompatible with the requested definition"
            )
    else:
        _atomic_json(path, expected)


def _member_core(member: EnsembleMember, definition_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "definition_sha256": definition_sha256,
        "index": member.index,
        "parameters": dict(member.parameters),
        "success": member.success,
        "metrics": dict(member.metrics),
        "error": member.error,
    }


def _write_member(path: Path, member: EnsembleMember, definition_sha256: str) -> None:
    core = _member_core(member, definition_sha256)
    _atomic_json(path, {**core, "member_sha256": _sha256(core)})


def _load_member(
    path: Path,
    index: int,
    definition: EnsembleDefinition,
) -> EnsembleMember | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checksum = payload.pop("member_sha256")
        if checksum != _sha256(payload):
            return None
        if (
            payload["schema_version"] != "1.0"
            or payload["definition_sha256"] != definition.sha256
            or payload["index"] != index
            or payload["parameters"] != dict(definition.samples[index])
        ):
            return None
        member = EnsembleMember(
            int(payload["index"]),
            {str(name): float(value) for name, value in payload["parameters"].items()},
            bool(payload["success"]),
            {str(name): float(value) for name, value in payload["metrics"].items()},
            None if payload["error"] is None else str(payload["error"]),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    return member


def _evaluate_member(
    index: int,
    parameters: Mapping[str, float],
    evaluator: EnsembleEvaluator,
) -> EnsembleMember:
    try:
        raw_metrics = evaluator(parameters)
        metrics = {
            str(name): float(raw_metrics[name])
            for name in sorted(raw_metrics, key=lambda item: str(item))
        }
        if not metrics or any(not name.strip() for name in metrics):
            raise ValueError("evaluator returned no metrics or an empty metric name")
        if not np.all(np.isfinite(tuple(metrics.values()))):
            raise FloatingPointError("evaluator returned nonfinite metrics")
        return EnsembleMember(index, parameters, True, metrics, None)
    except Exception as error:
        return EnsembleMember(
            index,
            parameters,
            False,
            {},
            f"{type(error).__name__}: {error}",
        )


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = first - np.mean(first)
    right = second - np.mean(second)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(left @ right / denominator)


def _summaries(
    members: Sequence[EnsembleMember],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, int]]:
    successful = [member for member in members if member.success]
    if not successful:
        return {}, {}, {}
    metric_names = tuple(sorted(successful[0].metrics))
    if any(set(member.metrics) != set(metric_names) for member in successful):
        raise ValueError("successful ensemble members returned inconsistent metric names")
    statistics: dict[str, dict[str, float]] = {}
    worst: dict[str, int] = {}
    for name in metric_names:
        values = np.array([member.metrics[name] for member in successful])
        statistics[name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "p02_5": float(np.percentile(values, 2.5)),
            "median": float(np.median(values)),
            "p97_5": float(np.percentile(values, 97.5)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        worst[name] = successful[int(np.argmax(np.abs(values - np.median(values))))].index
    correlations: dict[str, dict[str, float]] = {}
    for parameter_name in successful[0].parameters:
        parameter_values = np.array([member.parameters[parameter_name] for member in successful])
        correlations[parameter_name] = {
            metric_name: _correlation(
                parameter_values,
                np.array([member.metrics[metric_name] for member in successful]),
            )
            for metric_name in metric_names
        }
    return statistics, correlations, worst


def run_resumable_ensemble(
    definition: EnsembleDefinition,
    evaluator: EnsembleEvaluator,
    output_directory: str | Path,
    *,
    workers: int = 1,
    retry_failed: bool = False,
    new_member_limit: int | None = None,
) -> ResumableEnsembleSummary:
    """Reuse valid members, atomically persist new ones, and return ordered evidence."""
    if workers <= 0:
        raise ValueError("ensemble workers must be positive")
    if new_member_limit is not None and new_member_limit < 0:
        raise ValueError("new_member_limit must be nonnegative when provided")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    _prepare_manifest(definition, root)
    member_root = root / "members"
    member_root.mkdir(exist_ok=True)
    loaded: dict[int, EnsembleMember] = {}
    missing: list[int] = []
    for index in range(len(definition.samples)):
        path = member_root / f"{index:08d}.json"
        member = _load_member(path, index, definition)
        if member is None or (retry_failed and not member.success):
            missing.append(index)
        else:
            loaded[index] = member
    selected = missing if new_member_limit is None else missing[:new_member_limit]

    def execute(index: int) -> EnsembleMember:
        return _evaluate_member(index, definition.samples[index], evaluator)

    if workers == 1:
        executed = [execute(index) for index in selected]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executed = list(executor.map(execute, selected))
    for member in executed:
        _write_member(
            member_root / f"{member.index:08d}.json",
            member,
            definition.sha256,
        )
        loaded[member.index] = member
    ordered = tuple(loaded[index] for index in sorted(loaded))
    statistics, correlations, worst = _summaries(ordered)
    summary = ResumableEnsembleSummary(
        definition.sha256,
        ordered,
        len(ordered) == len(definition.samples),
        len(loaded) - len(executed),
        len(executed),
        MappingProxyType(statistics),
        MappingProxyType(correlations),
        MappingProxyType(worst),
    )
    write_ensemble_summary(summary, root / "ensemble_summary.json")
    return summary


def write_ensemble_summary(summary: ResumableEnsembleSummary, path: str | Path) -> Path:
    """Write deterministic aggregate/member evidence without runtime measurements."""
    destination = Path(path)
    payload = {
        "schema_version": "1.0",
        "definition_sha256": summary.definition_sha256,
        "complete": summary.complete,
        "successful_count": summary.successful_count,
        "failed_count": summary.failed_count,
        "members": [
            {
                "index": member.index,
                "parameters": dict(member.parameters),
                "success": member.success,
                "metrics": dict(member.metrics),
                "error": member.error,
            }
            for member in summary.members
        ],
        "metric_statistics": {
            name: dict(values) for name, values in summary.metric_statistics.items()
        },
        "correlations": {name: dict(values) for name, values in summary.correlations.items()},
        "worst_case_runs": dict(summary.worst_case_runs),
    }
    _atomic_json(destination, payload)
    return destination
