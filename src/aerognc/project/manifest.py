"""Immutable run identity, provenance, and manifest serialization."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

MANIFEST_SCHEMA_VERSION = "1.0"
RunStatus = Literal["completed", "failed", "cancelled"]
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def file_sha256(path: str | Path) -> str:
    """Return the lower-case SHA-256 of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def input_fingerprint(
    *,
    project_name: str,
    scenario_name: str,
    workflow: str,
    configuration_sha256: str,
    seed: int,
    solver_settings: Mapping[str, Any],
    parameters: Mapping[str, Any],
    safety_scope: str,
) -> str:
    """Hash every deterministic run input while excluding measured runtime."""
    if not _HASH_PATTERN.fullmatch(configuration_sha256):
        raise ValueError("configuration_sha256 must be a lower-case SHA-256")
    payload = {
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "project_name": project_name,
        "scenario_name": scenario_name,
        "workflow": workflow,
        "configuration_sha256": configuration_sha256,
        "seed": seed,
        "solver_settings": dict(solver_settings),
        "parameters": dict(parameters),
        "safety_scope": safety_scope,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def installed_version() -> str:
    """Return installed package metadata without requiring an installation."""
    try:
        return version("aerognc-lab")
    except PackageNotFoundError:
        return "0+source"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One immutable file produced inside a completed run directory."""

    role: str
    relative_path: str
    sha256: str
    media_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if not self.role or path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact role/path must be nonempty and relative")
        if not _HASH_PATTERN.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must be a lower-case SHA-256")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be nonnegative")


@dataclass(frozen=True, slots=True)
class RequirementOutcome:
    """One measurable pass/fail result attached to a run."""

    identifier: str
    passed: bool
    value: float | None = None
    limit: float | None = None
    margin: float | None = None
    unit: str = "1"
    detail: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z][A-Z0-9-]{2,63}", self.identifier):
            raise ValueError("requirement identifier has invalid form")
        for value, name in ((self.value, "value"), (self.limit, "limit"), (self.margin, "margin")):
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"requirement {name} must be numeric or null")
        if not self.unit:
            raise ValueError("requirement unit cannot be empty")


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Complete immutable provenance and assessment record for one run."""

    run_id: str
    input_fingerprint: str
    project_name: str
    scenario_name: str
    workflow: str
    safety_scope: str
    created_utc: str
    status: RunStatus
    software_version: str
    python_version: str
    platform: str
    configuration_path: str
    configuration_sha256: str
    seed: int
    solver_settings: Mapping[str, Any]
    parameters: Mapping[str, Any]
    execution_time_s: float
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None
    artifacts: tuple[ArtifactRecord, ...] = ()
    requirements: tuple[RequirementOutcome, ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    maxima: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported run-manifest schema")
        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id contains unsupported characters")
        if not _HASH_PATTERN.fullmatch(self.input_fingerprint):
            raise ValueError("input_fingerprint must be a lower-case SHA-256")
        if not _HASH_PATTERN.fullmatch(self.configuration_sha256):
            raise ValueError("configuration_sha256 must be a lower-case SHA-256")
        parsed = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("created_utc must carry UTC timezone information")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.execution_time_s < 0.0:
            raise ValueError("execution_time_s must be nonnegative")
        if self.status == "completed" and self.failure_reason is not None:
            raise ValueError("completed manifest cannot carry failure_reason")
        if self.status != "completed" and not self.failure_reason:
            raise ValueError("failed/cancelled manifest requires failure_reason")
        object.__setattr__(self, "solver_settings", MappingProxyType(dict(self.solver_settings)))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(
            self,
            "events",
            tuple(MappingProxyType(dict(event)) for event in self.events),
        )
        object.__setattr__(
            self,
            "maxima",
            MappingProxyType(
                {name: MappingProxyType(dict(record)) for name, record in self.maxima.items()}
            ),
        )


def _artifact_mapping(record: ArtifactRecord) -> dict[str, Any]:
    return {
        "role": record.role,
        "relative_path": record.relative_path,
        "sha256": record.sha256,
        "media_type": record.media_type,
        "size_bytes": record.size_bytes,
    }


def _requirement_mapping(record: RequirementOutcome) -> dict[str, Any]:
    return {
        "identifier": record.identifier,
        "passed": record.passed,
        "value": record.value,
        "limit": record.limit,
        "margin": record.margin,
        "unit": record.unit,
        "detail": record.detail,
    }


def manifest_as_mapping(manifest: RunManifest) -> dict[str, Any]:
    """Return the stable JSON-ready representation of a manifest."""
    return {
        "schema_version": manifest.schema_version,
        "run_id": manifest.run_id,
        "input_fingerprint": manifest.input_fingerprint,
        "project_name": manifest.project_name,
        "scenario_name": manifest.scenario_name,
        "workflow": manifest.workflow,
        "safety_scope": manifest.safety_scope,
        "created_utc": manifest.created_utc,
        "status": manifest.status,
        "software_version": manifest.software_version,
        "python_version": manifest.python_version,
        "platform": manifest.platform,
        "configuration_path": manifest.configuration_path,
        "configuration_sha256": manifest.configuration_sha256,
        "seed": manifest.seed,
        "solver_settings": dict(manifest.solver_settings),
        "parameters": dict(manifest.parameters),
        "execution_time_s": manifest.execution_time_s,
        "warnings": list(manifest.warnings),
        "failure_reason": manifest.failure_reason,
        "artifacts": [_artifact_mapping(item) for item in manifest.artifacts],
        "requirements": [_requirement_mapping(item) for item in manifest.requirements],
        "events": [dict(item) for item in manifest.events],
        "maxima": {name: dict(record) for name, record in manifest.maxima.items()},
    }


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a sequence")
    return cast(Sequence[object], value)


def _dict(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, Any], value)


def manifest_from_mapping(value: object) -> RunManifest:
    """Construct a validated manifest from decoded JSON."""
    data = _dict(value, "manifest")
    expected = {
        "schema_version",
        "run_id",
        "input_fingerprint",
        "project_name",
        "scenario_name",
        "workflow",
        "safety_scope",
        "created_utc",
        "status",
        "software_version",
        "python_version",
        "platform",
        "configuration_path",
        "configuration_sha256",
        "seed",
        "solver_settings",
        "parameters",
        "execution_time_s",
        "warnings",
        "failure_reason",
        "artifacts",
        "requirements",
        "events",
        "maxima",
    }
    if data.keys() != expected:
        raise ValueError(
            f"manifest keys differ; missing={sorted(expected - data.keys())}, "
            f"unknown={sorted(data.keys() - expected)}"
        )
    artifacts = tuple(
        ArtifactRecord(**_dict(item, "artifact"))
        for item in _sequence(data["artifacts"], "artifacts")
    )
    requirements = tuple(
        RequirementOutcome(**_dict(item, "requirement"))
        for item in _sequence(data["requirements"], "requirements")
    )
    warning_values = _sequence(data["warnings"], "warnings")
    if not all(isinstance(item, str) for item in warning_values):
        raise ValueError("warnings must contain strings")
    event_values = tuple(_dict(item, "event") for item in _sequence(data["events"], "events"))
    maxima_values = {
        name: _dict(item, f"maxima.{name}")
        for name, item in _dict(data["maxima"], "maxima").items()
    }
    status = data["status"]
    if status not in {"completed", "failed", "cancelled"}:
        raise ValueError("manifest status is invalid")
    return RunManifest(
        run_id=str(data["run_id"]),
        input_fingerprint=str(data["input_fingerprint"]),
        project_name=str(data["project_name"]),
        scenario_name=str(data["scenario_name"]),
        workflow=str(data["workflow"]),
        safety_scope=str(data["safety_scope"]),
        created_utc=str(data["created_utc"]),
        status=cast(RunStatus, status),
        software_version=str(data["software_version"]),
        python_version=str(data["python_version"]),
        platform=str(data["platform"]),
        configuration_path=str(data["configuration_path"]),
        configuration_sha256=str(data["configuration_sha256"]),
        seed=int(data["seed"]),
        solver_settings=_dict(data["solver_settings"], "solver_settings"),
        parameters=_dict(data["parameters"], "parameters"),
        execution_time_s=float(data["execution_time_s"]),
        warnings=tuple(cast(Sequence[str], warning_values)),
        failure_reason=None if data["failure_reason"] is None else str(data["failure_reason"]),
        artifacts=artifacts,
        requirements=requirements,
        events=event_values,
        maxima=maxima_values,
        schema_version=str(data["schema_version"]),
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write one manifest using deterministic JSON formatting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest_as_mapping(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def load_manifest(path: str | Path) -> RunManifest:
    """Load and validate one stored manifest."""
    return manifest_from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def new_run_manifest(
    *,
    project_name: str,
    scenario_name: str,
    workflow: str,
    safety_scope: str,
    configuration_path: str,
    configuration_sha256: str,
    seed: int,
    solver_settings: Mapping[str, Any],
    parameters: Mapping[str, Any],
    status: RunStatus,
    execution_time_s: float,
    warnings: Sequence[str] = (),
    failure_reason: str | None = None,
    requirements: Sequence[RequirementOutcome] = (),
    events: Sequence[Mapping[str, Any]] = (),
    maxima: Mapping[str, Mapping[str, Any]] | None = None,
    created: datetime | None = None,
) -> RunManifest:
    """Create a fully attributed manifest with deterministic input identity."""
    fingerprint = input_fingerprint(
        project_name=project_name,
        scenario_name=scenario_name,
        workflow=workflow,
        configuration_sha256=configuration_sha256,
        seed=seed,
        solver_settings=solver_settings,
        parameters=parameters,
        safety_scope=safety_scope,
    )
    instant = datetime.now(UTC) if created is None else created.astimezone(UTC)
    timestamp = instant.strftime("%Y%m%dT%H%M%S.%fZ")
    created_utc = instant.isoformat(timespec="microseconds").replace("+00:00", "Z")
    run_id = f"{scenario_name}-{timestamp}-{fingerprint[:12]}"
    return RunManifest(
        run_id=run_id,
        input_fingerprint=fingerprint,
        project_name=project_name,
        scenario_name=scenario_name,
        workflow=workflow,
        safety_scope=safety_scope,
        created_utc=created_utc,
        status=status,
        software_version=installed_version(),
        python_version=platform.python_version(),
        platform=f"{sys.platform}; {platform.machine()}; {platform.system()} {platform.release()}",
        configuration_path=configuration_path,
        configuration_sha256=configuration_sha256,
        seed=seed,
        solver_settings=solver_settings,
        parameters=parameters,
        execution_time_s=execution_time_s,
        warnings=tuple(warnings),
        failure_reason=failure_reason,
        requirements=tuple(requirements),
        events=tuple(events),
        maxima={} if maxima is None else maxima,
    )


def with_artifacts(manifest: RunManifest, artifacts: Sequence[ArtifactRecord]) -> RunManifest:
    """Return a manifest carrying its committed artefact inventory."""
    return replace(manifest, artifacts=tuple(artifacts))
