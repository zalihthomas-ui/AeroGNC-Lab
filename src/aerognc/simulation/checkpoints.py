"""Versioned, checksummed numerical-propagation checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.adaptive_integrators import AdaptiveIntegrationResult
from aerognc.mathematics.vectors import FloatArray

CHECKPOINT_SCHEMA_VERSION = "1.0"


class CheckpointIntegrityError(RuntimeError):
    """Raised when persisted checkpoint evidence fails integrity checks."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: FloatArray) -> str:
    portable = np.asarray(state, dtype="<f8")
    return hashlib.sha256(portable.tobytes(order="C")).hexdigest()


def _validated_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        encoded = _canonical_bytes(dict(metadata))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must contain finite JSON values") from error
    if not isinstance(decoded, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    return MappingProxyType(cast(dict[str, Any], decoded))


@dataclass(frozen=True, slots=True)
class IntegratorCheckpoint:
    """Restart state at an epoch-relative propagation time."""

    epoch: str
    time_s: float
    state: npt.ArrayLike
    next_step_s: float
    metadata: Mapping[str, Any]
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema {self.schema_version!r}")
        if not self.epoch.strip():
            raise ValueError("checkpoint epoch cannot be empty")
        if not np.isfinite(self.time_s):
            raise ValueError("checkpoint time_s must be finite")
        if not np.isfinite(self.next_step_s) or self.next_step_s <= 0.0:
            raise ValueError("checkpoint next_step_s must be positive and finite")
        state = np.asarray(self.state, dtype=np.float64)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("checkpoint state must be a non-empty finite vector")
        state = state.copy()
        state.flags.writeable = False
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))


def checkpoint_from_result(
    result: AdaptiveIntegrationResult,
    *,
    epoch: str,
    metadata: Mapping[str, Any] | None = None,
) -> IntegratorCheckpoint:
    """Create a restart record from the final accepted adaptive state."""
    return IntegratorCheckpoint(
        epoch=epoch,
        time_s=float(result.time_s[-1]),
        state=result.state[-1],
        next_step_s=result.statistics.recommended_next_step_s,
        metadata={} if metadata is None else metadata,
    )


def _content_mapping(checkpoint: IntegratorCheckpoint) -> dict[str, Any]:
    state = cast(FloatArray, checkpoint.state)
    return {
        "schema_version": checkpoint.schema_version,
        "epoch": checkpoint.epoch,
        "time_s": checkpoint.time_s,
        "next_step_s": checkpoint.next_step_s,
        "state_shape": list(state.shape),
        "state_dtype": "float64",
        "state_sha256": _state_sha256(state),
        "metadata": dict(checkpoint.metadata),
    }


def write_checkpoint(checkpoint: IntegratorCheckpoint, path: str | Path) -> Path:
    """Atomically write a JSON descriptor and paired compressed NPZ state."""
    json_path = Path(path).resolve()
    if json_path.suffix.lower() != ".json":
        raise ValueError("checkpoint descriptor path must end in .json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = json_path.with_suffix(".npz")
    token = uuid4().hex
    temporary_json = json_path.with_name(f".{json_path.name}.{token}.tmp")
    temporary_npz = npz_path.with_name(f".{npz_path.name}.{token}.tmp")
    state = cast(FloatArray, checkpoint.state)
    try:
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(stream, state=np.asarray(state, dtype="<f8"))
        content = _content_mapping(checkpoint)
        descriptor = {
            **content,
            "state_file": npz_path.name,
            "state_file_sha256": _file_sha256(temporary_npz),
            "content_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest(),
        }
        temporary_json.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_npz.replace(npz_path)
        temporary_json.replace(json_path)
    finally:
        temporary_json.unlink(missing_ok=True)
        temporary_npz.unlink(missing_ok=True)
    return json_path


def load_checkpoint(path: str | Path) -> IntegratorCheckpoint:
    """Load a checkpoint only after checking descriptor, payload, and content hashes."""
    json_path = Path(path).resolve()
    try:
        raw: object = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(f"cannot read checkpoint descriptor: {error}") from error
    if not isinstance(raw, dict):
        raise CheckpointIntegrityError("checkpoint descriptor must be a JSON object")
    descriptor = cast(dict[str, Any], raw)
    expected_keys = {
        "schema_version",
        "epoch",
        "time_s",
        "next_step_s",
        "state_shape",
        "state_dtype",
        "state_sha256",
        "metadata",
        "state_file",
        "state_file_sha256",
        "content_sha256",
    }
    if set(descriptor) != expected_keys:
        raise CheckpointIntegrityError("checkpoint descriptor fields do not match schema")
    state_file = descriptor["state_file"]
    if not isinstance(state_file, str) or Path(state_file).name != state_file:
        raise CheckpointIntegrityError("checkpoint state_file must be a local filename")
    npz_path = json_path.with_name(state_file)
    if not npz_path.is_file() or _file_sha256(npz_path) != descriptor["state_file_sha256"]:
        raise CheckpointIntegrityError("checkpoint state payload hash mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            if payload.files != ["state"]:
                raise CheckpointIntegrityError("checkpoint payload must contain only state")
            state = np.asarray(payload["state"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise CheckpointIntegrityError(f"cannot load checkpoint state: {error}") from error
    if list(state.shape) != descriptor["state_shape"] or descriptor["state_dtype"] != "float64":
        raise CheckpointIntegrityError("checkpoint state shape or dtype mismatch")
    if _state_sha256(state) != descriptor["state_sha256"]:
        raise CheckpointIntegrityError("checkpoint state content hash mismatch")

    content = {key: descriptor[key] for key in _content_mapping_keys()}
    actual_content_sha256 = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    if actual_content_sha256 != descriptor["content_sha256"]:
        raise CheckpointIntegrityError("checkpoint descriptor content hash mismatch")
    try:
        return IntegratorCheckpoint(
            schema_version=cast(str, descriptor["schema_version"]),
            epoch=cast(str, descriptor["epoch"]),
            time_s=float(descriptor["time_s"]),
            state=state,
            next_step_s=float(descriptor["next_step_s"]),
            metadata=cast(dict[str, Any], descriptor["metadata"]),
        )
    except (TypeError, ValueError) as error:
        raise CheckpointIntegrityError(f"invalid checkpoint values: {error}") from error


def _content_mapping_keys() -> tuple[str, ...]:
    return (
        "schema_version",
        "epoch",
        "time_s",
        "next_step_s",
        "state_shape",
        "state_dtype",
        "state_sha256",
        "metadata",
    )
