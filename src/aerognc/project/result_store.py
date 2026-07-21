"""Atomic, indexed, unit-aware local storage for engineering runs."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray
from aerognc.project.manifest import (
    ArtifactRecord,
    RunManifest,
    file_sha256,
    load_manifest,
    with_artifacts,
    write_manifest,
)
from aerognc.simulation.logging import SimulationResult

DATASET_SCHEMA_VERSION = "1.0"
_CHANNEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,95}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_EXACT_UNITS = {
    "mach": "1",
    "quaternion_norm": "1",
    "quaternion_q0": "1",
    "quaternion_q1": "1",
    "quaternion_q2": "1",
    "quaternion_q3": "1",
    "phase_code": "1",
}
_SUFFIX_UNITS = (
    ("_mps2", "m/s^2"),
    ("_degps", "deg/s"),
    ("_radps", "rad/s"),
    ("_kmps", "km/s"),
    ("_mps", "m/s"),
    ("_kgps", "kg/s"),
    ("_pa", "Pa"),
    ("_kg", "kg"),
    ("_nm", "N m"),
    ("_kn", "kN"),
    ("_n", "N"),
    ("_m2", "m^2"),
    ("_deg", "deg"),
    ("_rad", "rad"),
    ("_s", "s"),
    ("_m", "m"),
)


class ResultIntegrityError(RuntimeError):
    """Raised when a stored run is incomplete or fails its recorded hashes."""


def infer_channel_unit(name: str) -> str:
    """Infer an explicit unit from a stable unit-bearing channel name."""
    if name in _EXACT_UNITS:
        return _EXACT_UNITS[name]
    for suffix, unit in _SUFFIX_UNITS:
        if name.endswith(suffix):
            return unit
    if name.startswith("quaternion_"):
        return "1"
    raise ValueError(f"cannot infer unit for channel {name!r}; provide an explicit unit")


def _immutable_array(value: npt.ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _freeze_nested_mapping(
    value: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType(
        {name: MappingProxyType(dict(record)) for name, record in value.items()}
    )


@dataclass(frozen=True, slots=True)
class ResultDataset:
    """One validated common-time-base engineering trajectory."""

    scenario_name: str
    time_s: FloatArray
    channels: Mapping[str, FloatArray]
    units: Mapping[str, str]
    events: tuple[Mapping[str, Any], ...] = ()
    maxima: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.scenario_name:
            raise ValueError("dataset scenario_name cannot be empty")
        time_s = _immutable_array(self.time_s)
        if time_s.ndim != 1 or time_s.size < 2:
            raise ValueError("dataset time_s must be a one-dimensional trajectory")
        if not np.all(np.isfinite(time_s)) or not np.all(np.diff(time_s) > 0.0):
            raise ValueError("dataset time_s must be finite and strictly increasing")
        if not self.channels:
            raise ValueError("dataset must contain at least one channel")
        if self.channels.keys() != self.units.keys():
            raise ValueError("dataset units must exactly match channel names")
        frozen_channels: dict[str, FloatArray] = {}
        for name, raw_values in self.channels.items():
            if not _CHANNEL_PATTERN.fullmatch(name):
                raise ValueError(f"invalid channel name: {name!r}")
            values = _immutable_array(raw_values)
            if values.shape != time_s.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"channel {name!r} must be finite and match time_s")
            if not self.units[name]:
                raise ValueError(f"channel {name!r} has an empty unit")
            frozen_channels[name] = values
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "channels", MappingProxyType(frozen_channels))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        object.__setattr__(
            self,
            "events",
            tuple(MappingProxyType(dict(event)) for event in self.events),
        )
        object.__setattr__(self, "maxima", _freeze_nested_mapping(self.maxima))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_simulation_result(
        cls,
        result: SimulationResult,
        *,
        units: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ResultDataset:
        """Adapt the common atmospheric simulation result without losing units."""
        resolved_units = (
            {name: infer_channel_unit(name) for name in result.columns}
            if units is None
            else dict(units)
        )
        return cls(
            scenario_name=result.scenario_name,
            time_s=result.time_s,
            channels=result.columns,
            units=resolved_units,
            events=tuple(dict(record) for record in result.event_summary),
            maxima=result.maximum_summary,
            metadata={"source": "SimulationResult", **({} if metadata is None else metadata)},
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Compact index record used by CLI and desktop run browsers."""

    run_id: str
    created_utc: str
    project_name: str
    scenario_name: str
    workflow: str
    status: str
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class StoredRun:
    """A verified manifest and optional reloaded trajectory."""

    directory: Path
    manifest: RunManifest
    dataset: ResultDataset | None


def _json_value(value: object) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _dataset_summary(dataset: ResultDataset) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "scenario_name": dataset.scenario_name,
        "sample_count": int(dataset.time_s.size),
        "channel_order": list(dataset.channels),
        "units": dict(dataset.units),
        "events": _json_value(dataset.events),
        "maxima": _json_value(dataset.maxima),
        "metadata": _json_value(dataset.metadata),
    }


def _write_dataset(dataset: ResultDataset, directory: Path) -> tuple[Path, Path, Path]:
    npz_path = directory / "trajectory.npz"
    arrays: list[FloatArray] = [dataset.time_s]
    channel_storage: dict[str, str] = {}
    for index, (name, values) in enumerate(dataset.channels.items()):
        storage_name = f"arr_{index + 1}"
        arrays.append(values)
        channel_storage[name] = storage_name
    np.savez_compressed(npz_path, *arrays)

    csv_path = directory / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", *dataset.channels])
        for index, time_s in enumerate(dataset.time_s):
            writer.writerow(
                [
                    f"{time_s:.15g}",
                    *(f"{dataset.channels[name][index]:.15g}" for name in dataset.channels),
                ]
            )

    summary = _dataset_summary(dataset)
    summary["channel_storage"] = channel_storage
    summary_path = directory / "dataset.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return npz_path, csv_path, summary_path


def _load_dataset(directory: Path) -> ResultDataset:
    summary_path = directory / "dataset.json"
    npz_path = directory / "trajectory.npz"
    if not summary_path.is_file() or not npz_path.is_file():
        raise ResultIntegrityError("completed run is missing dataset files")
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ResultIntegrityError("unsupported or malformed dataset metadata")
    channel_order = raw.get("channel_order")
    units = raw.get("units")
    storage = raw.get("channel_storage")
    if (
        not isinstance(channel_order, list)
        or not all(isinstance(name, str) for name in channel_order)
        or not isinstance(units, dict)
        or not isinstance(storage, dict)
    ):
        raise ResultIntegrityError("dataset channel metadata is malformed")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            time_s = np.asarray(archive["arr_0"], dtype=np.float64)
            channels = {
                name: np.asarray(archive[str(storage[name])], dtype=np.float64)
                for name in cast(list[str], channel_order)
            }
    except (KeyError, OSError, ValueError) as error:
        raise ResultIntegrityError(f"cannot load trajectory archive: {error}") from error
    events = raw.get("events", [])
    maxima = raw.get("maxima", {})
    metadata = raw.get("metadata", {})
    if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
        raise ResultIntegrityError("dataset events are malformed")
    if not isinstance(maxima, dict) or not all(isinstance(item, dict) for item in maxima.values()):
        raise ResultIntegrityError("dataset maxima are malformed")
    if not isinstance(metadata, dict):
        raise ResultIntegrityError("dataset metadata is malformed")
    return ResultDataset(
        scenario_name=str(raw.get("scenario_name", "")),
        time_s=time_s,
        channels=channels,
        units={str(name): str(unit) for name, unit in units.items()},
        events=tuple(cast(list[dict[str, Any]], events)),
        maxima=cast(dict[str, dict[str, Any]], maxima),
        metadata=cast(dict[str, Any], metadata),
    )


def _artifact(path: Path, role: str, media_type: str, base: Path) -> ArtifactRecord:
    return ArtifactRecord(
        role=role,
        relative_path=path.relative_to(base).as_posix(),
        sha256=file_sha256(path),
        media_type=media_type,
        size_bytes=path.stat().st_size,
    )


class ResultStore:
    """Local immutable run directories with a rebuildable SQLite index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.index_path = self.root / "index.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.index_path)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT NOT NULL,
                project_name TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                workflow TEXT NOT NULL,
                status TEXT NOT NULL,
                input_fingerprint TEXT NOT NULL,
                manifest_path TEXT NOT NULL
            )
            """
        )
        return connection

    def _run_directory(self, run_id: str) -> Path:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id contains unsupported characters")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise ValueError("run_id escapes result store")
        return path

    def commit(self, manifest: RunManifest, dataset: ResultDataset | None) -> StoredRun:
        """Validate and atomically commit a terminal run."""
        if manifest.status == "completed" and dataset is None:
            raise ValueError("completed run requires a dataset")
        if manifest.status != "completed" and dataset is not None:
            raise ValueError("failed/cancelled run cannot carry a completed dataset")
        final_directory = self._run_directory(manifest.run_id)
        if final_directory.exists():
            raise FileExistsError(f"run already exists: {manifest.run_id}")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".partial-{manifest.run_id}-{os.getpid()}"
        if temporary.exists():
            raise FileExistsError(f"partial run already exists: {temporary.name}")
        temporary.mkdir()
        try:
            artifacts: list[ArtifactRecord] = []
            if dataset is not None:
                npz_path, csv_path, summary_path = _write_dataset(dataset, temporary)
                artifacts.extend(
                    (
                        _artifact(npz_path, "trajectory-npz", "application/x-npz", temporary),
                        _artifact(csv_path, "trajectory-csv", "text/csv", temporary),
                        _artifact(summary_path, "dataset-metadata", "application/json", temporary),
                    )
                )
            committed_manifest = with_artifacts(manifest, artifacts)
            write_manifest(committed_manifest, temporary / "manifest.json")
            loaded_manifest = load_manifest(temporary / "manifest.json")
            loaded_dataset = _load_dataset(temporary) if dataset is not None else None
            if loaded_manifest != committed_manifest:
                raise ResultIntegrityError("manifest round-trip changed values")
            temporary.replace(final_directory)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        relative_manifest = (final_directory / "manifest.json").relative_to(self.root).as_posix()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, created_utc, project_name, scenario_name, workflow,
                        status, input_fingerprint, manifest_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        committed_manifest.run_id,
                        committed_manifest.created_utc,
                        committed_manifest.project_name,
                        committed_manifest.scenario_name,
                        committed_manifest.workflow,
                        committed_manifest.status,
                        committed_manifest.input_fingerprint,
                        relative_manifest,
                    ),
                )
        except sqlite3.Error as error:
            raise ResultIntegrityError(
                "run files were committed but index update failed; use rebuild_index()"
            ) from error
        return StoredRun(final_directory, committed_manifest, loaded_dataset)

    def _verify_artifacts(self, directory: Path, manifest: RunManifest) -> None:
        for artifact in manifest.artifacts:
            path = (directory / artifact.relative_path).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ResultIntegrityError(f"missing run artifact: {artifact.relative_path}")
            if path.stat().st_size != artifact.size_bytes or file_sha256(path) != artifact.sha256:
                raise ResultIntegrityError(f"run artifact hash mismatch: {artifact.relative_path}")

    def load(self, run_id: str) -> StoredRun:
        """Load one run after validating every recorded artefact."""
        directory = self._run_directory(run_id)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise KeyError(f"run not found: {run_id}")
        manifest = load_manifest(manifest_path)
        if manifest.run_id != run_id:
            raise ResultIntegrityError("run directory and manifest identifier differ")
        self._verify_artifacts(directory, manifest)
        dataset = _load_dataset(directory) if manifest.status == "completed" else None
        return StoredRun(directory, manifest, dataset)

    def list_runs(
        self,
        *,
        project_name: str | None = None,
        scenario_name: str | None = None,
        status: str | None = None,
    ) -> tuple[RunRecord, ...]:
        """Query stable compact records newest first."""
        clauses: list[str] = []
        values: list[str] = []
        for column, value in (
            ("project_name", project_name),
            ("scenario_name", scenario_name),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        query = (
            "SELECT run_id, created_utc, project_name, scenario_name, workflow, status, "
            f"input_fingerprint FROM runs{where} ORDER BY created_utc DESC, run_id DESC"
        )
        with closing(self._connect()) as connection:
            rows = connection.execute(query, values).fetchall()
        return tuple(
            RunRecord(*cast(tuple[str, str, str, str, str, str, str], row)) for row in rows
        )

    def rebuild_index(self) -> int:
        """Rebuild the disposable index from immutable run manifests."""
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM runs")
            count = 0
            for manifest_path in sorted(self.root.glob("*/manifest.json")):
                manifest = load_manifest(manifest_path)
                relative = manifest_path.relative_to(self.root).as_posix()
                connection.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        manifest.run_id,
                        manifest.created_utc,
                        manifest.project_name,
                        manifest.scenario_name,
                        manifest.workflow,
                        manifest.status,
                        manifest.input_fingerprint,
                        relative,
                    ),
                )
                count += 1
        return count
