"""Strict versioned CSV telemetry mapping and SI normalisation boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import yaml

from aerognc.mathematics.vectors import FloatArray

MissingValuePolicy = Literal["error", "keep_nan", "drop_row"]


@dataclass(frozen=True, slots=True)
class TimestampMapping:
    """Source timestamp conversion into seconds on its declared local clock."""

    source_name: str
    source_unit: str
    scale_to_s: float
    offset_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_name.strip() or not self.source_unit.strip():
            raise ValueError("timestamp source name and unit cannot be empty")
        if not np.isfinite(self.scale_to_s) or self.scale_to_s <= 0.0:
            raise ValueError("timestamp scale_to_s must be positive and finite")
        if not np.isfinite(self.offset_s):
            raise ValueError("timestamp offset_s must be finite")


@dataclass(frozen=True, slots=True)
class QualityMapping:
    """Row-quality column and declared treatment of rejected values."""

    source_name: str
    accepted_values: tuple[str, ...]
    invalid_policy: MissingValuePolicy = "keep_nan"

    def __post_init__(self) -> None:
        if not self.source_name.strip() or not self.accepted_values:
            raise ValueError("quality mapping requires a source and accepted values")
        if any(not value.strip() for value in self.accepted_values):
            raise ValueError("accepted quality values cannot be empty")
        if len(set(self.accepted_values)) != len(self.accepted_values):
            raise ValueError("accepted quality values must be unique")
        _validate_missing_policy(self.invalid_policy, "quality invalid_policy")


@dataclass(frozen=True, slots=True)
class ChannelMapping:
    """One source-to-destination affine channel conversion and missing policy."""

    source_name: str
    destination_name: str
    source_unit: str
    destination_unit: str
    scale: float = 1.0
    offset: float = 0.0
    missing_policy: MissingValuePolicy = "error"

    def __post_init__(self) -> None:
        labels = (
            self.source_name,
            self.destination_name,
            self.source_unit,
            self.destination_unit,
        )
        if any(not value.strip() for value in labels):
            raise ValueError("telemetry channel names and units cannot be empty")
        if not np.all(np.isfinite([self.scale, self.offset])) or self.scale == 0.0:
            raise ValueError("telemetry scale must be nonzero and scale/offset finite")
        _validate_missing_policy(self.missing_policy, "channel missing_policy")


@dataclass(frozen=True, slots=True)
class TelemetryMapping:
    """Complete versioned mapping for one CSV source."""

    schema_version: str
    timestamp: TimestampMapping
    quality: QualityMapping
    channels: tuple[ChannelMapping, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported telemetry mapping schema_version; expected '1.0'")
        if not self.channels:
            raise ValueError("telemetry mapping requires at least one channel")
        source_names = tuple(channel.source_name for channel in self.channels)
        destination_names = tuple(channel.destination_name for channel in self.channels)
        if len(set(source_names)) != len(source_names):
            raise ValueError("telemetry source channel names must be unique")
        if len(set(destination_names)) != len(destination_names):
            raise ValueError("telemetry destination channel names must be unique")
        reserved = {self.timestamp.source_name, self.quality.source_name}
        if reserved & set(source_names):
            raise ValueError("timestamp/quality columns cannot also be mapped data channels")

    @property
    def sha256(self) -> str:
        """Return canonical SHA-256 of mapping semantics."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TelemetryProvenance:
    """Auditable source and mapping identity plus row accounting."""

    source_path: str
    source_sha256: str
    mapping_schema_version: str
    mapping_sha256: str
    rows_read: int
    rows_kept: int
    rows_dropped: int


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Normalised aligned-length channels on one local clock."""

    time_s: FloatArray
    channels: Mapping[str, FloatArray]
    units: Mapping[str, str]
    quality_valid: npt.NDArray[np.bool_]
    provenance: TelemetryProvenance

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=np.float64)
        quality = np.asarray(self.quality_valid, dtype=np.bool_)
        if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
            raise ValueError("normalised telemetry time_s must be a nonempty finite vector")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("normalised telemetry timestamps must be strictly increasing")
        if quality.shape != time.shape:
            raise ValueError("quality_valid must match telemetry time shape")
        channel_copy: dict[str, FloatArray] = {}
        for name, raw_values in self.channels.items():
            values = np.asarray(raw_values, dtype=np.float64)
            if not name.strip() or values.shape != time.shape:
                raise ValueError("every telemetry channel must match the time vector")
            values = values.copy()
            values.flags.writeable = False
            channel_copy[name] = values
        if not channel_copy or set(channel_copy) != set(self.units):
            raise ValueError("telemetry channel and unit keys must be matching and nonempty")
        if any(not unit.strip() for unit in self.units.values()):
            raise ValueError("telemetry destination units cannot be empty")
        time = time.copy()
        quality = quality.copy()
        time.flags.writeable = False
        quality.flags.writeable = False
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "quality_valid", quality)
        object.__setattr__(self, "channels", MappingProxyType(channel_copy))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))


def _validate_missing_policy(value: str, label: str) -> None:
    if value not in {"error", "keep_nan", "drop_row"}:
        raise ValueError(f"{label} must be error, keep_nan, or drop_row")


def _strict_keys(data: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing or unknown:
        raise ValueError(
            f"{label} keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_telemetry_mapping(path: str | Path) -> TelemetryMapping:
    """Load a strict schema-v1 YAML mapping."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream), "telemetry mapping")
    _strict_keys(root, {"schema_version", "timestamp", "quality", "channels"}, "mapping")
    timestamp = _mapping(root["timestamp"], "timestamp")
    _strict_keys(timestamp, {"source_name", "source_unit", "scale_to_s", "offset_s"}, "timestamp")
    quality = _mapping(root["quality"], "quality")
    _strict_keys(quality, {"source_name", "accepted_values", "invalid_policy"}, "quality")
    raw_accepted = quality["accepted_values"]
    if not isinstance(raw_accepted, Sequence) or isinstance(raw_accepted, str):
        raise ValueError("quality.accepted_values must be a sequence")
    raw_channels = root["channels"]
    if not isinstance(raw_channels, Sequence) or isinstance(raw_channels, str):
        raise ValueError("channels must be a sequence")
    channels: list[ChannelMapping] = []
    channel_keys = {
        "source_name",
        "destination_name",
        "source_unit",
        "destination_unit",
        "scale",
        "offset",
        "missing_policy",
    }
    for index, raw_channel in enumerate(raw_channels):
        channel = _mapping(raw_channel, f"channels[{index}]")
        _strict_keys(channel, channel_keys, f"channels[{index}]")
        channels.append(
            ChannelMapping(
                _text(channel["source_name"], f"channels[{index}].source_name"),
                _text(channel["destination_name"], f"channels[{index}].destination_name"),
                _text(channel["source_unit"], f"channels[{index}].source_unit"),
                _text(channel["destination_unit"], f"channels[{index}].destination_unit"),
                _number(channel["scale"], f"channels[{index}].scale"),
                _number(channel["offset"], f"channels[{index}].offset"),
                _text(channel["missing_policy"], f"channels[{index}].missing_policy"),  # type: ignore[arg-type]
            )
        )
    return TelemetryMapping(
        _text(root["schema_version"], "schema_version"),
        TimestampMapping(
            _text(timestamp["source_name"], "timestamp.source_name"),
            _text(timestamp["source_unit"], "timestamp.source_unit"),
            _number(timestamp["scale_to_s"], "timestamp.scale_to_s"),
            _number(timestamp["offset_s"], "timestamp.offset_s"),
        ),
        QualityMapping(
            _text(quality["source_name"], "quality.source_name"),
            tuple(_text(value, "quality.accepted_values item") for value in raw_accepted),
            _text(quality["invalid_policy"], "quality.invalid_policy"),  # type: ignore[arg-type]
        ),
        tuple(channels),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_telemetry_csv(path: str | Path, mapping: TelemetryMapping) -> TelemetryRecord:
    """Import, validate, quality-screen, and affine-normalise a CSV record."""
    source = Path(path)
    required_columns = {
        mapping.timestamp.source_name,
        mapping.quality.source_name,
        *(channel.source_name for channel in mapping.channels),
    }
    times: list[float] = []
    quality_values: list[bool] = []
    channel_values: dict[str, list[float]] = {
        channel.destination_name: [] for channel in mapping.channels
    }
    rows_read = 0
    rows_dropped = 0
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing_columns = sorted(required_columns - set(reader.fieldnames or ()))
            raise ValueError(f"telemetry CSV is missing required columns: {missing_columns}")
        for row_number, row in enumerate(reader, start=2):
            rows_read += 1
            try:
                raw_time = float(row[mapping.timestamp.source_name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {row_number}: timestamp is not numeric") from error
            time_s = raw_time * mapping.timestamp.scale_to_s + mapping.timestamp.offset_s
            if not np.isfinite(time_s):
                raise ValueError(f"row {row_number}: normalized timestamp is not finite")
            quality_valid = row[mapping.quality.source_name] in mapping.quality.accepted_values
            if not quality_valid and mapping.quality.invalid_policy == "error":
                raise ValueError(f"row {row_number}: rejected quality flag")
            if not quality_valid and mapping.quality.invalid_policy == "drop_row":
                rows_dropped += 1
                continue
            converted: dict[str, float] = {}
            if not quality_valid:
                converted.update({channel.destination_name: np.nan for channel in mapping.channels})
                times.append(time_s)
                quality_values.append(False)
                for name, value in converted.items():
                    channel_values[name].append(value)
                continue
            drop_row = False
            for channel in mapping.channels:
                raw_value = row[channel.source_name]
                missing = raw_value is None or raw_value.strip().casefold() in {"", "nan"}
                if missing:
                    if channel.missing_policy == "error":
                        raise ValueError(
                            f"row {row_number}: missing value in {channel.source_name!r}"
                        )
                    if channel.missing_policy == "drop_row":
                        drop_row = True
                        break
                    converted[channel.destination_name] = np.nan
                    continue
                try:
                    source_value = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"row {row_number}: {channel.source_name!r} is not numeric"
                    ) from error
                destination_value = source_value * channel.scale + channel.offset
                if not np.isfinite(destination_value):
                    raise ValueError(
                        f"row {row_number}: {channel.destination_name!r} converted nonfinite"
                    )
                converted[channel.destination_name] = destination_value
            if drop_row:
                rows_dropped += 1
                continue
            times.append(time_s)
            quality_values.append(quality_valid)
            for name, value in converted.items():
                channel_values[name].append(value)
    if not times:
        raise ValueError("telemetry import retained no rows")
    provenance = TelemetryProvenance(
        str(source.resolve()),
        _file_sha256(source),
        mapping.schema_version,
        mapping.sha256,
        rows_read,
        len(times),
        rows_dropped,
    )
    units = {channel.destination_name: channel.destination_unit for channel in mapping.channels}
    return TelemetryRecord(
        np.asarray(times),
        {name: np.asarray(values) for name, values in channel_values.items()},
        units,
        np.asarray(quality_values, dtype=np.bool_),
        provenance,
    )


def write_normalized_telemetry_csv(record: TelemetryRecord, path: str | Path) -> Path:
    """Write stable-column normalised data; units remain in the provenance sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(sorted(record.channels))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("time_s", *names, "quality_valid"))
        for index, time_s in enumerate(record.time_s):
            writer.writerow(
                (
                    f"{float(time_s):.12g}",
                    *(f"{float(record.channels[name][index]):.12g}" for name in names),
                    "1" if record.quality_valid[index] else "0",
                )
            )
    return destination


def write_telemetry_provenance(record: TelemetryRecord, path: str | Path) -> Path:
    """Write deterministic mapping/source evidence and destination units."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "provenance": asdict(record.provenance),
        "units": dict(sorted(record.units.items())),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
