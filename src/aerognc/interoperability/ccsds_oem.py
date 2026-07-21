"""Minimal deterministic CCSDS 502.0-B-3 OEM/KVN engineering export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class OemMetadata:
    """Mandatory OEM header/segment metadata supplied explicitly by the caller."""

    creation_date: datetime
    originator: str
    message_id: str
    object_name: str
    object_id: str
    center_name: str
    reference_frame: str
    time_system: str
    start_epoch: datetime
    interpolation_degree: int = 7

    def __post_init__(self) -> None:
        text_values = (
            self.originator,
            self.message_id,
            self.object_name,
            self.object_id,
            self.center_name,
            self.reference_frame,
            self.time_system,
        )
        if not all(value.strip() for value in text_values):
            raise ValueError("OEM text metadata cannot be empty")
        if self.interpolation_degree < 1:
            raise ValueError("OEM interpolation degree must be positive")


def _epoch_text(epoch: datetime) -> str:
    return epoch.replace(tzinfo=None).isoformat(timespec="microseconds")


def write_oem_kvn(
    elapsed_time_s: npt.ArrayLike,
    states_si: npt.ArrayLike,
    metadata: OemMetadata,
    path: str | Path,
) -> Path:
    """Write position/velocity samples as OEM 3.0 KVN in mandated km and km/s.

    AeroGNC remains SI internally. The conversion at this file boundary is explicit
    because the OEM data-line units are kilometres and kilometres per second.
    """
    time_s = np.asarray(elapsed_time_s, dtype=np.float64)
    states = np.asarray(states_si, dtype=np.float64)
    if time_s.ndim != 1 or states.shape != (time_s.size, 6) or time_s.size < 2:
        raise ValueError("OEM time/state arrays must have shapes (N,) and (N,6), N>=2")
    if (
        not np.all(np.isfinite(time_s))
        or not np.all(np.diff(time_s) > 0.0)
        or not np.all(np.isfinite(states))
    ):
        raise ValueError("OEM time/state values must be finite and time strictly increasing")
    relative_time_s = time_s - time_s[0]
    epochs = [metadata.start_epoch + timedelta(seconds=float(value)) for value in relative_time_s]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "CCSDS_OEM_VERS = 3.0",
        "COMMENT Fictional civilian synthetic trajectory; non-operational engineering export",
        f"CREATION_DATE = {_epoch_text(metadata.creation_date)}",
        f"ORIGINATOR = {metadata.originator}",
        f"MESSAGE_ID = {metadata.message_id}",
        "",
        "META_START",
        f"OBJECT_NAME = {metadata.object_name}",
        f"OBJECT_ID = {metadata.object_id}",
        f"CENTER_NAME = {metadata.center_name}",
        f"REF_FRAME = {metadata.reference_frame}",
        f"TIME_SYSTEM = {metadata.time_system}",
        f"START_TIME = {_epoch_text(epochs[0])}",
        f"STOP_TIME = {_epoch_text(epochs[-1])}",
        "INTERPOLATION = LAGRANGE",
        f"INTERPOLATION_DEGREE = {metadata.interpolation_degree}",
        "META_STOP",
        "",
        "COMMENT Epoch x y z vx vy vz; OEM distance units are km and km/s",
    ]
    states_km = states / 1_000.0
    for epoch, state in zip(epochs, states_km, strict=True):
        lines.append(f"{_epoch_text(epoch)} " + " ".join(f"{float(value):.15e}" for value in state))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_oem_kvn(path: str | Path) -> tuple[dict[str, str], FloatArray, FloatArray]:
    """Parse the deterministic AeroGNC OEM subset and return SI state values."""
    metadata: dict[str, str] = {}
    epochs: list[float] = []
    states: list[list[float]] = []
    first_epoch: datetime | None = None
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("COMMENT") or line in {"META_START", "META_STOP"}:
            continue
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", maxsplit=1))
            metadata[key] = value
            continue
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"invalid OEM data row at line {line_number}")
        try:
            epoch = datetime.fromisoformat(fields[0])
            state_km = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"invalid OEM value at line {line_number}") from error
        if first_epoch is None:
            first_epoch = epoch
        epochs.append((epoch - first_epoch).total_seconds())
        states.append([1_000.0 * value for value in state_km])
    if metadata.get("CCSDS_OEM_VERS") != "3.0" or len(states) < 2:
        raise ValueError("OEM must contain version 3.0 metadata and at least two states")
    return (
        metadata,
        np.asarray(epochs, dtype=np.float64),
        np.asarray(states, dtype=np.float64),
    )
