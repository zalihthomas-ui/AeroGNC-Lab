"""Scoped CCSDS AEM 2.0 KVN quaternion exchange for fictional records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerognc.interoperability.ccsds_common import (
    epoch_text,
    parse_epoch,
    require_text,
    validate_time_system,
)
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class AemMetadata:
    """Mandatory metadata for the supported scalar-first quaternion segment."""

    creation_date: datetime
    originator: str
    message_id: str
    object_name: str
    object_id: str
    reference_frame_a: str
    reference_frame_b: str
    time_system: str
    start_epoch: datetime
    attitude_direction: str = "A2B"
    fictional: bool = True

    def __post_init__(self) -> None:
        require_text(
            originator=self.originator,
            message_id=self.message_id,
            object_name=self.object_name,
            object_id=self.object_id,
            reference_frame_a=self.reference_frame_a,
            reference_frame_b=self.reference_frame_b,
        )
        validate_time_system(self.time_system)
        if self.attitude_direction not in {"A2B", "B2A"}:
            raise ValueError("AEM attitude_direction must be A2B or B2A")
        if not self.fictional:
            raise ValueError("AeroGNC public AEM records must be marked fictional")


def _validated_history(
    elapsed_time_s: npt.ArrayLike, quaternion_scalar_first: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    time_s = np.asarray(elapsed_time_s, dtype=np.float64)
    quaternion = np.asarray(quaternion_scalar_first, dtype=np.float64)
    if time_s.ndim != 1 or time_s.size < 2 or quaternion.shape != (time_s.size, 4):
        raise ValueError("AEM time/quaternion arrays must have shapes (N,) and (N,4), N>=2")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("AEM epochs must be finite and strictly increasing")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("AEM quaternion samples must be finite")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(np.abs(norms - 1.0) > 1.0e-9):
        raise ValueError("AEM quaternion samples must have unit norm within 1e-9")
    return time_s.copy(), quaternion.copy()


def write_aem_kvn(
    elapsed_time_s: npt.ArrayLike,
    quaternion_scalar_first: npt.ArrayLike,
    metadata: AemMetadata,
    path: str | Path,
) -> Path:
    """Write the supported AEM quaternion-only subset without claiming full conformance."""
    time_s, quaternion = _validated_history(elapsed_time_s, quaternion_scalar_first)
    epochs = [
        metadata.start_epoch + timedelta(seconds=float(value - time_s[0])) for value in time_s
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "CCSDS_AEM_VERS = 2.0",
        "COMMENT Fictional civilian synthetic attitude; scoped quaternion-only KVN subset",
        f"CREATION_DATE = {epoch_text(metadata.creation_date)}",
        f"ORIGINATOR = {metadata.originator}",
        f"MESSAGE_ID = {metadata.message_id}",
        "",
        "META_START",
        f"OBJECT_NAME = {metadata.object_name}",
        f"OBJECT_ID = {metadata.object_id}",
        f"REF_FRAME_A = {metadata.reference_frame_a}",
        f"REF_FRAME_B = {metadata.reference_frame_b}",
        f"ATTITUDE_DIR = {metadata.attitude_direction}",
        f"TIME_SYSTEM = {validate_time_system(metadata.time_system)}",
        f"START_TIME = {epoch_text(epochs[0])}",
        f"STOP_TIME = {epoch_text(epochs[-1])}",
        "ATTITUDE_TYPE = QUATERNION",
        "QUATERNION_TYPE = FIRST",
        "META_STOP",
        "",
        "DATA_START",
        "COMMENT Epoch q0 q1 q2 q3; Hamilton scalar-first unit quaternion",
    ]
    lines.extend(
        f"{epoch_text(epoch)} " + " ".join(f"{float(value):.17e}" for value in sample)
        for epoch, sample in zip(epochs, quaternion, strict=True)
    )
    lines.extend(("DATA_STOP", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def parse_aem_kvn(path: str | Path) -> tuple[dict[str, str], FloatArray, FloatArray]:
    """Parse and validate the AeroGNC quaternion-only AEM subset."""
    metadata: dict[str, str] = {}
    epochs: list[datetime] = []
    quaternions: list[list[float]] = []
    in_data = False
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("COMMENT") or line in {"META_START", "META_STOP"}:
            continue
        if line == "DATA_START":
            in_data = True
            continue
        if line == "DATA_STOP":
            in_data = False
            continue
        if in_data:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid AEM data row at line {line_number}")
            epoch = parse_epoch(fields[0], context="AEM data")
            try:
                quaternion_sample = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(f"invalid AEM quaternion at line {line_number}") from error
            epochs.append(epoch)
            quaternions.append(quaternion_sample)
        elif "=" in line:
            key, value = (part.strip() for part in line.split("=", maxsplit=1))
            if key in metadata:
                raise ValueError(f"duplicate AEM metadata field {key!r}")
            metadata[key] = value
        else:
            raise ValueError(f"invalid AEM KVN line {line_number}")
    mandatory = {
        "CCSDS_AEM_VERS",
        "CREATION_DATE",
        "ORIGINATOR",
        "MESSAGE_ID",
        "OBJECT_NAME",
        "OBJECT_ID",
        "REF_FRAME_A",
        "REF_FRAME_B",
        "ATTITUDE_DIR",
        "TIME_SYSTEM",
        "START_TIME",
        "STOP_TIME",
        "ATTITUDE_TYPE",
        "QUATERNION_TYPE",
    }
    missing = mandatory - metadata.keys()
    if missing:
        raise ValueError(f"AEM missing mandatory metadata: {sorted(missing)}")
    if metadata["CCSDS_AEM_VERS"] != "2.0":
        raise ValueError("AEM version must be 2.0")
    if metadata["ATTITUDE_TYPE"] != "QUATERNION" or metadata["QUATERNION_TYPE"] != "FIRST":
        raise ValueError("AEM parser supports scalar-first quaternion attitude only")
    if metadata["ATTITUDE_DIR"] not in {"A2B", "B2A"}:
        raise ValueError("AEM ATTITUDE_DIR must be A2B or B2A")
    validate_time_system(metadata["TIME_SYSTEM"])
    if len(epochs) < 2:
        raise ValueError("AEM requires at least two attitude samples")
    elapsed = np.array([(epoch - epochs[0]).total_seconds() for epoch in epochs])
    time_s, quaternion_array = _validated_history(elapsed, np.asarray(quaternions))
    if (
        parse_epoch(metadata["START_TIME"], context="AEM START_TIME") != epochs[0]
        or parse_epoch(metadata["STOP_TIME"], context="AEM STOP_TIME") != epochs[-1]
    ):
        raise ValueError("AEM START_TIME/STOP_TIME do not match data coverage")
    return metadata, time_s, quaternion_array
