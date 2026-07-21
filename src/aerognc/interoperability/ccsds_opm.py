"""Scoped CCSDS OPM 3.0 KVN Cartesian-state exchange for fictional records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
class OpmMetadata:
    """Mandatory header and object metadata for the supported OPM subset."""

    creation_date: datetime
    originator: str
    message_id: str
    object_name: str
    object_id: str
    center_name: str
    reference_frame: str
    time_system: str
    fictional: bool = True

    def __post_init__(self) -> None:
        require_text(
            originator=self.originator,
            message_id=self.message_id,
            object_name=self.object_name,
            object_id=self.object_id,
            center_name=self.center_name,
            reference_frame=self.reference_frame,
        )
        validate_time_system(self.time_system)
        if not self.fictional:
            raise ValueError("AeroGNC public OPM records must be marked fictional")


@dataclass(frozen=True, slots=True)
class OpmState:
    """One Cartesian SI orbit state and positive spacecraft mass."""

    epoch: datetime
    state_si: FloatArray
    mass_kg: float

    def __init__(self, epoch: datetime, state_si: npt.ArrayLike, mass_kg: float) -> None:
        state = np.asarray(state_si, dtype=np.float64)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("OPM Cartesian state must contain six finite SI values")
        if not np.isfinite(mass_kg) or mass_kg <= 0.0:
            raise ValueError("OPM mass_kg must be positive and finite")
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "state_si", state.copy())
        object.__setattr__(self, "mass_kg", float(mass_kg))


def write_opm_kvn(record: OpmState, metadata: OpmMetadata, path: str | Path) -> Path:
    """Write one Cartesian OPM state, converting SI metres to mandated kilometres."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_km = record.state_si / 1_000.0
    keys = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
    lines = [
        "CCSDS_OPM_VERS = 3.0",
        "COMMENT Fictional civilian synthetic orbit; scoped Cartesian KVN subset",
        f"CREATION_DATE = {epoch_text(metadata.creation_date)}",
        f"ORIGINATOR = {metadata.originator}",
        f"MESSAGE_ID = {metadata.message_id}",
        "",
        "META_START",
        f"OBJECT_NAME = {metadata.object_name}",
        f"OBJECT_ID = {metadata.object_id}",
        f"CENTER_NAME = {metadata.center_name}",
        f"REF_FRAME = {metadata.reference_frame}",
        f"TIME_SYSTEM = {validate_time_system(metadata.time_system)}",
        "META_STOP",
        "",
        f"EPOCH = {epoch_text(record.epoch)}",
        "COMMENT Cartesian position [km], velocity [km/s], mass [kg]",
    ]
    lines.extend(f"{key} = {float(value):.17e}" for key, value in zip(keys, state_km, strict=True))
    lines.extend((f"MASS = {record.mass_kg:.17e}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def parse_opm_kvn(path: str | Path) -> tuple[dict[str, str], OpmState]:
    """Parse and validate the supported Cartesian OPM subset into SI units."""
    fields: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("COMMENT") or line in {"META_START", "META_STOP"}:
            continue
        if "=" not in line:
            raise ValueError(f"invalid OPM KVN line {line_number}")
        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        if key in fields:
            raise ValueError(f"duplicate OPM field {key!r}")
        fields[key] = value
    state_keys = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
    mandatory = {
        "CCSDS_OPM_VERS",
        "CREATION_DATE",
        "ORIGINATOR",
        "MESSAGE_ID",
        "OBJECT_NAME",
        "OBJECT_ID",
        "CENTER_NAME",
        "REF_FRAME",
        "TIME_SYSTEM",
        "EPOCH",
        "MASS",
        *state_keys,
    }
    missing = mandatory - fields.keys()
    if missing:
        raise ValueError(f"OPM missing mandatory metadata/data: {sorted(missing)}")
    if fields["CCSDS_OPM_VERS"] != "3.0":
        raise ValueError("OPM version must be 3.0")
    validate_time_system(fields["TIME_SYSTEM"])
    try:
        state_si = 1_000.0 * np.array([float(fields[key]) for key in state_keys])
        mass_kg = float(fields["MASS"])
    except ValueError as error:
        raise ValueError("OPM Cartesian state and mass must be numeric") from error
    record = OpmState(parse_epoch(fields["EPOCH"], context="OPM"), state_si, mass_kg)
    return fields, record
