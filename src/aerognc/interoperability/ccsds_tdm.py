"""Scoped CCSDS TDM 2.0 KVN tracking-data exchange for fictional records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import numpy as np

from aerognc.interoperability.ccsds_common import (
    epoch_text,
    parse_epoch,
    require_text,
    validate_time_system,
)

Observable = Literal["RANGE", "DOPPLER_INSTANTANEOUS", "ANGLE_1", "ANGLE_2"]
_OBSERVABLES = {"RANGE", "DOPPLER_INSTANTANEOUS", "ANGLE_1", "ANGLE_2"}


@dataclass(frozen=True, slots=True)
class TdmMetadata:
    """Mandatory sequential two-participant metadata for the supported subset."""

    creation_date: datetime
    originator: str
    message_id: str
    participant_1: str
    participant_2: str
    time_system: str
    start_epoch: datetime
    mode: str = "SEQUENTIAL"
    path: str = "1,2"
    fictional: bool = True

    def __post_init__(self) -> None:
        require_text(
            originator=self.originator,
            message_id=self.message_id,
            participant_1=self.participant_1,
            participant_2=self.participant_2,
            mode=self.mode,
            path=self.path,
        )
        validate_time_system(self.time_system)
        if self.mode != "SEQUENTIAL":
            raise ValueError("TDM subset supports SEQUENTIAL mode only")
        if not self.fictional:
            raise ValueError("AeroGNC public TDM records must be marked fictional")


@dataclass(frozen=True, slots=True)
class TdmObservation:
    """One elapsed-time observation in AeroGNC SI/radian/Hz units."""

    time_s: float
    observable: Observable
    value: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.time_s, self.value])):
            raise ValueError("TDM observation time and value must be finite")
        if self.observable not in _OBSERVABLES:
            raise ValueError(f"unsupported TDM observable {self.observable!r}")


def _to_kvn_value(observation: TdmObservation) -> float:
    if observation.observable == "RANGE":
        return observation.value / 1_000.0
    if observation.observable in {"ANGLE_1", "ANGLE_2"}:
        return float(np.rad2deg(observation.value))
    return observation.value


def _from_kvn_value(observable: Observable, value: float) -> float:
    if observable == "RANGE":
        return 1_000.0 * value
    if observable in {"ANGLE_1", "ANGLE_2"}:
        return float(np.deg2rad(value))
    return value


def _validate_observations(observations: tuple[TdmObservation, ...]) -> None:
    if len(observations) < 2:
        raise ValueError("TDM requires at least two observations")
    time_s = np.array([item.time_s for item in observations])
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("TDM observation epochs must be strictly increasing")


def write_tdm_kvn(
    observations: tuple[TdmObservation, ...], metadata: TdmMetadata, path: str | Path
) -> Path:
    """Write range [km], angles [deg], and instantaneous Doppler [Hz] observations."""
    _validate_observations(observations)
    epochs = [metadata.start_epoch + timedelta(seconds=item.time_s) for item in observations]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "CCSDS_TDM_VERS = 2.0",
        "COMMENT Fictional civilian synthetic tracking; scoped KVN subset",
        f"CREATION_DATE = {epoch_text(metadata.creation_date)}",
        f"ORIGINATOR = {metadata.originator}",
        f"MESSAGE_ID = {metadata.message_id}",
        "",
        "META_START",
        f"TIME_SYSTEM = {validate_time_system(metadata.time_system)}",
        f"START_TIME = {epoch_text(epochs[0])}",
        f"STOP_TIME = {epoch_text(epochs[-1])}",
        f"PARTICIPANT_1 = {metadata.participant_1}",
        f"PARTICIPANT_2 = {metadata.participant_2}",
        f"MODE = {metadata.mode}",
        f"PATH = {metadata.path}",
        "META_STOP",
        "",
        "DATA_START",
        "COMMENT RANGE [km], ANGLE_1/2 [deg], DOPPLER_INSTANTANEOUS [Hz]",
    ]
    lines.extend(
        f"{item.observable} = {epoch_text(epoch)} {_to_kvn_value(item):.17e}"
        for item, epoch in zip(observations, epochs, strict=True)
    )
    lines.extend(("DATA_STOP", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def parse_tdm_kvn(path: str | Path) -> tuple[dict[str, str], tuple[TdmObservation, ...]]:
    """Parse the supported TDM subset and restore SI/radian/Hz values."""
    metadata: dict[str, str] = {}
    raw_observations: list[tuple[datetime, Observable, float]] = []
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
        if "=" not in line:
            raise ValueError(f"invalid TDM KVN line {line_number}")
        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        if in_data:
            if key not in _OBSERVABLES:
                raise ValueError(f"unsupported TDM observable {key!r} at line {line_number}")
            fields = value.split()
            if len(fields) != 2:
                raise ValueError(f"invalid TDM observation at line {line_number}")
            try:
                numeric_value = float(fields[1])
            except ValueError as error:
                raise ValueError(f"invalid TDM value at line {line_number}") from error
            observable = cast(Observable, key)
            raw_observations.append(
                (
                    parse_epoch(fields[0], context="TDM data"),
                    observable,
                    numeric_value,
                )
            )
        else:
            if key in metadata:
                raise ValueError(f"duplicate TDM metadata field {key!r}")
            metadata[key] = value
    mandatory = {
        "CCSDS_TDM_VERS",
        "CREATION_DATE",
        "ORIGINATOR",
        "MESSAGE_ID",
        "TIME_SYSTEM",
        "START_TIME",
        "STOP_TIME",
        "PARTICIPANT_1",
        "PARTICIPANT_2",
        "MODE",
        "PATH",
    }
    missing = mandatory - metadata.keys()
    if missing:
        raise ValueError(f"TDM missing mandatory metadata: {sorted(missing)}")
    if metadata["CCSDS_TDM_VERS"] != "2.0" or metadata["MODE"] != "SEQUENTIAL":
        raise ValueError("TDM parser requires version 2.0 SEQUENTIAL mode")
    validate_time_system(metadata["TIME_SYSTEM"])
    if len(raw_observations) < 2:
        raise ValueError("TDM requires at least two observations")
    first_epoch = raw_observations[0][0]
    last_epoch = raw_observations[-1][0]
    if (
        parse_epoch(metadata["START_TIME"], context="TDM START_TIME") != first_epoch
        or parse_epoch(metadata["STOP_TIME"], context="TDM STOP_TIME") != last_epoch
    ):
        raise ValueError("TDM START_TIME/STOP_TIME do not match observation coverage")
    observations = tuple(
        TdmObservation(
            (epoch - first_epoch).total_seconds(),
            observable,
            _from_kvn_value(observable, value),
        )
        for epoch, observable, value in raw_observations
    )
    _validate_observations(observations)
    return metadata, observations
