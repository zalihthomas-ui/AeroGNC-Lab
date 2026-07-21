"""Validated deterministic simulation result storage and serialisation."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aerognc.mathematics.integrators import EventOccurrence
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Monotonic trajectory columns, events, summaries, and measured runtime."""

    scenario_name: str
    time_s: FloatArray
    columns: dict[str, FloatArray]
    events: tuple[EventOccurrence, ...]
    event_summary: tuple[dict[str, float | str], ...]
    maximum_summary: dict[str, dict[str, float | str]]
    execution_time_s: float

    def __post_init__(self) -> None:
        if not self.scenario_name:
            raise ValueError("scenario_name cannot be empty")
        if self.time_s.ndim != 1 or self.time_s.size < 2:
            raise ValueError("time_s must be a one-dimensional trajectory")
        if not np.all(np.isfinite(self.time_s)) or not np.all(np.diff(self.time_s) > 0.0):
            raise ValueError("time_s must be finite and strictly increasing")
        for name, values in self.columns.items():
            if not name or values.shape != self.time_s.shape:
                raise ValueError(f"column {name!r} does not match time_s")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"column {name!r} contains non-finite values")
        if self.execution_time_s < 0.0 or not np.isfinite(self.execution_time_s):
            raise ValueError("execution_time_s must be finite and nonnegative")

    def value_at_event(self, column: str, event_name: str) -> float:
        """Return a logged column interpolated at the named event time."""
        event = next((item for item in self.events if item.name == event_name), None)
        if event is None:
            raise KeyError(f"event not found: {event_name}")
        return float(np.interp(event.time_s, self.time_s, self.columns[column]))


def write_result_csv(result: SimulationResult, path: str | Path) -> Path:
    """Write deterministic CSV with stable unit-bearing column names."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time_s", *result.columns.keys()]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fieldnames)
        for index, time_s in enumerate(result.time_s):
            writer.writerow(
                [
                    f"{time_s:.10g}",
                    *(f"{result.columns[name][index]:.10g}" for name in result.columns),
                ]
            )
    return output_path


def write_summary_json(result: SimulationResult, path: str | Path) -> Path:
    """Write deterministic event and maximum summaries (runtime intentionally omitted)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "scenario": result.scenario_name,
        "events": result.event_summary,
        "maxima": result.maximum_summary,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
