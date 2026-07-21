"""Deterministic synthetic sensor-fault schedules for verification scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

SensorFaultMode = Literal["bias_step", "spike", "stuck", "dropout"]


@dataclass(frozen=True, slots=True)
class SensorFaultEvent:
    """One explicitly synthetic fault active over a half-open time interval."""

    sensor_name: str
    mode: SensorFaultMode
    start_time_s: float
    end_time_s: float
    value: FloatArray

    def __init__(
        self,
        sensor_name: str,
        mode: SensorFaultMode,
        start_time_s: float,
        end_time_s: float,
        value: npt.ArrayLike = (),
    ) -> None:
        if not sensor_name.strip():
            raise ValueError("sensor fault name must be nonempty")
        if mode not in {"bias_step", "spike", "stuck", "dropout"}:
            raise ValueError(f"unsupported sensor fault mode: {mode}")
        if not np.all(np.isfinite([start_time_s, end_time_s])) or end_time_s <= start_time_s:
            raise ValueError("sensor fault interval must be finite and increasing")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("sensor fault value must be a finite vector")
        if mode in {"bias_step", "spike"} and array.size == 0:
            raise ValueError(f"{mode} fault requires a nonempty value vector")
        object.__setattr__(self, "sensor_name", sensor_name.strip())
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "start_time_s", float(start_time_s))
        object.__setattr__(self, "end_time_s", float(end_time_s))
        object.__setattr__(self, "value", array.copy())


class SensorFaultInjector:
    """Apply repeatable bias, one-shot spike, stuck, and dropout faults."""

    def __init__(self, events: tuple[SensorFaultEvent, ...]) -> None:
        self.events = tuple(events)
        self._triggered_spikes: set[int] = set()
        self._stuck_values: dict[int, FloatArray] = {}

    def reset(self) -> None:
        """Clear one-shot and held-value state."""
        self._triggered_spikes.clear()
        self._stuck_values.clear()

    def apply(
        self,
        sensor_name: str,
        sample_time_s: float,
        nominal_value: npt.ArrayLike,
    ) -> FloatArray | None:
        """Return the faulted measurement, or ``None`` for a forced dropout."""
        value = np.asarray(nominal_value, dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("nominal sensor value must be a finite vector")
        if not np.isfinite(sample_time_s) or sample_time_s < 0.0:
            raise ValueError("sensor fault sample time must be finite and nonnegative")
        result = value.copy()
        for index, event in enumerate(self.events):
            if event.sensor_name != sensor_name:
                continue
            active = event.start_time_s <= sample_time_s < event.end_time_s
            if not active:
                if event.mode == "stuck":
                    self._stuck_values.pop(index, None)
                continue
            if event.mode == "dropout":
                return None
            if event.mode in {"bias_step", "spike"} and event.value.shape != result.shape:
                raise ValueError(
                    f"fault value for {sensor_name} has shape {event.value.shape}; "
                    f"expected {result.shape}"
                )
            if event.mode == "bias_step":
                result += event.value
            elif event.mode == "spike":
                if index not in self._triggered_spikes:
                    result += event.value
                    self._triggered_spikes.add(index)
            elif event.mode == "stuck":
                if index not in self._stuck_values:
                    self._stuck_values[index] = result.copy()
                result = self._stuck_values[index].copy()
        return result
