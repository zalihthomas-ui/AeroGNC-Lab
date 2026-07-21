"""Deterministic logical-time scheduling for multi-rate simulation tasks."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class ScheduledInvocation:
    """One task dispatch at an exact logical simulation time."""

    task_name: str
    logical_time_s: float
    tick: int


TaskCallback = Callable[[ScheduledInvocation], None]


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """A periodic callback with a phase and execution-time deadline."""

    name: str
    period_s: float
    callback: TaskCallback
    phase_s: float = 0.0
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scheduled task name cannot be empty")
        if not np.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ValueError("scheduled task period_s must be positive and finite")
        if not np.isfinite(self.phase_s) or self.phase_s < 0.0:
            raise ValueError("scheduled task phase_s must be non-negative and finite")
        if self.deadline_s is not None and (
            not np.isfinite(self.deadline_s) or self.deadline_s <= 0.0
        ):
            raise ValueError("scheduled task deadline_s must be positive and finite")


@dataclass(frozen=True, slots=True)
class TaskTimingStatistics:
    """Measured callback timing; it never changes logical dispatch order."""

    invocations: int
    missed_deadlines: int
    total_execution_time_s: float
    maximum_execution_time_s: float


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """Dispatch sequence and per-task timing statistics."""

    invocations: tuple[ScheduledInvocation, ...]
    statistics: Mapping[str, TaskTimingStatistics]


class CancellationProbe(Protocol):
    """Minimum cancellation interface shared with project jobs."""

    def raise_if_cancelled(self) -> None:
        """Raise when cancellation has been requested."""


class LogicalTimeScheduler:
    """Run periodic callbacks by logical time with stable registration-order ties."""

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock

    def run(
        self,
        tasks: Sequence[ScheduledTask],
        time_span_s: tuple[float, float],
        *,
        cancellation: CancellationProbe | None = None,
    ) -> ScheduleResult:
        """Dispatch every task due within the inclusive logical-time interval."""
        start_s, end_s = time_span_s
        if not np.isfinite([start_s, end_s]).all() or end_s < start_s:
            raise ValueError("time_span_s must be finite and non-decreasing")
        if len({task.name for task in tasks}) != len(tasks):
            raise ValueError("scheduled task names must be unique")

        queue: list[tuple[float, int, int]] = []
        for index, task in enumerate(tasks):
            first_time_s = start_s + task.phase_s
            if first_time_s <= end_s:
                heapq.heappush(queue, (first_time_s, index, 0))

        records: list[ScheduledInvocation] = []
        invocation_count = [0] * len(tasks)
        missed_deadlines = [0] * len(tasks)
        total_execution = [0.0] * len(tasks)
        maximum_execution = [0.0] * len(tasks)

        while queue:
            logical_time_s, index, tick = heapq.heappop(queue)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            task = tasks[index]
            invocation = ScheduledInvocation(task.name, float(logical_time_s), tick)
            started_s = self._clock()
            task.callback(invocation)
            elapsed_s = max(0.0, self._clock() - started_s)
            records.append(invocation)
            invocation_count[index] += 1
            total_execution[index] += elapsed_s
            maximum_execution[index] = max(maximum_execution[index], elapsed_s)
            if task.deadline_s is not None and elapsed_s > task.deadline_s:
                missed_deadlines[index] += 1

            next_tick = tick + 1
            next_time_s = start_s + task.phase_s + next_tick * task.period_s
            tolerance_s = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(end_s))
            if next_time_s <= end_s + tolerance_s:
                heapq.heappush(queue, (min(next_time_s, end_s), index, next_tick))

        statistics = {
            task.name: TaskTimingStatistics(
                invocations=invocation_count[index],
                missed_deadlines=missed_deadlines[index],
                total_execution_time_s=total_execution[index],
                maximum_execution_time_s=maximum_execution[index],
            )
            for index, task in enumerate(tasks)
        }
        return ScheduleResult(tuple(records), MappingProxyType(statistics))
