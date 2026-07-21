"""Repeatable resource benchmarking with explicit budgets and no real-time claim."""

from __future__ import annotations

import json
import os
import platform
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    """Optional inclusive resource and minimum-throughput acceptance limits."""

    maximum_wall_time_s: float | None = None
    maximum_cpu_time_s: float | None = None
    maximum_peak_traced_memory_mb: float | None = None
    minimum_samples_per_second: float | None = None
    minimum_steps_per_second: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"benchmark budget {name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class BenchmarkTrial:
    """One measured invocation."""

    wall_time_s: float
    cpu_time_s: float
    peak_traced_memory_bytes: int


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironment:
    """Machine/software context needed to interpret a local measurement."""

    python_version: str
    python_implementation: str
    python_compiler: str
    platform: str
    machine: str
    processor: str
    logical_cpu_count: int | None
    numpy_version: str


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Median resource evidence, all trials, budgets, and scope disclaimer."""

    name: str
    trials: tuple[BenchmarkTrial, ...]
    wall_time_s: float
    cpu_time_s: float
    peak_traced_memory_bytes: int
    sample_count: int
    step_count: int
    samples_per_second: float
    steps_per_second: float
    budget_results: dict[str, bool]
    passed: bool
    environment: BenchmarkEnvironment
    real_time_guarantee: bool = False


def benchmark_environment() -> BenchmarkEnvironment:
    """Capture stable environment fields without inventing hardware capabilities."""
    return BenchmarkEnvironment(
        platform.python_version(),
        platform.python_implementation(),
        platform.python_compiler(),
        platform.platform(),
        platform.machine(),
        platform.processor(),
        os.cpu_count(),
        np.__version__,
    )


def _assess_budget(result_values: dict[str, float], budget: BenchmarkBudget) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if budget.maximum_wall_time_s is not None:
        checks["maximum_wall_time"] = result_values["wall_time_s"] <= budget.maximum_wall_time_s
    if budget.maximum_cpu_time_s is not None:
        checks["maximum_cpu_time"] = result_values["cpu_time_s"] <= budget.maximum_cpu_time_s
    if budget.maximum_peak_traced_memory_mb is not None:
        checks["maximum_peak_traced_memory"] = (
            result_values["peak_traced_memory_bytes"]
            <= budget.maximum_peak_traced_memory_mb * 1.0e6
        )
    if budget.minimum_samples_per_second is not None:
        checks["minimum_samples_per_second"] = (
            result_values["samples_per_second"] >= budget.minimum_samples_per_second
        )
    if budget.minimum_steps_per_second is not None:
        checks["minimum_steps_per_second"] = (
            result_values["steps_per_second"] >= budget.minimum_steps_per_second
        )
    return checks


def run_benchmark(
    name: str,
    operation: Callable[[], object],
    *,
    sample_count: int,
    step_count: int,
    repetitions: int = 3,
    warmup: bool = True,
    budget: BenchmarkBudget | None = None,
) -> BenchmarkResult:
    """Measure one callable with wall/CPU clocks and Python allocation tracing."""
    if not name.strip():
        raise ValueError("benchmark name cannot be empty")
    if sample_count <= 0 or step_count <= 0:
        raise ValueError("benchmark sample_count and step_count must be positive")
    if not 1 <= repetitions <= 100:
        raise ValueError("benchmark repetitions must lie in [1, 100]")
    if warmup:
        operation()
    trials: list[BenchmarkTrial] = []
    for _ in range(repetitions):
        tracemalloc.start()
        start_wall_s = time.perf_counter()
        start_cpu_s = time.process_time()
        try:
            operation()
        finally:
            cpu_time_s = time.process_time() - start_cpu_s
            wall_time_s = time.perf_counter() - start_wall_s
            _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        trials.append(BenchmarkTrial(wall_time_s, cpu_time_s, peak_bytes))
    wall_time_s = float(np.median([trial.wall_time_s for trial in trials]))
    cpu_time_s = float(np.median([trial.cpu_time_s for trial in trials]))
    peak_bytes = round(float(np.median([trial.peak_traced_memory_bytes for trial in trials])))
    samples_per_second = sample_count / wall_time_s
    steps_per_second = step_count / wall_time_s
    values = {
        "wall_time_s": wall_time_s,
        "cpu_time_s": cpu_time_s,
        "peak_traced_memory_bytes": float(peak_bytes),
        "samples_per_second": samples_per_second,
        "steps_per_second": steps_per_second,
    }
    checks = _assess_budget(values, BenchmarkBudget() if budget is None else budget)
    return BenchmarkResult(
        name,
        tuple(trials),
        wall_time_s,
        cpu_time_s,
        peak_bytes,
        sample_count,
        step_count,
        samples_per_second,
        steps_per_second,
        checks,
        all(checks.values()),
        benchmark_environment(),
    )


def benchmark_payload(result: BenchmarkResult) -> dict[str, object]:
    """Return a JSON-safe report with the non-certification scope made explicit."""
    return {
        "schema_version": "1.0",
        "name": result.name,
        "representative_statistic": "median",
        "wall_time_s": result.wall_time_s,
        "cpu_time_s": result.cpu_time_s,
        "peak_traced_memory_bytes": result.peak_traced_memory_bytes,
        "sample_count": result.sample_count,
        "step_count": result.step_count,
        "samples_per_second": result.samples_per_second,
        "steps_per_second": result.steps_per_second,
        "trials": [asdict(trial) for trial in result.trials],
        "budget_results": result.budget_results,
        "passed": result.passed,
        "environment": asdict(result.environment),
        "real_time_guarantee": False,
        "scope_note": (
            "Local development-machine evidence only; this is not a hard real-time, "
            "deadline, WCET, or deployment-platform guarantee."
        ),
    }


def write_benchmark_report(result: BenchmarkResult, path: str | Path) -> Path:
    """Write one local benchmark record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(benchmark_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
