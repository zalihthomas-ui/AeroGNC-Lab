"""Unit-aware deterministic comparison of stored engineering trajectories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aerognc.mathematics.vectors import FloatArray
from aerognc.project.result_store import ResultDataset


@dataclass(frozen=True, slots=True)
class ChannelComparison:
    """Difference metrics for one aligned same-unit channel."""

    channel: str
    unit: str
    sample_count: int
    bias: float
    rms_difference: float
    maximum_absolute_difference: float
    final_difference: float
    correlation: float


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Common time base and per-channel metrics for two trajectories."""

    baseline_scenario: str
    candidate_scenario: str
    start_time_s: float
    end_time_s: float
    time_s: FloatArray
    channels: tuple[ChannelComparison, ...]


def compare_datasets(
    baseline: ResultDataset,
    candidate: ResultDataset,
    *,
    channels: Sequence[str] | None = None,
    sample_count: int | None = None,
) -> RunComparison:
    """Interpolate two datasets over their common domain and calculate differences."""
    selected = list(baseline.channels) if channels is None else list(channels)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("comparison channels must be nonempty and unique")
    missing_baseline = [name for name in selected if name not in baseline.channels]
    missing_candidate = [name for name in selected if name not in candidate.channels]
    if missing_baseline or missing_candidate:
        raise KeyError(
            f"comparison channels missing; baseline={missing_baseline}, "
            f"candidate={missing_candidate}"
        )
    for name in selected:
        if baseline.units[name] != candidate.units[name]:
            raise ValueError(
                f"unit mismatch for {name!r}: {baseline.units[name]!r} vs {candidate.units[name]!r}"
            )
    start_time_s = max(float(baseline.time_s[0]), float(candidate.time_s[0]))
    end_time_s = min(float(baseline.time_s[-1]), float(candidate.time_s[-1]))
    if end_time_s <= start_time_s:
        raise ValueError("datasets have no common time domain")
    if sample_count is None:
        sample_count = min(max(baseline.time_s.size, candidate.time_s.size), 5000)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    time_s = np.linspace(start_time_s, end_time_s, sample_count, dtype=np.float64)
    metrics: list[ChannelComparison] = []
    for name in selected:
        baseline_values = np.interp(time_s, baseline.time_s, baseline.channels[name])
        candidate_values = np.interp(time_s, candidate.time_s, candidate.channels[name])
        difference = candidate_values - baseline_values
        baseline_std = float(np.std(baseline_values))
        candidate_std = float(np.std(candidate_values))
        if baseline_std <= np.finfo(np.float64).eps or candidate_std <= np.finfo(np.float64).eps:
            correlation = 1.0 if np.allclose(baseline_values, candidate_values) else 0.0
        else:
            correlation = float(np.corrcoef(baseline_values, candidate_values)[0, 1])
        metrics.append(
            ChannelComparison(
                channel=name,
                unit=baseline.units[name],
                sample_count=sample_count,
                bias=float(np.mean(difference)),
                rms_difference=float(np.sqrt(np.mean(difference**2))),
                maximum_absolute_difference=float(np.max(np.abs(difference))),
                final_difference=float(difference[-1]),
                correlation=correlation,
            )
        )
    time_s.setflags(write=False)
    return RunComparison(
        baseline_scenario=baseline.scenario_name,
        candidate_scenario=candidate.scenario_name,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        time_s=time_s,
        channels=tuple(metrics),
    )


def comparison_as_mapping(comparison: RunComparison) -> dict[str, Any]:
    """Return deterministic machine-readable comparison metrics."""
    return {
        "baseline_scenario": comparison.baseline_scenario,
        "candidate_scenario": comparison.candidate_scenario,
        "start_time_s": comparison.start_time_s,
        "end_time_s": comparison.end_time_s,
        "sample_count": int(comparison.time_s.size),
        "channels": [
            {
                "channel": item.channel,
                "unit": item.unit,
                "sample_count": item.sample_count,
                "bias": item.bias,
                "rms_difference": item.rms_difference,
                "maximum_absolute_difference": item.maximum_absolute_difference,
                "final_difference": item.final_difference,
                "correlation": item.correlation,
            }
            for item in comparison.channels
        ],
    }


def write_comparison_json(comparison: RunComparison, path: str | Path) -> Path:
    """Write comparison metrics without the redundant aligned vectors."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(comparison_as_mapping(comparison), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination
