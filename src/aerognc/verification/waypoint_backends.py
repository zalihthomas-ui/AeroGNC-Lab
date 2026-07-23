"""Quantitative reduced-versus-coefficient waypoint backend comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aerognc.mission.mission import Mission
from aerognc.simulation.waypoint_backends import VehicleBackendKind
from aerognc.simulation.waypoint_mission import (
    MissionSample,
    WaypointMissionConfig,
    WaypointMissionResult,
    run_waypoint_mission,
)


@dataclass(frozen=True, slots=True)
class WaypointCrossModelTolerances:
    """Acceptance limits for a matched mission comparison."""

    maximum_cross_track_m: float = 175.0
    maximum_duration_ratio: float = 1.5
    maximum_final_horizontal_separation_m: float = 5.0
    maximum_final_altitude_difference_m: float = 5.0
    maximum_final_airspeed_difference_mps: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("cross-model tolerances must be positive and finite")
        if self.maximum_duration_ratio < 1.0:
            raise ValueError("maximum_duration_ratio must be at least one")


@dataclass(frozen=True, slots=True)
class WaypointCrossModelComparison:
    """Matched mission results and independent cross-model acceptance metrics."""

    reduced: WaypointMissionResult
    coefficient: WaypointMissionResult
    tolerances: WaypointCrossModelTolerances

    @property
    def duration_ratio(self) -> float:
        """Return longer duration divided by shorter duration."""
        durations = np.asarray(
            [self.reduced.duration_s, self.coefficient.duration_s], dtype=np.float64
        )
        return float(np.max(durations) / max(float(np.min(durations)), 1.0e-12))

    @property
    def final_horizontal_separation_m(self) -> float:
        """Return separation between terminal local-NED horizontal positions."""
        reduced, coefficient = self._final_samples()
        return float(
            np.linalg.norm(
                np.array([reduced.north_m, reduced.east_m])
                - np.array([coefficient.north_m, coefficient.east_m])
            )
        )

    @property
    def final_altitude_difference_m(self) -> float:
        reduced, coefficient = self._final_samples()
        return abs(reduced.altitude_m - coefficient.altitude_m)

    @property
    def final_airspeed_difference_mps(self) -> float:
        reduced, coefficient = self._final_samples()
        return abs(reduced.airspeed_mps - coefficient.airspeed_mps)

    @property
    def passed(self) -> bool:
        """Return whether both models meet the declared matched-mission limits."""
        if not self.reduced.completed or not self.coefficient.completed:
            return False
        if self.reduced.metadata.get("safety_events") or self.coefficient.metadata.get(
            "safety_events"
        ):
            return False
        reduced_summary = self.reduced.summary()
        coefficient_summary = self.coefficient.summary()
        maximum_cross_track = max(
            float(reduced_summary["max_abs_cross_track_m"]),
            float(coefficient_summary["max_abs_cross_track_m"]),
        )
        return bool(
            maximum_cross_track <= self.tolerances.maximum_cross_track_m
            and self.duration_ratio <= self.tolerances.maximum_duration_ratio
            and self.final_horizontal_separation_m
            <= self.tolerances.maximum_final_horizontal_separation_m
            and self.final_altitude_difference_m
            <= self.tolerances.maximum_final_altitude_difference_m
            and self.final_airspeed_difference_mps
            <= self.tolerances.maximum_final_airspeed_difference_mps
        )

    def summary(self) -> dict[str, object]:
        """Return a provenance-rich JSON-compatible validation record."""
        return {
            "schema_version": "1.0",
            "scope": "simulation-only fictional civilian fixed-wing waypoint backends",
            "passed": self.passed,
            "tolerances": asdict(self.tolerances),
            "metrics": {
                "duration_ratio": self.duration_ratio,
                "final_horizontal_separation_m": self.final_horizontal_separation_m,
                "final_altitude_difference_m": self.final_altitude_difference_m,
                "final_airspeed_difference_mps": self.final_airspeed_difference_mps,
            },
            "reduced": {
                "summary": self.reduced.summary(),
                "backend": self.reduced.metadata["vehicle_backend_details"],
            },
            "coefficient": {
                "summary": self.coefficient.summary(),
                "backend": self.coefficient.metadata["vehicle_backend_details"],
            },
            "interpretation": (
                "Agreement means both independently structured simulation models satisfy the "
                "declared mission-level bounds; it is not aircraft certification or proof of "
                "model identity."
            ),
        }

    def _final_samples(self) -> tuple[MissionSample, MissionSample]:
        if not self.reduced.samples or not self.coefficient.samples:
            raise ValueError("cross-model comparison requires nonempty mission logs")
        return self.reduced.samples[-1], self.coefficient.samples[-1]


def compare_waypoint_vehicle_models(
    mission: Mission,
    reduced_config: WaypointMissionConfig,
    coefficient_config: WaypointMissionConfig,
    tolerances: WaypointCrossModelTolerances | None = None,
) -> WaypointCrossModelComparison:
    """Run a matched mission through both built-in backends and compare outcomes."""
    if reduced_config.vehicle_backend is not VehicleBackendKind.INTERNAL_REDUCED:
        raise ValueError("reduced_config must select internal_reduced")
    if coefficient_config.vehicle_backend is not VehicleBackendKind.INTERNAL_COEFFICIENT:
        raise ValueError("coefficient_config must select internal_coefficient")
    if reduced_config.mission_sha256 != coefficient_config.mission_sha256:
        raise ValueError("cross-model configurations must identify the same mission input")
    reduced_result = run_waypoint_mission(mission, reduced_config)
    coefficient_result = run_waypoint_mission(mission, coefficient_config)
    return WaypointCrossModelComparison(
        reduced_result,
        coefficient_result,
        tolerances or WaypointCrossModelTolerances(),
    )


def write_waypoint_cross_model_comparison(
    comparison: WaypointCrossModelComparison,
    path: str | Path,
) -> Path:
    """Write the complete comparison summary as deterministic JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "WaypointCrossModelComparison",
    "WaypointCrossModelTolerances",
    "compare_waypoint_vehicle_models",
    "write_waypoint_cross_model_comparison",
]
