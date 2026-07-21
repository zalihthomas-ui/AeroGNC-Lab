"""Requirement assessment and deterministic records for advanced navigation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.simulation.advanced_navigation import (
    AdvancedNavigationResult,
    AidingUpdateLog,
    NavigationConsistencyResult,
)


@dataclass(frozen=True, slots=True)
class AdvancedNavigationAssessment:
    """Explicit performance, integrity, replay, and consistency outcomes."""

    position_rms_pass: bool
    velocity_rms_pass: bool
    attitude_rms_pass: bool
    delayed_replay_pass: bool
    fault_rejection_pass: bool
    sensor_recovery_pass: bool
    observability_pass: bool
    nees_consistency_pass: bool
    gnss_nis_consistency_pass: bool
    barometer_nis_consistency_pass: bool
    coning_sculling_effect_pass: bool

    @property
    def all_pass(self) -> bool:
        """Return whether every declared advanced-navigation requirement passes."""
        return all(
            (
                self.position_rms_pass,
                self.velocity_rms_pass,
                self.attitude_rms_pass,
                self.delayed_replay_pass,
                self.fault_rejection_pass,
                self.sensor_recovery_pass,
                self.observability_pass,
                self.nees_consistency_pass,
                self.gnss_nis_consistency_pass,
                self.barometer_nis_consistency_pass,
                self.coning_sculling_effect_pass,
            )
        )


def assess_advanced_navigation(
    result: AdvancedNavigationResult,
    consistency: NavigationConsistencyResult,
) -> AdvancedNavigationAssessment:
    """Assess the configured synthetic scenario against measurable limits."""
    gnss_updates = tuple(update for update in result.aiding_updates if update.sensor_name == "gnss")
    barometer_updates = tuple(
        update for update in result.aiding_updates if update.sensor_name == "barometer"
    )
    gnss_rejected = sum(not update.accepted for update in gnss_updates)
    barometer_rejected = sum(not update.accepted for update in barometer_updates)

    def recovered(updates: tuple[AidingUpdateLog, ...]) -> bool:
        if not updates:
            return False
        rejected_indices = [index for index, update in enumerate(updates) if not update.accepted]
        if not rejected_indices:
            return False
        return any(
            update.accepted and update.health == "healthy"
            for update in updates[rejected_indices[-1] + 1 :]
        )

    return AdvancedNavigationAssessment(
        position_rms_pass=result.position_rms_m <= 3.0,
        velocity_rms_pass=result.velocity_rms_mps <= 0.5,
        attitude_rms_pass=result.attitude_rms_deg <= 3.0,
        delayed_replay_pass=result.maximum_replayed_step_count >= 18,
        fault_rejection_pass=(gnss_rejected >= 5 and barometer_rejected >= 3),
        sensor_recovery_pass=(recovered(gnss_updates) and recovered(barometer_updates)),
        observability_pass=result.observability_rank == 15,
        nees_consistency_pass=consistency.nees_inside_fraction >= 0.80,
        gnss_nis_consistency_pass=(
            consistency.gnss_nis_lower <= consistency.mean_gnss_nis <= consistency.gnss_nis_upper
        ),
        barometer_nis_consistency_pass=(
            consistency.barometer_nis_lower
            <= consistency.mean_barometer_nis
            <= consistency.barometer_nis_upper
        ),
        coning_sculling_effect_pass=result.uncompensated_position_error_m >= 0.02,
    )


def advanced_navigation_payload(
    result: AdvancedNavigationResult,
    consistency: NavigationConsistencyResult,
) -> dict[str, object]:
    """Return a stable JSON-safe verification record without wall-clock values."""
    assessment = assess_advanced_navigation(result, consistency)
    sensor_payload: dict[str, object] = {}
    for sensor_name in ("gnss", "barometer"):
        updates = [update for update in result.aiding_updates if update.sensor_name == sensor_name]
        sensor_payload[sensor_name] = {
            "measurement_count": len(updates),
            "accepted_count": sum(update.accepted for update in updates),
            "rejected_count": sum(not update.accepted for update in updates),
            "maximum_nis": max((update.nis for update in updates), default=float("nan")),
            "final_health": updates[-1].health if updates else "unavailable",
        }
    return {
        "seed": result.seed,
        "scope": (
            "Synthetic civilian rotating-planet strapdown navigation and integrity "
            "verification; not a flight-ready navigation product."
        ),
        "performance": {
            "position_rms_m": result.position_rms_m,
            "velocity_rms_mps": result.velocity_rms_mps,
            "attitude_rms_deg": result.attitude_rms_deg,
            "final_position_error_ned_m": result.position_error_m[-1].tolist(),
            "final_velocity_error_ned_mps": result.velocity_error_mps[-1].tolist(),
        },
        "fixed_lag_and_integrity": {
            "maximum_replayed_step_count": result.maximum_replayed_step_count,
            "sensors": sensor_payload,
        },
        "coning_sculling": {
            "uncompensated_position_difference_m": result.uncompensated_position_error_m,
            "uncompensated_attitude_difference_deg": (result.uncompensated_attitude_error_deg),
        },
        "observability": {
            "normalised_gramian_rank": result.observability_rank,
            "normalised_gramian_singular_values": (result.observability_singular_values.tolist()),
            "rank_tolerance_relative": 1.0e-8,
        },
        "consistency": {
            "run_count": consistency.run_count,
            "confidence": consistency.confidence,
            "seeds": list(consistency.seeds),
            "nees_15_mean_over_time": float(np.mean(consistency.mean_nees_15)),
            "nees_15_bounds_for_ensemble_mean": [
                consistency.nees_lower,
                consistency.nees_upper,
            ],
            "nees_inside_fraction": consistency.nees_inside_fraction,
            "gnss_mean_nis": consistency.mean_gnss_nis,
            "gnss_mean_nis_bounds": [
                consistency.gnss_nis_lower,
                consistency.gnss_nis_upper,
            ],
            "barometer_mean_nis": consistency.mean_barometer_nis,
            "barometer_mean_nis_bounds": [
                consistency.barometer_nis_lower,
                consistency.barometer_nis_upper,
            ],
        },
        "requirements": {
            "position_rms_at_most_3_m": assessment.position_rms_pass,
            "velocity_rms_at_most_0_5_mps": assessment.velocity_rms_pass,
            "attitude_rms_at_most_3_deg": assessment.attitude_rms_pass,
            "delayed_replay_exercised": assessment.delayed_replay_pass,
            "injected_faults_rejected": assessment.fault_rejection_pass,
            "sensor_health_recovered": assessment.sensor_recovery_pass,
            "observability_rank_15": assessment.observability_pass,
            "nees_inside_fraction_at_least_0_80": assessment.nees_consistency_pass,
            "gnss_mean_nis_inside_confidence_bounds": (assessment.gnss_nis_consistency_pass),
            "barometer_mean_nis_inside_confidence_bounds": (
                assessment.barometer_nis_consistency_pass
            ),
            "coning_sculling_effect_resolved": assessment.coning_sculling_effect_pass,
            "all_pass": assessment.all_pass,
        },
        "limitations": [
            "The truth motion and all sensor characteristics are synthetic.",
            "The 15-state model omits lever arms, scale factors, misalignment states, "
            "clock states, and atmosphere-dependent GNSS errors.",
            "The observability rank applies only to the configured maneuver and sensor set.",
            "Chi-square bounds diagnose statistical consistency; they do not certify a "
            "physical navigation system.",
        ],
    }


def write_advanced_navigation_results(
    result: AdvancedNavigationResult,
    consistency: NavigationConsistencyResult,
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Write deterministic state history, aiding audit, and JSON report."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "advanced_navigation_trajectory.csv"
    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "time_s",
                "position_error_north_m",
                "position_error_east_m",
                "position_error_down_m",
                "velocity_error_north_mps",
                "velocity_error_east_mps",
                "velocity_error_down_mps",
                "attitude_error_roll_rad",
                "attitude_error_pitch_rad",
                "attitude_error_yaw_rad",
                "position_sigma_north_m",
                "position_sigma_east_m",
                "position_sigma_down_m",
                "nees_15",
            )
        )
        for index, time_s in enumerate(result.time_s):
            row = (
                time_s,
                *result.position_error_m[index],
                *result.velocity_error_mps[index],
                *result.attitude_error_rad[index],
                *result.position_sigma_m[index],
                result.nees_15[index],
            )
            writer.writerow(f"{float(value):.10g}" for value in row)

    aiding_path = output / "advanced_navigation_aiding_audit.csv"
    with aiding_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "sensor",
                "sample_time_s",
                "processing_time_s",
                "accepted",
                "nis",
                "threshold",
                "replayed_step_count",
                "health",
            )
        )
        for update in result.aiding_updates:
            writer.writerow(
                (
                    update.sensor_name,
                    f"{update.sample_time_s:.10g}",
                    f"{update.processing_time_s:.10g}",
                    str(update.accepted).lower(),
                    f"{update.nis:.10g}",
                    f"{update.threshold:.10g}",
                    update.replayed_step_count,
                    update.health,
                )
            )

    report_path = output / "advanced_navigation_report.json"
    report_path.write_text(
        json.dumps(
            advanced_navigation_payload(result, consistency),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return trajectory_path, aiding_path, report_path
