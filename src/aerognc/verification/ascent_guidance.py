"""Requirement assessment and deterministic outputs for constrained ascent."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.simulation.guided_ascent import AscentGuidanceOptimizationResult, GuidedAscentRun
from aerognc.simulation.logging import write_result_csv


@dataclass(frozen=True, slots=True)
class AscentGuidanceAssessment:
    """Explicit performance and powered-ascent constraint outcomes."""

    apogee_requirement_pass: bool
    max_q_requirement_pass: bool
    proper_load_requirement_pass: bool
    angle_of_attack_requirement_pass: bool
    mass_consistency_pass: bool
    objective_improvement_pass: bool

    @property
    def all_pass(self) -> bool:
        """Return whether every constrained-ascent check passed."""
        return all(
            (
                self.apogee_requirement_pass,
                self.max_q_requirement_pass,
                self.proper_load_requirement_pass,
                self.angle_of_attack_requirement_pass,
                self.mass_consistency_pass,
                self.objective_improvement_pass,
            )
        )


def assess_ascent_guidance(
    optimization: AscentGuidanceOptimizationResult,
) -> AscentGuidanceAssessment:
    """Compare optimized metrics with configured requirements."""
    configuration = optimization.configuration
    run = optimization.optimized_run
    mass = run.result.columns["mass_kg"]
    propellant = run.result.columns["propellant_mass_kg"]
    dry_mass = configuration.base_scenario.vehicle.mass_properties.dry_mass_kg
    mass_consistent = bool(
        np.all(np.diff(mass) <= 1.0e-10)
        and np.min(mass) >= dry_mass - 1.0e-10
        and np.allclose(mass, dry_mass + propellant, atol=1.0e-10)
    )
    return AscentGuidanceAssessment(
        apogee_requirement_pass=(abs(run.apogee_error_m) <= configuration.apogee_tolerance_m),
        max_q_requirement_pass=(
            run.maximum_dynamic_pressure_pa <= configuration.maximum_dynamic_pressure_pa
        ),
        proper_load_requirement_pass=(
            run.maximum_proper_load_factor <= configuration.maximum_proper_load_factor
        ),
        angle_of_attack_requirement_pass=(
            run.maximum_absolute_angle_of_attack_rad <= configuration.maximum_angle_of_attack_rad
        ),
        mass_consistency_pass=mass_consistent,
        objective_improvement_pass=(run.objective < optimization.reference_run.objective),
    )


def _run_payload(run: GuidedAscentRun) -> dict[str, object]:
    columns = run.result.columns
    time_s = run.result.time_s
    burnout_time = next(
        (event.time_s for event in run.result.events if event.name == "motor_window_end"),
        time_s[-1],
    )
    powered = time_s <= burnout_time
    limiter_durations = {
        name: float(np.trapezoid(columns[name][powered], time_s[powered]))
        for name in (
            "alpha_limiter_active",
            "max_q_limiter_active",
            "load_limiter_active",
            "apogee_limiter_active",
        )
    }
    return {
        "governor_enabled": run.governor_enabled,
        "decision": {
            "terminal_elevation_offset_deg": float(
                np.rad2deg(run.decision.terminal_elevation_offset_rad)
            ),
            "throttle_scale": run.decision.throttle_scale,
        },
        "objective": run.objective,
        "apogee_m": run.apogee_m,
        "apogee_error_m": run.apogee_error_m,
        "maximum_powered_dynamic_pressure_pa": run.maximum_dynamic_pressure_pa,
        "maximum_powered_proper_load_factor": run.maximum_proper_load_factor,
        "maximum_powered_absolute_angle_of_attack_deg": float(
            np.rad2deg(run.maximum_absolute_angle_of_attack_rad)
        ),
        "all_powered_constraints_satisfied": run.all_constraints_satisfied,
        "final_propellant_mass_kg": float(columns["propellant_mass_kg"][-1]),
        "limiter_active_duration_s": limiter_durations,
        "events": run.result.event_summary,
    }


def ascent_guidance_payload(
    optimization: AscentGuidanceOptimizationResult,
) -> dict[str, object]:
    """Return a JSON-safe comparison and verification record."""
    configuration = optimization.configuration
    assessment = assess_ascent_guidance(optimization)
    return {
        "scenario": configuration.name,
        "safety_scope": configuration.safety_scope,
        "constraint_scope": (
            "Max-Q and proper load are assessed from launch through the motor-window end. "
            "Angle of attack is assessed over that interval where dynamic pressure is at "
            f"least {configuration.minimum_alpha_constraint_dynamic_pressure_pa:.1f} Pa."
        ),
        "requirements": {
            "desired_apogee_m": configuration.desired_apogee_m,
            "apogee_tolerance_m": configuration.apogee_tolerance_m,
            "maximum_dynamic_pressure_pa": configuration.maximum_dynamic_pressure_pa,
            "maximum_proper_load_factor": configuration.maximum_proper_load_factor,
            "maximum_angle_of_attack_deg": float(
                np.rad2deg(configuration.maximum_angle_of_attack_rad)
            ),
            "apogee_requirement_pass": assessment.apogee_requirement_pass,
            "max_q_requirement_pass": assessment.max_q_requirement_pass,
            "proper_load_requirement_pass": assessment.proper_load_requirement_pass,
            "angle_of_attack_requirement_pass": (assessment.angle_of_attack_requirement_pass),
            "mass_consistency_pass": assessment.mass_consistency_pass,
            "objective_improvement_pass": assessment.objective_improvement_pass,
            "all_pass": assessment.all_pass,
        },
        "reference_run": _run_payload(optimization.reference_run),
        "optimized_run": _run_payload(optimization.optimized_run),
        "optimizer": {
            "method": "bounded deterministic coordinate search",
            "evaluation_count": len(optimization.evaluations),
            "evaluations": [
                {
                    "evaluation_index": item.evaluation_index,
                    "terminal_elevation_offset_deg": float(
                        np.rad2deg(item.decision.terminal_elevation_offset_rad)
                    ),
                    "throttle_scale": item.decision.throttle_scale,
                    "objective": item.objective,
                    "apogee_m": item.apogee_m,
                    "all_constraints_satisfied": item.all_constraints_satisfied,
                }
                for item in optimization.evaluations
            ],
        },
        "limitations": [
            "The offline optimizer has two design variables and is not a general "
            "optimal-control solver.",
            "Pitch response is a bounded first-order surrogate, not a full closed-loop "
            "6-DOF plant.",
            "The post-apogee attitude/recovery system is intentionally outside "
            "constraint assessment.",
        ],
    }


def write_ascent_guidance_results(
    optimization: AscentGuidanceOptimizationResult,
    output_directory: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Write reference/optimized trajectories, search history, and JSON report."""
    output = Path(output_directory or optimization.configuration.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = write_result_csv(
        optimization.reference_run.result, output / "reference_trajectory.csv"
    )
    optimized_path = write_result_csv(
        optimization.optimized_run.result, output / "optimized_trajectory.csv"
    )
    history_path = output / "optimization_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "evaluation_index",
                "terminal_elevation_offset_deg",
                "throttle_scale",
                "objective",
                "apogee_m",
                "all_constraints_satisfied",
            )
        )
        for item in optimization.evaluations:
            writer.writerow(
                (
                    item.evaluation_index,
                    f"{np.rad2deg(item.decision.terminal_elevation_offset_rad):.10g}",
                    f"{item.decision.throttle_scale:.10g}",
                    f"{item.objective:.10g}",
                    f"{item.apogee_m:.10g}",
                    str(item.all_constraints_satisfied).lower(),
                )
            )
    report_path = output / "ascent_guidance_report.json"
    report_path.write_text(
        json.dumps(ascent_guidance_payload(optimization), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return reference_path, optimized_path, history_path, report_path
