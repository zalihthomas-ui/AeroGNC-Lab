"""Deterministic reports for flight-envelope and gain-schedule verification."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.gnc.flight_envelope import FlightEnvelopeResult


@dataclass(frozen=True, slots=True)
class EnvelopeRequirementAssessment:
    """Named, quantitative pass/fail outcomes for the configured envelope."""

    trim_convergence_pass: bool
    controllability_pass: bool
    observability_pass: bool
    design_grid_stability_pass: bool
    interpolated_schedule_stability_pass: bool
    robust_stability_pass: bool
    control_authority_pass: bool
    damping_pass: bool

    @property
    def all_pass(self) -> bool:
        """Return whether every envelope verification item passed."""
        return all(
            (
                self.trim_convergence_pass,
                self.controllability_pass,
                self.observability_pass,
                self.design_grid_stability_pass,
                self.interpolated_schedule_stability_pass,
                self.robust_stability_pass,
                self.control_authority_pass,
                self.damping_pass,
            )
        )


def assess_flight_envelope(result: FlightEnvelopeResult) -> EnvelopeRequirementAssessment:
    """Assess the result against explicit numerical configuration requirements."""
    configuration = result.configuration
    return EnvelopeRequirementAssessment(
        trim_convergence_pass=result.all_trim_converged,
        controllability_pass=all(item.controllability_rank == 2 for item in result.analyses),
        observability_pass=all(item.observability_rank == 2 for item in result.analyses),
        design_grid_stability_pass=all(
            all(mode.stable for mode in item.closed_loop_modes) for item in result.analyses
        ),
        interpolated_schedule_stability_pass=(
            result.schedule_verification.stable_point_count
            == result.schedule_verification.evaluated_point_count
        ),
        robust_stability_pass=(
            result.robustness_verification.stable_sample_count
            == result.robustness_verification.sample_count
        ),
        control_authority_pass=(
            result.minimum_control_authority_fraction
            >= configuration.minimum_control_authority_fraction
        ),
        damping_pass=(
            min(
                result.minimum_closed_loop_damping_ratio,
                result.schedule_verification.minimum_damping_ratio,
                result.robustness_verification.minimum_damping_ratio,
            )
            >= configuration.minimum_closed_loop_damping_ratio
        ),
    )


def _mode_payload(eigenvalue: complex) -> dict[str, float]:
    return {
        "real_radps": float(eigenvalue.real),
        "imaginary_radps": float(eigenvalue.imag),
    }


def flight_envelope_payload(result: FlightEnvelopeResult) -> dict[str, object]:
    """Create the stable, JSON-safe envelope evidence payload."""
    assessment = assess_flight_envelope(result)
    configuration = result.configuration
    points: list[dict[str, object]] = []
    for analysis in result.analyses:
        operating = analysis.operating_point
        points.append(
            {
                "mach": operating.mach,
                "altitude_m": operating.altitude_m,
                "mass_kg": operating.mass_kg,
                "airspeed_mps": operating.airspeed_mps,
                "dynamic_pressure_pa": operating.dynamic_pressure_pa,
                "centre_of_gravity_from_nose_m": operating.centre_of_gravity_from_nose_m,
                "pitch_inertia_kgm2": operating.pitch_inertia_kgm2,
                "trim_converged": analysis.trim.converged,
                "trim_alpha_rad": float(analysis.trim.decision[0]),
                "trim_command_rad": float(analysis.trim.decision[1]),
                "trim_residual_infinity_norm": float(
                    np.linalg.norm(analysis.trim.residual, ord=np.inf)
                ),
                "system_matrix": analysis.linear_model.system_matrix.tolist(),
                "input_matrix": analysis.linear_model.input_matrix.tolist(),
                "lqr_gain": analysis.lqr.gain.tolist(),
                "open_loop_modes": [
                    _mode_payload(mode.eigenvalue) for mode in analysis.open_loop_modes
                ],
                "closed_loop_modes": [
                    _mode_payload(mode.eigenvalue) for mode in analysis.closed_loop_modes
                ],
                "controllability_rank": analysis.controllability_rank,
                "observability_rank": analysis.observability_rank,
                "control_authority_fraction": analysis.control_authority_fraction,
                "remaining_control_moment_nm": analysis.remaining_control_moment_nm,
            }
        )
    return {
        "scenario": configuration.name,
        "safety_scope": configuration.safety_scope,
        "base_scenario": str(configuration.base_scenario.source_path),
        "grid": {
            "mach": list(configuration.mach_points),
            "altitude_m": list(configuration.altitude_points_m),
            "mass_kg": list(configuration.mass_points_kg),
            "point_count": len(points),
        },
        "summary": {
            "all_trim_converged": result.all_trim_converged,
            "minimum_control_authority_fraction": result.minimum_control_authority_fraction,
            "minimum_design_grid_damping_ratio": result.minimum_closed_loop_damping_ratio,
        },
        "schedule_verification": {
            "evaluated_point_count": result.schedule_verification.evaluated_point_count,
            "stable_point_count": result.schedule_verification.stable_point_count,
            "minimum_damping_ratio": result.schedule_verification.minimum_damping_ratio,
            "worst_real_eigenvalue_radps": (
                result.schedule_verification.worst_real_eigenvalue_radps
            ),
        },
        "robustness_verification": {
            "random_seed": configuration.random_seed,
            "sample_count": result.robustness_verification.sample_count,
            "stable_sample_count": result.robustness_verification.stable_sample_count,
            "stable_fraction": result.robustness_verification.stable_fraction,
            "minimum_damping_ratio": result.robustness_verification.minimum_damping_ratio,
            "worst_real_eigenvalue_radps": (
                result.robustness_verification.worst_real_eigenvalue_radps
            ),
        },
        "requirements": {
            "minimum_control_authority_fraction": (
                configuration.minimum_control_authority_fraction
            ),
            "minimum_closed_loop_damping_ratio": (configuration.minimum_closed_loop_damping_ratio),
            "trim_convergence_pass": assessment.trim_convergence_pass,
            "controllability_pass": assessment.controllability_pass,
            "observability_pass": assessment.observability_pass,
            "design_grid_stability_pass": assessment.design_grid_stability_pass,
            "interpolated_schedule_stability_pass": (
                assessment.interpolated_schedule_stability_pass
            ),
            "robust_stability_pass": assessment.robust_stability_pass,
            "control_authority_pass": assessment.control_authority_pass,
            "damping_pass": assessment.damping_pass,
            "all_pass": assessment.all_pass,
        },
        "points": points,
    }


def write_flight_envelope_results(
    result: FlightEnvelopeResult,
    output_directory: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a full JSON report and compact grid-point CSV."""
    output = Path(output_directory or result.configuration.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "flight_envelope_report.json"
    json_path.write_text(
        json.dumps(flight_envelope_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output / "flight_envelope_points.csv"
    fieldnames = (
        "mach",
        "altitude_m",
        "mass_kg",
        "dynamic_pressure_pa",
        "pitch_inertia_kgm2",
        "trim_alpha_deg",
        "trim_command_deg",
        "angle_gain",
        "rate_gain",
        "control_authority_fraction",
        "remaining_control_moment_nm",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for analysis in result.analyses:
            operating = analysis.operating_point
            writer.writerow(
                {
                    "mach": operating.mach,
                    "altitude_m": operating.altitude_m,
                    "mass_kg": operating.mass_kg,
                    "dynamic_pressure_pa": operating.dynamic_pressure_pa,
                    "pitch_inertia_kgm2": operating.pitch_inertia_kgm2,
                    "trim_alpha_deg": float(np.rad2deg(analysis.trim.decision[0])),
                    "trim_command_deg": float(np.rad2deg(analysis.trim.decision[1])),
                    "angle_gain": float(analysis.lqr.gain[0, 0]),
                    "rate_gain": float(analysis.lqr.gain[0, 1]),
                    "control_authority_fraction": analysis.control_authority_fraction,
                    "remaining_control_moment_nm": analysis.remaining_control_moment_nm,
                }
            )
    return json_path, csv_path
