"""Configured launch-window optimization and numerical acceptance evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.astrodynamics.kepler import propagate_universal
from aerognc.astrodynamics.launch_window import LaunchWindowOptimization, optimize_launch_window
from aerognc.configuration.launch_window_loader import LaunchWindowConfiguration


@dataclass(frozen=True, slots=True)
class LaunchWindowAssessment:
    """Convergence, feasibility, endpoint, and performance assertions."""

    converged_pass: bool
    feasible_pass: bool
    endpoint_pass: bool
    delta_v_pass: bool
    refinement_nonworsening_pass: bool

    @property
    def all_pass(self) -> bool:
        """Return whether every launch-window requirement passes."""
        return all(
            (
                self.converged_pass,
                self.feasible_pass,
                self.endpoint_pass,
                self.delta_v_pass,
                self.refinement_nonworsening_pass,
            )
        )


@dataclass(frozen=True, slots=True)
class LaunchWindowRun:
    """Optimization output plus independent endpoint and assessment values."""

    configuration: LaunchWindowConfiguration
    optimization: LaunchWindowOptimization
    endpoint_error_m: float
    best_feasible_grid_delta_v_mps: float
    assessment: LaunchWindowAssessment


def run_launch_window_optimization(
    configuration: LaunchWindowConfiguration,
) -> LaunchWindowRun:
    """Run the configured deterministic grid/refinement search."""
    catalog = configuration.catalog
    departure = catalog.body(configuration.departure_body, role="departure")
    destination = catalog.body(configuration.destination_body, role="destination")
    primary_mu = catalog.primary.gravitational_parameter_m3_s2
    optimization = optimize_launch_window(
        departure,
        destination,
        primary_mu,
        departure_bounds_s=configuration.departure_bounds_s,
        arrival_bounds_s=configuration.arrival_bounds_s,
        departure_grid_count=configuration.departure_grid_count,
        arrival_grid_count=configuration.arrival_grid_count,
        departure_parking_altitude_m=configuration.departure_parking_altitude_m,
        arrival_parking_altitude_m=configuration.destination_parking_altitude_m,
        maximum_c3_m2_s2=configuration.maximum_c3_m2_s2,
        maximum_arrival_excess_speed_mps=(configuration.maximum_arrival_excess_speed_mps),
        maximum_refinement_iterations=configuration.maximum_refinement_iterations,
        epoch_tolerance_s=configuration.epoch_tolerance_s,
    )
    optimum = optimization.optimum
    departure_position, _departure_velocity = departure.state_at_time(
        optimum.departure_time_s, primary_mu
    )
    endpoint = propagate_universal(
        departure_position,
        optimum.transfer.lambert.departure_velocity_mps,
        optimum.arrival_time_s - optimum.departure_time_s,
        primary_mu,
    )
    destination_position, _destination_velocity = destination.state_at_time(
        optimum.arrival_time_s, primary_mu
    )
    endpoint_error_m = float(np.linalg.norm(endpoint.position_m - destination_position))
    feasible_grid_values = optimization.total_delta_v_grid_mps[optimization.feasible_grid]
    if feasible_grid_values.size == 0:
        best_grid_delta_v = np.inf
    else:
        best_grid_delta_v = float(np.min(feasible_grid_values))
    assessment = LaunchWindowAssessment(
        converged_pass=optimization.converged,
        feasible_pass=optimum.feasible,
        endpoint_pass=endpoint_error_m <= 0.1,
        delta_v_pass=(optimum.total_delta_v_mps <= configuration.maximum_total_delta_v_mps),
        refinement_nonworsening_pass=(optimum.total_delta_v_mps <= best_grid_delta_v + 1.0e-9),
    )
    return LaunchWindowRun(
        configuration,
        optimization,
        endpoint_error_m,
        best_grid_delta_v,
        assessment,
    )


def launch_window_payload(run: LaunchWindowRun) -> dict[str, object]:
    """Return a stable report without runtime or workspace-dependent paths."""
    optimum = run.optimization.optimum
    assessment = run.assessment
    return {
        "scenario": run.configuration.name,
        "safety_scope": run.configuration.safety_scope,
        "route": {
            "departure": run.configuration.departure_body,
            "destination": run.configuration.destination_body,
        },
        "optimizer": {
            "method": "complete rectangular grid plus bounded coordinate refinement",
            "grid_shape": [
                run.configuration.departure_grid_count,
                run.configuration.arrival_grid_count,
            ],
            "evaluation_count": run.optimization.evaluation_count,
            "improvement_count": len(run.optimization.history) - 1,
            "epoch_tolerance_s": run.configuration.epoch_tolerance_s,
            "converged": run.optimization.converged,
        },
        "optimum": {
            "departure_day": optimum.departure_time_s / 86_400.0,
            "arrival_day": optimum.arrival_time_s / 86_400.0,
            "time_of_flight_days": (optimum.arrival_time_s - optimum.departure_time_s) / 86_400.0,
            "departure_c3_m2_s2": optimum.transfer.departure_c3_m2_s2,
            "arrival_excess_speed_mps": optimum.transfer.arrival_excess_speed_mps,
            "injection_delta_v_mps": optimum.injection_delta_v_mps,
            "capture_delta_v_mps": optimum.capture_delta_v_mps,
            "total_delta_v_mps": optimum.total_delta_v_mps,
            "best_feasible_grid_delta_v_mps": run.best_feasible_grid_delta_v_mps,
            "lambert_endpoint_error_m": run.endpoint_error_m,
        },
        "requirements": {
            "converged_pass": assessment.converged_pass,
            "feasible_pass": assessment.feasible_pass,
            "endpoint_pass": assessment.endpoint_pass,
            "delta_v_pass": assessment.delta_v_pass,
            "refinement_nonworsening_pass": assessment.refinement_nonworsening_pass,
            "all_pass": assessment.all_pass,
        },
        "limitations": [
            "Worlds and epochs are fictional and synthetic.",
            "The search is deterministic local refinement after a finite grid, not global proof.",
            "The objective uses ideal impulsive parking-orbit injection and capture.",
            "Operational ephemerides, navigation, finite burns, launch, entry, and "
            "landing are excluded.",
        ],
    }


def write_launch_window_report(run: LaunchWindowRun, output_directory: str | Path) -> Path:
    """Write the deterministic optimization report."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "launch_window_optimization_report.json"
    path.write_text(
        json.dumps(launch_window_payload(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
