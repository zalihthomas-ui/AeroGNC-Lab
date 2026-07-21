"""Deterministic constrained launch-window grid and coordinate refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.astrodynamics.mission_design import (
    TransferOpportunity,
    evaluate_lambert_transfer,
    injection_delta_v_mps,
)


@dataclass(frozen=True, slots=True)
class LaunchWindowCandidate:
    """One evaluated direct-transfer launch opportunity."""

    departure_time_s: float
    arrival_time_s: float
    transfer: TransferOpportunity
    injection_delta_v_mps: float
    capture_delta_v_mps: float
    total_delta_v_mps: float
    feasible: bool
    constraint_penalty_mps: float
    penalized_objective_mps: float


@dataclass(frozen=True, slots=True)
class LaunchWindowOptimization:
    """Coarse-grid arrays, refined optimum, and deterministic convergence record."""

    departure_grid_s: np.ndarray[tuple[int], np.dtype[np.float64]]
    arrival_grid_s: np.ndarray[tuple[int], np.dtype[np.float64]]
    total_delta_v_grid_mps: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    feasible_grid: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    optimum: LaunchWindowCandidate
    history: tuple[LaunchWindowCandidate, ...]
    evaluation_count: int
    converged: bool


def _candidate(
    departure_body: CircularOrbitBody,
    arrival_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    departure_time_s: float,
    arrival_time_s: float,
    departure_parking_radius_m: float,
    arrival_parking_radius_m: float,
    maximum_c3_m2_s2: float,
    maximum_arrival_excess_speed_mps: float,
) -> LaunchWindowCandidate:
    transfer = evaluate_lambert_transfer(
        departure_body,
        arrival_body,
        primary_mu_m3_s2,
        departure_time_s,
        arrival_time_s,
    )
    departure_excess_speed = float(np.sqrt(transfer.departure_c3_m2_s2))
    injection = injection_delta_v_mps(
        departure_excess_speed,
        departure_body.gravitational_parameter_m3_s2,
        departure_parking_radius_m,
    )
    capture = injection_delta_v_mps(
        transfer.arrival_excess_speed_mps,
        arrival_body.gravitational_parameter_m3_s2,
        arrival_parking_radius_m,
    )
    c3_violation_speed = max(
        0.0,
        departure_excess_speed - np.sqrt(maximum_c3_m2_s2),
    )
    arrival_violation_speed = max(
        0.0,
        transfer.arrival_excess_speed_mps - maximum_arrival_excess_speed_mps,
    )
    penalty = float(100.0 * (c3_violation_speed + arrival_violation_speed))
    total = injection + capture
    return LaunchWindowCandidate(
        departure_time_s=departure_time_s,
        arrival_time_s=arrival_time_s,
        transfer=transfer,
        injection_delta_v_mps=injection,
        capture_delta_v_mps=capture,
        total_delta_v_mps=total,
        feasible=penalty == 0.0,
        constraint_penalty_mps=penalty,
        penalized_objective_mps=total + penalty,
    )


def optimize_launch_window(
    departure_body: CircularOrbitBody,
    arrival_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    *,
    departure_bounds_s: tuple[float, float],
    arrival_bounds_s: tuple[float, float],
    departure_grid_count: int,
    arrival_grid_count: int,
    departure_parking_altitude_m: float,
    arrival_parking_altitude_m: float,
    maximum_c3_m2_s2: float,
    maximum_arrival_excess_speed_mps: float,
    maximum_refinement_iterations: int = 20,
    epoch_tolerance_s: float = 60.0,
) -> LaunchWindowOptimization:
    """Find a constrained minimum injection-plus-capture opportunity.

    A complete rectangular grid supplies the global screen. A manually implemented
    bounded coordinate/pattern search then halves epoch steps until the declared
    tolerance is met. Invalid Lambert cells remain explicit NaNs.
    """
    values = np.array(
        [
            primary_mu_m3_s2,
            *departure_bounds_s,
            *arrival_bounds_s,
            departure_parking_altitude_m,
            arrival_parking_altitude_m,
            maximum_c3_m2_s2,
            maximum_arrival_excess_speed_mps,
            epoch_tolerance_s,
        ]
    )
    if not np.all(np.isfinite(values)) or primary_mu_m3_s2 <= 0.0:
        raise ValueError("launch-window scalar inputs must be finite and physical")
    if departure_bounds_s[1] <= departure_bounds_s[0] or arrival_bounds_s[1] <= arrival_bounds_s[0]:
        raise ValueError("launch-window epoch bounds must be increasing")
    if departure_bounds_s[0] < 0.0 or np.any(values[5:] <= 0.0):
        raise ValueError("launch-window altitudes, constraints, and tolerance must be positive")
    if departure_grid_count < 3 or arrival_grid_count < 3:
        raise ValueError("launch-window grid dimensions must each be at least three")
    if maximum_refinement_iterations <= 0:
        raise ValueError("launch-window refinement iteration limit must be positive")
    departure_grid = np.linspace(*departure_bounds_s, departure_grid_count)
    arrival_grid = np.linspace(*arrival_bounds_s, arrival_grid_count)
    total_grid = np.full((departure_grid_count, arrival_grid_count), np.nan)
    feasible_grid = np.zeros(total_grid.shape, dtype=np.bool_)
    departure_radius = departure_body.radius_m + departure_parking_altitude_m
    arrival_radius = arrival_body.radius_m + arrival_parking_altitude_m
    cache: dict[tuple[float, float], LaunchWindowCandidate | None] = {}

    def evaluate(departure_time_s: float, arrival_time_s: float) -> LaunchWindowCandidate | None:
        key = (float(departure_time_s), float(arrival_time_s))
        if key in cache:
            return cache[key]
        if arrival_time_s <= departure_time_s:
            cache[key] = None
            return None
        try:
            result = _candidate(
                departure_body,
                arrival_body,
                primary_mu_m3_s2,
                float(departure_time_s),
                float(arrival_time_s),
                departure_radius,
                arrival_radius,
                maximum_c3_m2_s2,
                maximum_arrival_excess_speed_mps,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            result = None
        cache[key] = result
        return result

    candidates: list[LaunchWindowCandidate] = []
    for departure_index, grid_departure_time_s in enumerate(departure_grid):
        for arrival_index, grid_arrival_time_s in enumerate(arrival_grid):
            result = evaluate(float(grid_departure_time_s), float(grid_arrival_time_s))
            if result is None:
                continue
            candidates.append(result)
            total_grid[departure_index, arrival_index] = result.total_delta_v_mps
            feasible_grid[departure_index, arrival_index] = result.feasible
    if not candidates:
        raise ValueError("launch-window grid produced no valid Lambert solution")
    feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
    current = min(
        feasible_candidates or candidates,
        key=lambda item: item.penalized_objective_mps,
    )
    history = [current]
    step_departure = float(departure_grid[1] - departure_grid[0])
    step_arrival = float(arrival_grid[1] - arrival_grid[0])
    for _iteration in range(maximum_refinement_iterations):
        if max(step_departure, step_arrival) <= epoch_tolerance_s:
            break
        neighbours: list[LaunchWindowCandidate] = []
        for departure_direction, arrival_direction in (
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        ):
            departure_time_s = float(
                np.clip(
                    current.departure_time_s + departure_direction * step_departure,
                    *departure_bounds_s,
                )
            )
            arrival_time_s = float(
                np.clip(
                    current.arrival_time_s + arrival_direction * step_arrival,
                    *arrival_bounds_s,
                )
            )
            result = evaluate(departure_time_s, arrival_time_s)
            if result is not None:
                neighbours.append(result)
        best = min(neighbours, key=lambda item: item.penalized_objective_mps)
        if best.penalized_objective_mps + 1.0e-9 < current.penalized_objective_mps:
            current = best
            history.append(current)
        else:
            step_departure *= 0.5
            step_arrival *= 0.5
    return LaunchWindowOptimization(
        departure_grid_s=np.asarray(departure_grid, dtype=np.float64),
        arrival_grid_s=np.asarray(arrival_grid, dtype=np.float64),
        total_delta_v_grid_mps=total_grid,
        feasible_grid=feasible_grid,
        optimum=current,
        history=tuple(history),
        evaluation_count=len(cache),
        converged=max(step_departure, step_arrival) <= epoch_tolerance_s,
    )
