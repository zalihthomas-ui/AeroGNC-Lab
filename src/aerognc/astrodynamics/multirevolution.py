"""Direct multi-revolution Lambert enumeration, verification, and ranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.astrodynamics.kepler import propagate_universal, stumpff_c, stumpff_s
from aerognc.mathematics.vectors import FloatArray, as_vector

TransferDirection = Literal["prograde", "retrograde"]
TransferObjective = Literal["sum_v_infinity", "departure_v_infinity", "arrival_v_infinity"]
SolutionBranch = Literal["single", "lower-z", "upper-z"]


@dataclass(frozen=True, slots=True)
class MultiRevolutionLambertSolution:
    """One independently endpoint-verified Lambert branch."""

    departure_velocity_mps: FloatArray
    arrival_velocity_mps: FloatArray
    transfer_angle_rad: float
    time_of_flight_s: float
    revolutions: int
    direction: TransferDirection
    branch: SolutionBranch
    universal_parameter_z: float
    iterations: int
    endpoint_position_error_m: float
    endpoint_velocity_error_mps: float


@dataclass(frozen=True, slots=True)
class RankedTransferCandidate:
    """Lambert solution scored against departure and arrival reference velocities."""

    solution: MultiRevolutionLambertSolution
    departure_excess_velocity_mps: FloatArray
    arrival_excess_velocity_mps: FloatArray
    objective_mps: float
    rank: int


@dataclass(frozen=True, slots=True)
class MultiRevolutionSearchResult:
    """Feasible verified candidates in deterministic objective order."""

    candidates: tuple[RankedTransferCandidate, ...]
    attempted_geometry_count: int
    rejected_endpoint_count: int
    objective: TransferObjective

    @property
    def best(self) -> RankedTransferCandidate:
        """Return the minimum-objective candidate."""
        if not self.candidates:
            raise ValueError("multi-revolution search contains no feasible candidates")
        return self.candidates[0]


def _transfer_angle(
    position_1: FloatArray,
    position_2: FloatArray,
    direction: TransferDirection,
) -> float:
    cosine = float(
        np.clip(
            np.dot(position_1, position_2)
            / (np.linalg.norm(position_1) * np.linalg.norm(position_2)),
            -1.0,
            1.0,
        )
    )
    angle = float(np.arccos(cosine))
    cross_z = float(np.cross(position_1, position_2)[2])
    if abs(cross_z) <= 1.0e-14 * np.linalg.norm(position_1) * np.linalg.norm(position_2):
        raise ValueError("multi-revolution Lambert geometry is singular for collinear endpoints")
    if (direction == "prograde" and cross_z < 0.0) or (direction == "retrograde" and cross_z > 0.0):
        angle = 2.0 * np.pi - angle
    return angle


def _residual_function(
    radius_1_m: float,
    radius_2_m: float,
    parameter_a: float,
    square_root_mu: float,
    time_of_flight_s: float,
) -> Callable[[float], float]:
    def residual(z: float) -> float:
        c_value = stumpff_c(z)
        if c_value <= 0.0 or not np.isfinite(c_value):
            return np.nan
        s_value = stumpff_s(z)
        y_value = radius_1_m + radius_2_m + parameter_a * (z * s_value - 1.0) / np.sqrt(c_value)
        if y_value < 0.0 or not np.isfinite(y_value):
            return np.nan
        x_value = np.sqrt(y_value / c_value)
        calculated_time_s = (x_value**3 * s_value + parameter_a * np.sqrt(y_value)) / square_root_mu
        return float(calculated_time_s - time_of_flight_s)

    return residual


def _root_brackets(
    residual: Callable[[float], float],
    revolutions: int,
    *,
    samples: int = 6001,
) -> tuple[tuple[float, float], ...]:
    if revolutions == 0:
        lower = -16.0 * np.pi**2
        upper = 4.0 * np.pi**2
    else:
        lower = (2.0 * revolutions * np.pi) ** 2
        upper = (2.0 * (revolutions + 1) * np.pi) ** 2
    inset = max(1.0e-7, (upper - lower) * 1.0e-10)
    grid = np.linspace(lower + inset, upper - inset, samples)
    brackets: list[tuple[float, float]] = []
    previous_z: float | None = None
    previous_value: float | None = None
    for candidate in grid:
        z = float(candidate)
        value = residual(z)
        if not np.isfinite(value):
            previous_z = None
            previous_value = None
            continue
        if value == 0.0:
            brackets.append((z, z))
        elif previous_value is not None and np.signbit(value) != np.signbit(previous_value):
            if previous_z is None:
                raise RuntimeError("Lambert bracket state is inconsistent")
            brackets.append((previous_z, z))
        previous_z = z
        previous_value = value
    unique: list[tuple[float, float]] = []
    for bracket in brackets:
        midpoint = 0.5 * (bracket[0] + bracket[1])
        if not unique or abs(midpoint - 0.5 * sum(unique[-1])) > 1.0e-7:
            unique.append(bracket)
    if revolutions == 0 and unique:
        return (unique[0],)
    return tuple(unique[:2])


def _bisect_root(
    residual: Callable[[float], float],
    bracket: tuple[float, float],
    tolerance_s: float,
    maximum_iterations: int,
) -> tuple[float, int]:
    lower, upper = bracket
    if lower == upper:
        return lower, 0
    lower_value = residual(lower)
    for iteration in range(1, maximum_iterations + 1):
        middle = 0.5 * (lower + upper)
        middle_value = residual(middle)
        if not np.isfinite(middle_value):
            raise FloatingPointError("Lambert bisection entered an invalid branch")
        if abs(middle_value) <= tolerance_s:
            return middle, iteration
        if np.signbit(middle_value) == np.signbit(lower_value):
            lower = middle
            lower_value = middle_value
        else:
            upper = middle
    raise RuntimeError("multi-revolution Lambert bisection did not converge")


def _velocities_from_z(
    position_1: FloatArray,
    position_2: FloatArray,
    mu_m3_s2: float,
    parameter_a: float,
    z: float,
) -> tuple[FloatArray, FloatArray]:
    radius_1_m = float(np.linalg.norm(position_1))
    radius_2_m = float(np.linalg.norm(position_2))
    c_value = stumpff_c(z)
    s_value = stumpff_s(z)
    y_value = radius_1_m + radius_2_m + parameter_a * (z * s_value - 1.0) / np.sqrt(c_value)
    if y_value <= 0.0:
        raise FloatingPointError("Lambert branch produced a nonpositive y parameter")
    f_value = 1.0 - y_value / radius_1_m
    g_value = parameter_a * np.sqrt(y_value / mu_m3_s2)
    g_dot = 1.0 - y_value / radius_2_m
    if abs(g_value) <= 1.0e-14:
        raise FloatingPointError("Lambert branch produced singular g")
    return (
        (position_2 - f_value * position_1) / g_value,
        (g_dot * position_2 - position_1) / g_value,
    )


def solve_lambert_revolutions(
    departure_position_m: npt.ArrayLike,
    arrival_position_m: npt.ArrayLike,
    time_of_flight_s: float,
    gravitational_parameter_m3_s2: float,
    *,
    revolutions: int,
    direction: TransferDirection,
    time_tolerance_s: float = 1.0e-8,
    endpoint_tolerance_m: float = 1.0,
    maximum_iterations: int = 160,
) -> tuple[MultiRevolutionLambertSolution, ...]:
    """Enumerate and independently propagate every branch for one geometry."""
    position_1 = as_vector(departure_position_m, 3, name="departure_position_m")
    position_2 = as_vector(arrival_position_m, 3, name="arrival_position_m")
    values = np.array(
        [time_of_flight_s, gravitational_parameter_m3_s2, time_tolerance_s, endpoint_tolerance_m]
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Lambert time, gravitational parameter, and tolerances must be positive")
    if isinstance(revolutions, bool) or revolutions < 0 or revolutions > 12:
        raise ValueError("revolutions must be an integer in [0, 12]")
    if direction not in {"prograde", "retrograde"}:
        raise ValueError("direction must be prograde or retrograde")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    radius_1_m = float(np.linalg.norm(position_1))
    radius_2_m = float(np.linalg.norm(position_2))
    if radius_1_m <= 0.0 or radius_2_m <= 0.0:
        raise ValueError("Lambert endpoint radii must be positive")
    angle = _transfer_angle(position_1, position_2, direction)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    parameter_a = sine * np.sqrt(radius_1_m * radius_2_m / (1.0 - cosine))
    residual = _residual_function(
        radius_1_m,
        radius_2_m,
        parameter_a,
        float(np.sqrt(gravitational_parameter_m3_s2)),
        time_of_flight_s,
    )
    brackets = _root_brackets(residual, revolutions)
    solutions: list[MultiRevolutionLambertSolution] = []
    roots: list[tuple[float, int]] = [
        _bisect_root(residual, bracket, time_tolerance_s, maximum_iterations)
        for bracket in brackets
    ]
    for root_index, (z, iterations) in enumerate(roots):
        departure_velocity, arrival_velocity = _velocities_from_z(
            position_1,
            position_2,
            gravitational_parameter_m3_s2,
            parameter_a,
            z,
        )
        propagated = propagate_universal(
            position_1,
            departure_velocity,
            time_of_flight_s,
            gravitational_parameter_m3_s2,
        )
        position_error_m = float(np.linalg.norm(propagated.position_m - position_2))
        velocity_error_mps = float(np.linalg.norm(propagated.velocity_mps - arrival_velocity))
        if position_error_m > endpoint_tolerance_m:
            continue
        branch: SolutionBranch = (
            "single" if revolutions == 0 else "lower-z" if root_index == 0 else "upper-z"
        )
        solutions.append(
            MultiRevolutionLambertSolution(
                departure_velocity,
                arrival_velocity,
                angle,
                time_of_flight_s,
                revolutions,
                direction,
                branch,
                z,
                iterations,
                position_error_m,
                velocity_error_mps,
            )
        )
    return tuple(solutions)


def search_lambert_transfers(
    departure_position_m: npt.ArrayLike,
    departure_reference_velocity_mps: npt.ArrayLike,
    arrival_position_m: npt.ArrayLike,
    arrival_reference_velocity_mps: npt.ArrayLike,
    time_of_flight_s: float,
    gravitational_parameter_m3_s2: float,
    *,
    revolutions: Sequence[int] = (0,),
    directions: Sequence[TransferDirection] = ("prograde", "retrograde"),
    objective: TransferObjective = "sum_v_infinity",
    endpoint_tolerance_m: float = 1.0,
) -> MultiRevolutionSearchResult:
    """Enumerate, verify, score, and deterministically rank transfer branches."""
    unique_revolutions = tuple(sorted(set(revolutions)))
    if len(unique_revolutions) != len(tuple(revolutions)) or not unique_revolutions:
        raise ValueError("revolutions must be a nonempty unique sequence")
    unique_directions = tuple(dict.fromkeys(directions))
    if len(unique_directions) != len(tuple(directions)) or not unique_directions:
        raise ValueError("directions must be a nonempty unique sequence")
    if objective not in {
        "sum_v_infinity",
        "departure_v_infinity",
        "arrival_v_infinity",
    }:
        raise ValueError("unsupported Lambert ranking objective")
    departure_reference = as_vector(
        departure_reference_velocity_mps, 3, name="departure_reference_velocity_mps"
    )
    arrival_reference = as_vector(
        arrival_reference_velocity_mps, 3, name="arrival_reference_velocity_mps"
    )
    scored: list[tuple[MultiRevolutionLambertSolution, FloatArray, FloatArray, float]] = []
    attempted = 0
    rejected = 0
    for revolution_count in unique_revolutions:
        for direction in unique_directions:
            attempted += 1
            solutions = solve_lambert_revolutions(
                departure_position_m,
                arrival_position_m,
                time_of_flight_s,
                gravitational_parameter_m3_s2,
                revolutions=revolution_count,
                direction=direction,
                endpoint_tolerance_m=endpoint_tolerance_m,
            )
            expected_branches = 1 if revolution_count == 0 else 2
            rejected += expected_branches - len(solutions)
            for solution in solutions:
                departure_excess = solution.departure_velocity_mps - departure_reference
                arrival_excess = solution.arrival_velocity_mps - arrival_reference
                departure_speed = float(np.linalg.norm(departure_excess))
                arrival_speed = float(np.linalg.norm(arrival_excess))
                if objective == "departure_v_infinity":
                    score = departure_speed
                elif objective == "arrival_v_infinity":
                    score = arrival_speed
                else:
                    score = departure_speed + arrival_speed
                scored.append((solution, departure_excess, arrival_excess, score))
    direction_order = {"prograde": 0, "retrograde": 1}
    branch_order = {"single": 0, "lower-z": 1, "upper-z": 2}
    scored.sort(
        key=lambda item: (
            item[3],
            item[0].revolutions,
            direction_order[item[0].direction],
            branch_order[item[0].branch],
            item[0].universal_parameter_z,
        )
    )
    candidates = tuple(
        RankedTransferCandidate(solution, departure_excess, arrival_excess, score, rank)
        for rank, (solution, departure_excess, arrival_excess, score) in enumerate(scored, start=1)
    )
    return MultiRevolutionSearchResult(candidates, attempted, rejected, objective)
