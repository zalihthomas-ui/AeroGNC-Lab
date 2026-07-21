"""Lambert launch-window, porkchop, and multi-leg gravity-assist design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.astrodynamics.bplane import BPlaneTarget, target_b_plane
from aerognc.astrodynamics.lambert import LambertSolution, solve_lambert_universal
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class TransferOpportunity:
    """One Lambert transfer with planet-relative departure and arrival metrics."""

    departure_time_s: float
    arrival_time_s: float
    lambert: LambertSolution
    departure_excess_velocity_mps: FloatArray
    arrival_excess_velocity_mps: FloatArray
    departure_c3_m2_s2: float
    arrival_excess_speed_mps: float
    objective_mps: float


@dataclass(frozen=True, slots=True)
class PorkchopGrid:
    """Launch-window metrics on a departure/arrival epoch grid."""

    departure_time_s: FloatArray
    arrival_time_s: FloatArray
    departure_c3_m2_s2: FloatArray
    arrival_excess_speed_mps: FloatArray
    objective_mps: FloatArray
    feasible: npt.NDArray[np.bool_]

    def best_indices(self) -> tuple[int, int]:
        """Return the minimum finite objective as departure/arrival indices."""
        if not np.any(self.feasible):
            raise ValueError("porkchop grid contains no feasible transfer")
        masked = np.where(self.feasible, self.objective_mps, np.inf)
        flat_index = int(np.argmin(masked))
        return cast_index(np.unravel_index(flat_index, masked.shape))


@dataclass(frozen=True, slots=True)
class GravityAssistDesign:
    """Two Lambert legs joined by an evaluated planetary flyby."""

    departure_time_s: float
    assist_time_s: float
    arrival_time_s: float
    first_leg: TransferOpportunity
    second_leg: TransferOpportunity
    flyby: BPlaneTarget
    objective_mps: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class DifferentialCorrectionResult:
    """Finite-difference correction record for flyby encounter and arrival times."""

    design: GravityAssistDesign
    iterations: int
    residual_norm: float
    converged: bool


def cast_index(index: tuple[np.intp, ...]) -> tuple[int, int]:
    """Convert NumPy's two-dimensional unravel result to builtin integers."""
    if len(index) != 2:
        raise RuntimeError("porkchop grid index is not two-dimensional")
    return int(index[0]), int(index[1])


def evaluate_lambert_transfer(
    departure_body: CircularOrbitBody,
    arrival_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    departure_time_s: float,
    arrival_time_s: float,
    *,
    prograde: bool = True,
    long_way: bool = False,
) -> TransferOpportunity:
    """Solve and score one planet-to-planet zero-revolution Lambert transfer."""
    if not np.isfinite([departure_time_s, arrival_time_s]).all() or (
        arrival_time_s <= departure_time_s
    ):
        raise ValueError("arrival time must be finite and later than departure time")
    departure_position, departure_body_velocity = departure_body.state_at_time(
        departure_time_s, primary_mu_m3_s2
    )
    arrival_position, arrival_body_velocity = arrival_body.state_at_time(
        arrival_time_s, primary_mu_m3_s2
    )
    lambert = solve_lambert_universal(
        departure_position,
        arrival_position,
        arrival_time_s - departure_time_s,
        primary_mu_m3_s2,
        prograde=prograde,
        long_way=long_way,
    )
    departure_excess = lambert.departure_velocity_mps - departure_body_velocity
    arrival_excess = lambert.arrival_velocity_mps - arrival_body_velocity
    departure_speed = float(np.linalg.norm(departure_excess))
    arrival_speed = float(np.linalg.norm(arrival_excess))
    return TransferOpportunity(
        departure_time_s=float(departure_time_s),
        arrival_time_s=float(arrival_time_s),
        lambert=lambert,
        departure_excess_velocity_mps=departure_excess,
        arrival_excess_velocity_mps=arrival_excess,
        departure_c3_m2_s2=departure_speed**2,
        arrival_excess_speed_mps=arrival_speed,
        objective_mps=departure_speed + arrival_speed,
    )


def injection_delta_v_mps(
    excess_speed_mps: float,
    body_mu_m3_s2: float,
    parking_radius_m: float,
) -> float:
    """Return ideal impulsive injection/capture magnitude from a circular parking orbit."""
    values = np.array([excess_speed_mps, body_mu_m3_s2, parking_radius_m])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("injection inputs must be positive and finite")
    hyperbolic_periapsis_speed = np.sqrt(
        excess_speed_mps**2 + 2.0 * body_mu_m3_s2 / parking_radius_m
    )
    circular_speed = np.sqrt(body_mu_m3_s2 / parking_radius_m)
    return float(abs(hyperbolic_periapsis_speed - circular_speed))


def compute_porkchop_grid(
    departure_body: CircularOrbitBody,
    arrival_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    departure_time_s: npt.ArrayLike,
    arrival_time_s: npt.ArrayLike,
    *,
    maximum_c3_m2_s2: float = np.inf,
    maximum_arrival_excess_speed_mps: float = np.inf,
) -> PorkchopGrid:
    """Evaluate launch energy and arrival excess speed over an epoch grid."""
    departures = np.asarray(departure_time_s, dtype=np.float64)
    arrivals = np.asarray(arrival_time_s, dtype=np.float64)
    if departures.ndim != 1 or arrivals.ndim != 1 or departures.size < 2 or arrivals.size < 2:
        raise ValueError("porkchop epoch arrays must be one-dimensional with at least two values")
    if not np.all(np.isfinite(departures)) or not np.all(np.isfinite(arrivals)):
        raise ValueError("porkchop epochs must be finite")
    if maximum_c3_m2_s2 <= 0.0 or maximum_arrival_excess_speed_mps <= 0.0:
        raise ValueError("porkchop constraints must be positive")
    shape = (departures.size, arrivals.size)
    c3 = np.full(shape, np.nan)
    arrival_speed = np.full(shape, np.nan)
    objective = np.full(shape, np.nan)
    feasible = np.zeros(shape, dtype=np.bool_)
    for departure_index, departure_time in enumerate(departures):
        for arrival_index, arrival_time in enumerate(arrivals):
            if arrival_time <= departure_time:
                continue
            try:
                opportunity = evaluate_lambert_transfer(
                    departure_body,
                    arrival_body,
                    primary_mu_m3_s2,
                    float(departure_time),
                    float(arrival_time),
                )
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            c3[departure_index, arrival_index] = opportunity.departure_c3_m2_s2
            arrival_speed[departure_index, arrival_index] = opportunity.arrival_excess_speed_mps
            objective[departure_index, arrival_index] = opportunity.objective_mps
            feasible[departure_index, arrival_index] = (
                opportunity.departure_c3_m2_s2 <= maximum_c3_m2_s2
                and opportunity.arrival_excess_speed_mps <= maximum_arrival_excess_speed_mps
            )
    return PorkchopGrid(departures.copy(), arrivals.copy(), c3, arrival_speed, objective, feasible)


def design_gravity_assist(
    departure_body: CircularOrbitBody,
    assist_body: CircularOrbitBody,
    destination_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    departure_time_s: float,
    assist_time_s: float,
    arrival_time_s: float,
    *,
    minimum_flyby_altitude_m: float,
    excess_speed_tolerance_mps: float = 50.0,
) -> GravityAssistDesign:
    """Join two Lambert legs and evaluate their B-plane flyby compatibility."""
    if not departure_time_s < assist_time_s < arrival_time_s:
        raise ValueError("gravity-assist epochs must be strictly ordered")
    first_leg = evaluate_lambert_transfer(
        departure_body,
        assist_body,
        primary_mu_m3_s2,
        departure_time_s,
        assist_time_s,
    )
    second_leg = evaluate_lambert_transfer(
        assist_body,
        destination_body,
        primary_mu_m3_s2,
        assist_time_s,
        arrival_time_s,
    )
    _assist_position, assist_velocity = assist_body.state_at_time(assist_time_s, primary_mu_m3_s2)
    incoming_excess = first_leg.lambert.arrival_velocity_mps - assist_velocity
    outgoing_excess = second_leg.lambert.departure_velocity_mps - assist_velocity
    flyby = target_b_plane(
        incoming_excess,
        outgoing_excess,
        assist_body.gravitational_parameter_m3_s2,
        assist_body.radius_m,
        minimum_altitude_m=minimum_flyby_altitude_m,
        excess_speed_tolerance_mps=excess_speed_tolerance_mps,
    )
    altitude_violation_m = max(0.0, minimum_flyby_altitude_m - flyby.periapsis_altitude_m)
    objective = (
        np.sqrt(first_leg.departure_c3_m2_s2)
        + second_leg.arrival_excess_speed_mps
        + 20.0 * flyby.powered_flyby_delta_v_mps
        + altitude_violation_m / 10_000.0
    )
    return GravityAssistDesign(
        departure_time_s=departure_time_s,
        assist_time_s=assist_time_s,
        arrival_time_s=arrival_time_s,
        first_leg=first_leg,
        second_leg=second_leg,
        flyby=flyby,
        objective_mps=float(objective),
        feasible=flyby.feasible_unpowered,
    )


def search_gravity_assist(
    departure_body: CircularOrbitBody,
    assist_body: CircularOrbitBody,
    destination_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    departure_time_s: float,
    assist_time_candidates_s: npt.ArrayLike,
    arrival_time_candidates_s: npt.ArrayLike,
    *,
    minimum_flyby_altitude_m: float,
    excess_speed_tolerance_mps: float = 250.0,
) -> GravityAssistDesign:
    """Return the lowest-cost feasible grid member, or the lowest-cost member overall."""
    assist_candidates = np.asarray(assist_time_candidates_s, dtype=np.float64)
    arrival_candidates = np.asarray(arrival_time_candidates_s, dtype=np.float64)
    if assist_candidates.ndim != 1 or arrival_candidates.ndim != 1:
        raise ValueError("gravity-assist candidate epochs must be one-dimensional")
    designs: list[GravityAssistDesign] = []
    for assist_time in assist_candidates:
        for arrival_time in arrival_candidates:
            if not departure_time_s < assist_time < arrival_time:
                continue
            try:
                designs.append(
                    design_gravity_assist(
                        departure_body,
                        assist_body,
                        destination_body,
                        primary_mu_m3_s2,
                        departure_time_s,
                        float(assist_time),
                        float(arrival_time),
                        minimum_flyby_altitude_m=minimum_flyby_altitude_m,
                        excess_speed_tolerance_mps=excess_speed_tolerance_mps,
                    )
                )
            except (ValueError, RuntimeError, FloatingPointError):
                continue
    if not designs:
        raise ValueError("gravity-assist search produced no valid Lambert combinations")
    feasible = [design for design in designs if design.feasible]
    return min(feasible or designs, key=lambda design: design.objective_mps)


def refine_gravity_assist_times(
    initial_design: GravityAssistDesign,
    departure_body: CircularOrbitBody,
    assist_body: CircularOrbitBody,
    destination_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    *,
    target_flyby_altitude_m: float,
    minimum_flyby_altitude_m: float,
    maximum_iterations: int = 12,
    finite_difference_step_s: float = 43_200.0,
) -> DifferentialCorrectionResult:
    """Refine encounter/arrival epochs with finite-difference damped least squares."""
    times = np.array([initial_design.assist_time_s, initial_design.arrival_time_s])
    departure_time = initial_design.departure_time_s

    def evaluate(candidate: np.ndarray) -> tuple[GravityAssistDesign, np.ndarray]:
        design = design_gravity_assist(
            departure_body,
            assist_body,
            destination_body,
            primary_mu_m3_s2,
            departure_time,
            float(candidate[0]),
            float(candidate[1]),
            minimum_flyby_altitude_m=minimum_flyby_altitude_m,
            excess_speed_tolerance_mps=1.0e9,
        )
        residual = np.array(
            [
                design.flyby.powered_flyby_delta_v_mps / 100.0,
                (design.flyby.periapsis_altitude_m - target_flyby_altitude_m) / 1.0e7,
            ]
        )
        return design, residual

    current_design, residual = evaluate(times)
    converged = False
    iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        if float(np.linalg.norm(residual)) < 1.0e-3:
            converged = True
            break
        jacobian = np.empty((2, 2))
        for column in range(2):
            shifted = times.copy()
            shifted[column] += finite_difference_step_s
            try:
                _shifted_design, shifted_residual = evaluate(shifted)
            except (ValueError, RuntimeError, FloatingPointError):
                shifted[column] -= 2.0 * finite_difference_step_s
                _shifted_design, shifted_residual = evaluate(shifted)
                jacobian[:, column] = (residual - shifted_residual) / finite_difference_step_s
            else:
                jacobian[:, column] = (shifted_residual - residual) / finite_difference_step_s
        damping = 1.0e-5 * np.eye(2)
        correction = -np.linalg.solve(jacobian.T @ jacobian + damping, jacobian.T @ residual)
        correction = np.clip(correction, -30.0 * 86_400.0, 30.0 * 86_400.0)
        candidate_times = times + correction
        candidate_times[0] = max(candidate_times[0], departure_time + 86_400.0)
        candidate_times[1] = max(candidate_times[1], candidate_times[0] + 86_400.0)
        try:
            candidate_design, candidate_residual = evaluate(candidate_times)
        except (ValueError, RuntimeError, FloatingPointError):
            break
        if np.linalg.norm(candidate_residual) <= np.linalg.norm(residual):
            times = candidate_times
            current_design = candidate_design
            residual = candidate_residual
        else:
            break
    if float(np.linalg.norm(residual)) < 1.0e-3:
        converged = True
    return DifferentialCorrectionResult(
        current_design,
        iterations,
        float(np.linalg.norm(residual)),
        converged,
    )
