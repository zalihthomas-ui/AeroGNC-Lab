"""Rendezvous and proximity-operations (RPO) relative-orbit dynamics.

This module models how one spacecraft (the *chaser*) moves relative to another
(the *target*) in a near-circular orbit, using the linear Clohessy-Wiltshire
(Hill) equations in the target's LVLH frame (x = radial/outward, y = along-track,
z = cross-track). It provides:

* the CW state-transition matrix and propagation,
* a two-impulse rendezvous solver that drives the chaser to a specified hold
  point (e.g. a V-bar or R-bar station-keeping point) with a chosen arrival
  velocity, and a multi-leg approach through a corridor of hold points,
* closest-approach (conjunction) reporting over a horizon, and
* a helper showing how a single impulsive burn changes the orbit
  (semi-major axis / apoapsis / periapsis).

**Scope (public-safety).** This is cooperative rendezvous, inspection, and
station-keeping mathematics — the same used for docking, servicing, and
debris-avoidance. It contains no interception-to-destroy, terminal-homing, or
engagement logic, consistent with the project's public-safety posture.
"""

from dataclasses import dataclass

import numpy as np

from aerognc.astrodynamics.orbital_elements import state_to_elements
from aerognc.mathematics.vectors import FloatArray, as_vector

EARTH_MU_M3_S2 = 3.986004418e14


def mean_motion_radps(gravitational_parameter_m3_s2: float, semi_major_axis_m: float) -> float:
    """Return the orbital mean motion ``n = sqrt(mu / a^3)`` [rad/s]."""
    if gravitational_parameter_m3_s2 <= 0.0 or semi_major_axis_m <= 0.0:
        raise ValueError("gravitational parameter and semi-major axis must be positive")
    return float(np.sqrt(gravitational_parameter_m3_s2 / semi_major_axis_m**3))


def cw_state_transition(mean_motion_n: float, time_s: float) -> FloatArray:
    """Return the 6x6 Clohessy-Wiltshire state-transition matrix.

    State ordering is ``[x, y, z, xdot, ydot, zdot]`` in the LVLH frame.
    """
    if mean_motion_n <= 0.0:
        raise ValueError("mean_motion_n must be positive")
    if not np.isfinite(time_s):
        raise ValueError("time_s must be finite")
    n = mean_motion_n
    s = np.sin(n * time_s)
    c = np.cos(n * time_s)
    nt = n * time_s

    phi_rr = np.array([[4.0 - 3.0 * c, 0.0, 0.0], [6.0 * (s - nt), 1.0, 0.0], [0.0, 0.0, c]])
    phi_rv = np.array(
        [
            [s / n, 2.0 * (1.0 - c) / n, 0.0],
            [-2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, s / n],
        ]
    )
    phi_vr = np.array(
        [[3.0 * n * s, 0.0, 0.0], [-6.0 * n * (1.0 - c), 0.0, 0.0], [0.0, 0.0, -n * s]]
    )
    phi_vv = np.array([[c, 2.0 * s, 0.0], [-2.0 * s, 4.0 * c - 3.0, 0.0], [0.0, 0.0, c]])

    stm = np.zeros((6, 6), dtype=np.float64)
    stm[:3, :3] = phi_rr
    stm[:3, 3:] = phi_rv
    stm[3:, :3] = phi_vr
    stm[3:, 3:] = phi_vv
    return stm


@dataclass(frozen=True, slots=True)
class ImpulsiveBurn:
    """A single impulsive relative-velocity change [m/s] at a time [s]."""

    time_s: float
    delta_v_lvlh_mps: FloatArray


@dataclass(frozen=True, slots=True)
class RendezvousLeg:
    """Result of a two-impulse transfer to a hold point."""

    departure_burn: ImpulsiveBurn
    arrival_burn: ImpulsiveBurn
    arrival_state_lvlh: FloatArray  # [x,y,z,xdot,ydot,zdot]
    total_delta_v_mps: float


@dataclass(frozen=True, slots=True)
class RelativeTrajectory:
    """Sampled relative trajectory plus burns and closest-approach info."""

    time_s: FloatArray
    states_lvlh: FloatArray  # (N, 6)
    burns: tuple[ImpulsiveBurn, ...]
    total_delta_v_mps: float
    closest_approach_m: float
    closest_approach_time_s: float


class ClohessyWiltshireModel:
    """Linear relative-orbit model in the target's LVLH frame."""

    def __init__(self, mean_motion_n: float) -> None:
        if mean_motion_n <= 0.0:
            raise ValueError("mean_motion_n must be positive")
        self.mean_motion_n = mean_motion_n

    @classmethod
    def from_orbit(
        cls,
        semi_major_axis_m: float,
        gravitational_parameter_m3_s2: float = EARTH_MU_M3_S2,
    ) -> "ClohessyWiltshireModel":
        """Build from the target's circular-orbit radius/semi-major axis."""
        return cls(mean_motion_radps(gravitational_parameter_m3_s2, semi_major_axis_m))

    def propagate(self, state_lvlh: FloatArray, time_s: float) -> FloatArray:
        """Propagate a relative state by ``time_s`` (no burns)."""
        state = as_vector(state_lvlh, 6, name="state_lvlh")
        return cw_state_transition(self.mean_motion_n, time_s) @ state

    def two_impulse_rendezvous(
        self,
        initial_state_lvlh: FloatArray,
        target_position_lvlh_m: FloatArray,
        transfer_time_s: float,
        *,
        arrival_velocity_lvlh_mps: FloatArray | None = None,
    ) -> RendezvousLeg:
        """Solve the two-burn transfer to a hold point.

        Burn 1 sets the departure velocity so the chaser arrives at
        ``target_position`` after ``transfer_time_s``; burn 2 matches the desired
        arrival velocity (zero by default, i.e. station-keeping at the hold point).
        """
        if transfer_time_s <= 0.0:
            raise ValueError("transfer_time_s must be positive")
        state0 = as_vector(initial_state_lvlh, 6, name="initial_state_lvlh")
        target_position = as_vector(target_position_lvlh_m, 3, name="target_position_lvlh_m")
        arrival_velocity = (
            np.zeros(3)
            if arrival_velocity_lvlh_mps is None
            else as_vector(arrival_velocity_lvlh_mps, 3, name="arrival_velocity_lvlh_mps")
        )

        stm = cw_state_transition(self.mean_motion_n, transfer_time_s)
        phi_rr, phi_rv = stm[:3, :3], stm[:3, 3:]
        phi_vr, phi_vv = stm[3:, :3], stm[3:, 3:]

        r0, v0 = state0[:3], state0[3:]
        required_v0 = np.linalg.solve(phi_rv, target_position - phi_rr @ r0)
        delta_v1 = required_v0 - v0

        arrival_velocity_before = phi_vr @ r0 + phi_vv @ required_v0
        delta_v2 = arrival_velocity - arrival_velocity_before

        arrival_state = np.concatenate((target_position, arrival_velocity))
        total = float(np.linalg.norm(delta_v1) + np.linalg.norm(delta_v2))
        return RendezvousLeg(
            departure_burn=ImpulsiveBurn(0.0, delta_v1),
            arrival_burn=ImpulsiveBurn(transfer_time_s, delta_v2),
            arrival_state_lvlh=arrival_state,
            total_delta_v_mps=total,
        )

    def closest_approach(
        self, state_lvlh: FloatArray, horizon_s: float, *, samples: int = 400
    ) -> tuple[float, float]:
        """Return (minimum separation [m], time [s]) over the horizon (no burns)."""
        if horizon_s <= 0.0 or samples < 2:
            raise ValueError("horizon_s must be positive and samples >= 2")
        times = np.linspace(0.0, horizon_s, samples)
        ranges = [float(np.linalg.norm(self.propagate(state_lvlh, t)[:3])) for t in times]
        index = int(np.argmin(ranges))
        return ranges[index], float(times[index])


def simulate_rendezvous(
    model: ClohessyWiltshireModel,
    initial_state_lvlh: FloatArray,
    hold_points_lvlh_m: list[FloatArray],
    *,
    leg_time_s: float,
    samples_per_leg: int = 120,
) -> RelativeTrajectory:
    """Fly a multi-leg two-impulse approach through a corridor of hold points.

    Each leg is a two-impulse transfer to the next hold point with zero arrival
    velocity (station-keep), producing a safe stepped approach (e.g. along the
    V-bar). Returns the sampled trajectory, the burns, total delta-v, and the
    closest approach to the target (origin) across the whole profile.
    """
    if leg_time_s <= 0.0:
        raise ValueError("leg_time_s must be positive")
    if not hold_points_lvlh_m:
        raise ValueError("at least one hold point is required")

    state = as_vector(initial_state_lvlh, 6, name="initial_state_lvlh")
    all_times: list[float] = []
    all_states: list[FloatArray] = []
    burns: list[ImpulsiveBurn] = []
    total_delta_v = 0.0
    elapsed = 0.0
    min_range = float(np.linalg.norm(state[:3]))
    min_time = 0.0

    for hold_point in hold_points_lvlh_m:
        leg = model.two_impulse_rendezvous(state, hold_point, leg_time_s)
        burns.append(ImpulsiveBurn(elapsed, leg.departure_burn.delta_v_lvlh_mps))
        total_delta_v += float(np.linalg.norm(leg.departure_burn.delta_v_lvlh_mps))
        # Apply departure burn, then sample the coast to the hold point.
        state_after_burn = state.copy()
        state_after_burn[3:] = state[3:] + leg.departure_burn.delta_v_lvlh_mps
        leg_times = np.linspace(0.0, leg_time_s, samples_per_leg)
        for t in leg_times:
            propagated = model.propagate(state_after_burn, float(t))
            all_times.append(elapsed + float(t))
            all_states.append(propagated)
            separation = float(np.linalg.norm(propagated[:3]))
            if separation < min_range:
                min_range, min_time = separation, elapsed + float(t)
        elapsed += leg_time_s
        burns.append(ImpulsiveBurn(elapsed, leg.arrival_burn.delta_v_lvlh_mps))
        total_delta_v += float(np.linalg.norm(leg.arrival_burn.delta_v_lvlh_mps))
        state = leg.arrival_state_lvlh

    return RelativeTrajectory(
        time_s=np.asarray(all_times, dtype=np.float64),
        states_lvlh=np.asarray(all_states, dtype=np.float64),
        burns=tuple(burns),
        total_delta_v_mps=total_delta_v,
        closest_approach_m=min_range,
        closest_approach_time_s=min_time,
    )


@dataclass(frozen=True, slots=True)
class OrbitChange:
    """Before/after orbit summary for an impulsive burn."""

    semi_major_axis_before_m: float
    semi_major_axis_after_m: float
    eccentricity_before: float
    eccentricity_after: float
    apoapsis_altitude_before_m: float
    apoapsis_altitude_after_m: float
    periapsis_altitude_before_m: float
    periapsis_altitude_after_m: float


def orbit_change_from_impulse(
    position_eci_m: FloatArray,
    velocity_eci_mps: FloatArray,
    delta_v_eci_mps: FloatArray,
    *,
    gravitational_parameter_m3_s2: float = EARTH_MU_M3_S2,
    body_radius_m: float = 6_378_137.0,
) -> OrbitChange:
    """Show how altitude/orbit changes when an impulsive burn is introduced.

    Converts the before/after Cartesian states to classical elements and reports
    the change in semi-major axis, eccentricity, and apoapsis/periapsis altitude.
    """
    position = as_vector(position_eci_m, 3, name="position_eci_m")
    velocity = as_vector(velocity_eci_mps, 3, name="velocity_eci_mps")
    delta_v = as_vector(delta_v_eci_mps, 3, name="delta_v_eci_mps")

    before = state_to_elements(position, velocity, gravitational_parameter_m3_s2)
    after = state_to_elements(position, velocity + delta_v, gravitational_parameter_m3_s2)

    def apses(elements: object) -> tuple[float, float]:
        a = elements.semi_major_axis_m  # type: ignore[attr-defined]
        e = elements.eccentricity  # type: ignore[attr-defined]
        return a * (1.0 + e), a * (1.0 - e)

    apoapsis_before, periapsis_before = apses(before)
    apoapsis_after, periapsis_after = apses(after)
    return OrbitChange(
        semi_major_axis_before_m=before.semi_major_axis_m,
        semi_major_axis_after_m=after.semi_major_axis_m,
        eccentricity_before=before.eccentricity,
        eccentricity_after=after.eccentricity,
        apoapsis_altitude_before_m=apoapsis_before - body_radius_m,
        apoapsis_altitude_after_m=apoapsis_after - body_radius_m,
        periapsis_altitude_before_m=periapsis_before - body_radius_m,
        periapsis_altitude_after_m=periapsis_after - body_radius_m,
    )
