"""Inspectable sphere-of-influence patches and orbit-assisted transfer budgets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.astrodynamics.maneuvers import ideal_propellant_used_kg
from aerognc.astrodynamics.mission_design import (
    TransferOpportunity,
    evaluate_lambert_transfer,
    injection_delta_v_mps,
)
from aerognc.astrodynamics.perturbations import laplace_sphere_of_influence_radius_m
from aerognc.mathematics.vectors import FloatArray

GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11


@dataclass(frozen=True, slots=True)
class HyperbolicPatch:
    """One ideal body-centred hyperbolic branch between periapsis and the SOI."""

    excess_speed_mps: float
    periapsis_radius_m: float
    sphere_of_influence_radius_m: float
    eccentricity: float
    semi_major_axis_m: float
    periapsis_speed_mps: float
    sphere_of_influence_speed_mps: float
    time_periapsis_to_soi_s: float
    true_anomaly_at_soi_rad: float
    asymptote_true_anomaly_rad: float
    total_unpowered_turn_angle_rad: float


def hyperbolic_patch(
    excess_speed_mps: float,
    body_mu_m3_s2: float,
    periapsis_radius_m: float,
    sphere_of_influence_radius_m: float,
) -> HyperbolicPatch:
    """Derive a two-body hyperbola and its finite SOI crossing analytically."""
    values = np.array(
        [excess_speed_mps, body_mu_m3_s2, periapsis_radius_m, sphere_of_influence_radius_m]
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("hyperbolic-patch inputs must be positive and finite")
    if sphere_of_influence_radius_m <= periapsis_radius_m:
        raise ValueError("sphere of influence must lie outside hyperbolic periapsis")
    semi_major_axis_m = -body_mu_m3_s2 / excess_speed_mps**2
    semi_major_axis_magnitude = -semi_major_axis_m
    eccentricity = 1.0 + periapsis_radius_m * excess_speed_mps**2 / body_mu_m3_s2
    semi_latus_rectum_m = periapsis_radius_m * (1.0 + eccentricity)
    cosine_true_anomaly = (semi_latus_rectum_m / sphere_of_influence_radius_m - 1.0) / eccentricity
    if not -1.0 <= cosine_true_anomaly <= 1.0:
        raise ValueError("configured sphere of influence does not intersect the hyperbola")
    true_anomaly = float(np.arccos(np.clip(cosine_true_anomaly, -1.0, 1.0)))
    hyperbolic_anomaly = float(
        np.arccosh((sphere_of_influence_radius_m / semi_major_axis_magnitude + 1.0) / eccentricity)
    )
    hyperbolic_mean_anomaly = eccentricity * np.sinh(hyperbolic_anomaly) - hyperbolic_anomaly
    time_to_soi_s = float(
        np.sqrt(semi_major_axis_magnitude**3 / body_mu_m3_s2) * hyperbolic_mean_anomaly
    )
    periapsis_speed = float(np.sqrt(excess_speed_mps**2 + 2.0 * body_mu_m3_s2 / periapsis_radius_m))
    soi_speed = float(
        np.sqrt(excess_speed_mps**2 + 2.0 * body_mu_m3_s2 / sphere_of_influence_radius_m)
    )
    asymptote_true_anomaly = float(np.arccos(-1.0 / eccentricity))
    return HyperbolicPatch(
        excess_speed_mps=float(excess_speed_mps),
        periapsis_radius_m=float(periapsis_radius_m),
        sphere_of_influence_radius_m=float(sphere_of_influence_radius_m),
        eccentricity=float(eccentricity),
        semi_major_axis_m=float(semi_major_axis_m),
        periapsis_speed_mps=periapsis_speed,
        sphere_of_influence_speed_mps=soi_speed,
        time_periapsis_to_soi_s=time_to_soi_s,
        true_anomaly_at_soi_rad=true_anomaly,
        asymptote_true_anomaly_rad=asymptote_true_anomaly,
        total_unpowered_turn_angle_rad=float(2.0 * np.arcsin(1.0 / eccentricity)),
    )


@dataclass(frozen=True, slots=True)
class TourBurn:
    """One ideal impulsive maneuver and sequential mass state."""

    name: str
    epoch_s: float
    delta_v_mps: float
    mass_before_kg: float
    propellant_used_kg: float
    mass_after_kg: float


@dataclass(frozen=True, slots=True)
class OrbitAssistedTour:
    """Two Lambert legs joined by capture, orbit dwell, and powered departure."""

    departure_body: CircularOrbitBody
    assist_body: CircularOrbitBody
    destination_body: CircularOrbitBody
    departure_time_s: float
    assist_arrival_time_s: float
    assist_departure_time_s: float
    destination_arrival_time_s: float
    first_leg: TransferOpportunity
    second_leg: TransferOpportunity
    departure_patch: HyperbolicPatch
    assist_arrival_patch: HyperbolicPatch
    assist_departure_patch: HyperbolicPatch
    destination_patch: HyperbolicPatch
    assist_parking_radius_m: float
    assist_circular_speed_mps: float
    assist_orbit_period_s: float
    dwell_revolutions: int
    asymptote_alignment_angle_rad: float
    alignment_delta_v_mps: float
    burns: tuple[TourBurn, ...]
    total_delta_v_mps: float
    final_mass_kg: float
    dry_mass_kg: float
    departure_oberth_energy_gain_jpkg: float
    feasible: bool


def _angle_between(first: FloatArray, second: FloatArray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        raise ValueError("excess-velocity directions must have nonzero magnitude")
    return float(np.arccos(np.clip(float(first @ second) / denominator, -1.0, 1.0)))


def _append_burn(
    burns: list[TourBurn],
    name: str,
    epoch_s: float,
    delta_v_mps: float,
    mass_kg: float,
    specific_impulse_s: float,
) -> float:
    used_kg = ideal_propellant_used_kg(mass_kg, delta_v_mps, specific_impulse_s)
    after_kg = mass_kg - used_kg
    burns.append(TourBurn(name, epoch_s, delta_v_mps, mass_kg, used_kg, after_kg))
    return after_kg


def plan_orbit_assisted_tour(
    departure_body: CircularOrbitBody,
    assist_body: CircularOrbitBody,
    destination_body: CircularOrbitBody,
    primary_mu_m3_s2: float,
    primary_mass_kg: float,
    departure_time_s: float,
    assist_arrival_time_s: float,
    destination_arrival_time_s: float,
    *,
    departure_parking_altitude_m: float,
    assist_parking_altitude_m: float,
    destination_parking_altitude_m: float,
    dwell_revolutions: int,
    initial_mass_kg: float,
    dry_mass_kg: float,
    specific_impulse_s: float,
) -> OrbitAssistedTour:
    """Plan an ideal capture-dwell-departure tour with explicit mass accounting.

    The angle between incoming and outgoing excess-velocity vectors is charged as a
    conservative impulsive circular-orbit alignment maneuver. This prevents a hidden
    free change of parking-orbit plane in the preliminary patched-conic model.
    """
    scalar_values = np.array(
        [
            primary_mu_m3_s2,
            primary_mass_kg,
            departure_time_s,
            assist_arrival_time_s,
            destination_arrival_time_s,
            departure_parking_altitude_m,
            assist_parking_altitude_m,
            destination_parking_altitude_m,
            initial_mass_kg,
            dry_mass_kg,
            specific_impulse_s,
        ]
    )
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("orbit-assisted tour inputs must be finite")
    if primary_mu_m3_s2 <= 0.0 or primary_mass_kg <= 0.0:
        raise ValueError("primary gravity and mass must be positive")
    if not departure_time_s < assist_arrival_time_s < destination_arrival_time_s:
        raise ValueError("tour encounter epochs must be strictly ordered")
    if np.any(scalar_values[5:] <= 0.0) or dry_mass_kg > initial_mass_kg:
        raise ValueError("parking altitudes, masses, and specific impulse must be physical")
    if dwell_revolutions < 1:
        raise ValueError("assist dwell must contain at least one complete parking orbit")

    assist_parking_radius_m = assist_body.radius_m + assist_parking_altitude_m
    assist_circular_speed_mps = float(
        np.sqrt(assist_body.gravitational_parameter_m3_s2 / assist_parking_radius_m)
    )
    assist_period_s = float(
        2.0
        * np.pi
        * np.sqrt(assist_parking_radius_m**3 / assist_body.gravitational_parameter_m3_s2)
    )
    assist_departure_time_s = assist_arrival_time_s + dwell_revolutions * assist_period_s
    if assist_departure_time_s >= destination_arrival_time_s:
        raise ValueError("assist parking dwell leaves no time for the destination leg")

    first_leg = evaluate_lambert_transfer(
        departure_body,
        assist_body,
        primary_mu_m3_s2,
        departure_time_s,
        assist_arrival_time_s,
    )
    second_leg = evaluate_lambert_transfer(
        assist_body,
        destination_body,
        primary_mu_m3_s2,
        assist_departure_time_s,
        destination_arrival_time_s,
    )
    departure_v_inf = float(np.linalg.norm(first_leg.departure_excess_velocity_mps))
    assist_incoming_v_inf = float(np.linalg.norm(first_leg.arrival_excess_velocity_mps))
    assist_outgoing_v_inf = float(np.linalg.norm(second_leg.departure_excess_velocity_mps))
    destination_v_inf = second_leg.arrival_excess_speed_mps

    def soi(body: CircularOrbitBody) -> float:
        return laplace_sphere_of_influence_radius_m(
            body.semi_major_axis_m,
            body.gravitational_parameter_m3_s2 / GRAVITATIONAL_CONSTANT_M3_KG_S2,
            primary_mass_kg,
        )

    departure_radius_m = departure_body.radius_m + departure_parking_altitude_m
    destination_radius_m = destination_body.radius_m + destination_parking_altitude_m
    departure_patch = hyperbolic_patch(
        departure_v_inf,
        departure_body.gravitational_parameter_m3_s2,
        departure_radius_m,
        soi(departure_body),
    )
    assist_arrival_patch = hyperbolic_patch(
        assist_incoming_v_inf,
        assist_body.gravitational_parameter_m3_s2,
        assist_parking_radius_m,
        soi(assist_body),
    )
    assist_departure_patch = hyperbolic_patch(
        assist_outgoing_v_inf,
        assist_body.gravitational_parameter_m3_s2,
        assist_parking_radius_m,
        soi(assist_body),
    )
    destination_patch = hyperbolic_patch(
        destination_v_inf,
        destination_body.gravitational_parameter_m3_s2,
        destination_radius_m,
        soi(destination_body),
    )

    alignment_angle_rad = _angle_between(
        first_leg.arrival_excess_velocity_mps,
        second_leg.departure_excess_velocity_mps,
    )
    alignment_delta_v_mps = float(
        2.0 * assist_circular_speed_mps * np.sin(0.5 * alignment_angle_rad)
    )
    departure_delta_v_mps = injection_delta_v_mps(
        departure_v_inf,
        departure_body.gravitational_parameter_m3_s2,
        departure_radius_m,
    )
    assist_capture_delta_v_mps = (
        assist_arrival_patch.periapsis_speed_mps - assist_circular_speed_mps
    )
    assist_departure_delta_v_mps = (
        assist_departure_patch.periapsis_speed_mps - assist_circular_speed_mps
    )
    destination_circular_speed_mps = float(
        np.sqrt(destination_body.gravitational_parameter_m3_s2 / destination_radius_m)
    )
    destination_capture_delta_v_mps = (
        destination_patch.periapsis_speed_mps - destination_circular_speed_mps
    )

    burns: list[TourBurn] = []
    mass_kg = initial_mass_kg
    mass_kg = _append_burn(
        burns,
        "departure_parking_orbit_injection",
        departure_time_s,
        departure_delta_v_mps,
        mass_kg,
        specific_impulse_s,
    )
    mass_kg = _append_burn(
        burns,
        "assist_orbit_capture",
        assist_arrival_time_s,
        assist_capture_delta_v_mps,
        mass_kg,
        specific_impulse_s,
    )
    mass_kg = _append_burn(
        burns,
        "assist_orbit_alignment",
        assist_arrival_time_s + 0.5 * dwell_revolutions * assist_period_s,
        alignment_delta_v_mps,
        mass_kg,
        specific_impulse_s,
    )
    mass_kg = _append_burn(
        burns,
        "assist_periapsis_departure",
        assist_departure_time_s,
        assist_departure_delta_v_mps,
        mass_kg,
        specific_impulse_s,
    )
    mass_kg = _append_burn(
        burns,
        "destination_orbit_capture",
        destination_arrival_time_s,
        destination_capture_delta_v_mps,
        mass_kg,
        specific_impulse_s,
    )
    total_delta_v_mps = float(sum(burn.delta_v_mps for burn in burns))
    departure_oberth_gain = float(
        assist_circular_speed_mps * assist_departure_delta_v_mps
        + 0.5 * assist_departure_delta_v_mps**2
    )
    return OrbitAssistedTour(
        departure_body=departure_body,
        assist_body=assist_body,
        destination_body=destination_body,
        departure_time_s=departure_time_s,
        assist_arrival_time_s=assist_arrival_time_s,
        assist_departure_time_s=assist_departure_time_s,
        destination_arrival_time_s=destination_arrival_time_s,
        first_leg=first_leg,
        second_leg=second_leg,
        departure_patch=departure_patch,
        assist_arrival_patch=assist_arrival_patch,
        assist_departure_patch=assist_departure_patch,
        destination_patch=destination_patch,
        assist_parking_radius_m=assist_parking_radius_m,
        assist_circular_speed_mps=assist_circular_speed_mps,
        assist_orbit_period_s=assist_period_s,
        dwell_revolutions=dwell_revolutions,
        asymptote_alignment_angle_rad=alignment_angle_rad,
        alignment_delta_v_mps=alignment_delta_v_mps,
        burns=tuple(burns),
        total_delta_v_mps=total_delta_v_mps,
        final_mass_kg=mass_kg,
        dry_mass_kg=dry_mass_kg,
        departure_oberth_energy_gain_jpkg=departure_oberth_gain,
        feasible=mass_kg >= dry_mass_kg,
    )
