"""Patched-conic capture, parking-orbit dwell, and departure simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.astrodynamics.kepler import propagate_universal
from aerognc.astrodynamics.patched_conics import (
    GRAVITATIONAL_CONSTANT_M3_KG_S2,
    OrbitAssistedTour,
    plan_orbit_assisted_tour,
)
from aerognc.configuration.orbit_tour_loader import OrbitTourConfiguration
from aerognc.mathematics.integrators import EventOccurrence
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.logging import SimulationResult, write_result_csv


@dataclass(frozen=True, slots=True)
class OrbitTourAssessment:
    """Requirement outcomes for the configured orbit-assisted transfer."""

    ordered_events_pass: bool
    sphere_of_influence_pass: bool
    parking_revolutions_pass: bool
    delta_v_pass: bool
    final_mass_pass: bool
    dry_mass_pass: bool
    lambert_endpoint_pass: bool

    @property
    def all_pass(self) -> bool:
        """Return whether every declared orbit-tour requirement passes."""
        return all(
            (
                self.ordered_events_pass,
                self.sphere_of_influence_pass,
                self.parking_revolutions_pass,
                self.delta_v_pass,
                self.final_mass_pass,
                self.dry_mass_pass,
                self.lambert_endpoint_pass,
            )
        )


@dataclass(frozen=True, slots=True)
class OrbitTourSimulation:
    """Planned tour, sampled three-phase trajectory, and acceptance evidence."""

    configuration: OrbitTourConfiguration
    tour: OrbitAssistedTour
    result: SimulationResult
    assessment: OrbitTourAssessment


def _propagate_lambert_arc(
    initial_position_m: FloatArray,
    initial_velocity_mps: FloatArray,
    elapsed_time_s: FloatArray,
    primary_mu_m3_s2: float,
) -> FloatArray:
    states = np.empty((elapsed_time_s.size, 6), dtype=np.float64)
    for index, time_s in enumerate(elapsed_time_s):
        propagated = propagate_universal(
            initial_position_m,
            initial_velocity_mps,
            float(time_s),
            primary_mu_m3_s2,
        )
        states[index, :3] = propagated.position_m
        states[index, 3:] = propagated.velocity_mps
    return states


def _unit(vector: FloatArray, fallback: FloatArray | None = None) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm > 1.0e-12:
        return np.asarray(vector / norm, dtype=np.float64)
    if fallback is None:
        raise ValueError("cannot normalize the zero vector")
    return fallback.copy()


def _sample_assist_parking_orbit(
    tour: OrbitAssistedTour,
    time_s: FloatArray,
    primary_mu_m3_s2: float,
) -> FloatArray:
    incoming = _unit(tour.first_leg.arrival_excess_velocity_mps)
    outgoing = _unit(tour.second_leg.departure_excess_velocity_mps)
    normal = _unit(np.cross(incoming, outgoing), np.array([0.0, 0.0, 1.0]))
    radial = _unit(np.cross(normal, incoming), np.array([1.0, 0.0, 0.0]))
    tangent = _unit(np.cross(normal, radial), np.array([0.0, 1.0, 0.0]))
    phase = (
        2.0
        * np.pi
        * tour.dwell_revolutions
        * (time_s - tour.assist_arrival_time_s)
        / (tour.assist_departure_time_s - tour.assist_arrival_time_s)
    )
    relative_position_m = tour.assist_parking_radius_m * (
        np.cos(phase)[:, None] * radial + np.sin(phase)[:, None] * tangent
    )
    relative_velocity_mps = tour.assist_circular_speed_mps * (
        -np.sin(phase)[:, None] * radial + np.cos(phase)[:, None] * tangent
    )
    states = np.empty((time_s.size, 6), dtype=np.float64)
    for index, epoch_s in enumerate(time_s):
        body_position_m, body_velocity_mps = tour.assist_body.state_at_time(
            float(epoch_s), primary_mu_m3_s2
        )
        states[index, :3] = body_position_m + relative_position_m[index]
        states[index, 3:] = body_velocity_mps + relative_velocity_mps[index]
    return states


def _body_history(
    body: object,
    time_s: FloatArray,
    primary_mu_m3_s2: float,
) -> tuple[FloatArray, FloatArray]:
    positions = np.empty((time_s.size, 3), dtype=np.float64)
    velocities = np.empty((time_s.size, 3), dtype=np.float64)
    for index, epoch_s in enumerate(time_s):
        position, velocity = body.state_at_time(float(epoch_s), primary_mu_m3_s2)  # type: ignore[attr-defined]
        positions[index] = position
        velocities[index] = velocity
    return positions, velocities


def _event_state(time_s: FloatArray, states: FloatArray, epoch_s: float) -> FloatArray:
    return np.array(
        [np.interp(epoch_s, time_s, states[:, component]) for component in range(6)],
        dtype=np.float64,
    )


def simulate_orbit_assisted_tour(
    configuration: OrbitTourConfiguration,
) -> OrbitTourSimulation:
    """Solve the two Lambert legs and sample transfer/orbit phases deterministically."""
    catalog = configuration.catalog
    primary_mu = catalog.primary.gravitational_parameter_m3_s2
    departure = catalog.body(configuration.departure_body, role="departure")
    assist = catalog.body(configuration.assist_body, role="assist")
    destination = catalog.body(configuration.destination_body, role="destination")
    tour = plan_orbit_assisted_tour(
        departure,
        assist,
        destination,
        primary_mu,
        primary_mu / GRAVITATIONAL_CONSTANT_M3_KG_S2,
        configuration.departure_time_s,
        configuration.assist_arrival_time_s,
        configuration.destination_arrival_time_s,
        departure_parking_altitude_m=configuration.departure_parking_altitude_m,
        assist_parking_altitude_m=configuration.assist_parking_altitude_m,
        destination_parking_altitude_m=configuration.destination_parking_altitude_m,
        dwell_revolutions=configuration.assist_dwell_revolutions,
        initial_mass_kg=configuration.initial_mass_kg,
        dry_mass_kg=configuration.dry_mass_kg,
        specific_impulse_s=configuration.specific_impulse_s,
    )
    first_time = np.linspace(
        tour.departure_time_s,
        tour.assist_arrival_time_s,
        configuration.first_leg_samples,
        endpoint=False,
    )
    orbit_time = np.linspace(
        tour.assist_arrival_time_s,
        tour.assist_departure_time_s,
        configuration.parking_orbit_samples,
        endpoint=False,
    )
    second_time = np.linspace(
        tour.assist_departure_time_s,
        tour.destination_arrival_time_s,
        configuration.second_leg_samples,
    )
    first_position, _first_body_velocity = departure.state_at_time(
        tour.departure_time_s, primary_mu
    )
    first_states = _propagate_lambert_arc(
        first_position,
        tour.first_leg.lambert.departure_velocity_mps,
        first_time - tour.departure_time_s,
        primary_mu,
    )
    orbit_states = _sample_assist_parking_orbit(tour, orbit_time, primary_mu)
    second_position, _second_body_velocity = assist.state_at_time(
        tour.assist_departure_time_s, primary_mu
    )
    second_states = _propagate_lambert_arc(
        second_position,
        tour.second_leg.lambert.departure_velocity_mps,
        second_time - tour.assist_departure_time_s,
        primary_mu,
    )
    time_s = np.concatenate((first_time, orbit_time, second_time))
    states = np.vstack((first_states, orbit_states, second_states))
    phase_code = np.concatenate(
        (
            np.zeros(first_time.size),
            np.ones(orbit_time.size),
            np.full(second_time.size, 2.0),
        )
    )
    mass_kg = np.full(time_s.shape, configuration.initial_mass_kg)
    for burn in tour.burns:
        mass_kg[time_s >= burn.epoch_s] = burn.mass_after_kg

    columns: dict[str, FloatArray] = {
        "position_x_m": states[:, 0].copy(),
        "position_y_m": states[:, 1].copy(),
        "position_z_m": states[:, 2].copy(),
        "velocity_x_mps": states[:, 3].copy(),
        "velocity_y_mps": states[:, 4].copy(),
        "velocity_z_mps": states[:, 5].copy(),
        "heliocentric_distance_m": np.linalg.norm(states[:, :3], axis=1),
        "heliocentric_speed_mps": np.linalg.norm(states[:, 3:], axis=1),
        "mass_kg": mass_kg,
        "phase_code": phase_code,
    }
    body_histories: dict[str, tuple[FloatArray, FloatArray]] = {}
    for body in (departure, assist, destination):
        body_position, body_velocity = _body_history(body, time_s, primary_mu)
        body_histories[body.name] = body_position, body_velocity
        prefix = body.name.casefold()
        columns[f"distance_to_{prefix}_m"] = np.linalg.norm(states[:, :3] - body_position, axis=1)
        columns[f"relative_speed_to_{prefix}_mps"] = np.linalg.norm(
            states[:, 3:] - body_velocity, axis=1
        )
        for component, axis_name in enumerate("xyz"):
            columns[f"{prefix}_position_{axis_name}_m"] = body_position[:, component]

    event_epochs = (
        ("departure_periapsis_injection", tour.departure_time_s),
        (
            "departure_soi_exit",
            tour.departure_time_s + tour.departure_patch.time_periapsis_to_soi_s,
        ),
        (
            "assist_soi_entry",
            tour.assist_arrival_time_s - tour.assist_arrival_patch.time_periapsis_to_soi_s,
        ),
        ("assist_orbit_capture", tour.assist_arrival_time_s),
        ("assist_orbit_alignment", tour.burns[2].epoch_s),
        ("assist_periapsis_departure", tour.assist_departure_time_s),
        (
            "assist_soi_exit",
            tour.assist_departure_time_s + tour.assist_departure_patch.time_periapsis_to_soi_s,
        ),
        (
            "destination_soi_entry",
            tour.destination_arrival_time_s - tour.destination_patch.time_periapsis_to_soi_s,
        ),
        ("destination_orbit_capture", tour.destination_arrival_time_s),
    )
    events = tuple(
        EventOccurrence(name, epoch_s, _event_state(time_s, states, epoch_s))
        for name, epoch_s in event_epochs
    )
    burn_by_name = {burn.name: burn for burn in tour.burns}
    event_summary: list[dict[str, float | str]] = []
    for name, epoch_s in event_epochs:
        matching_burn = burn_by_name.get(name)
        event_summary.append(
            {
                "name": name,
                "time_days": epoch_s / 86_400.0,
                "delta_v_mps": 0.0 if matching_burn is None else matching_burn.delta_v_mps,
                "mass_after_kg": float(np.interp(epoch_s, time_s, mass_kg)),
            }
        )

    destination_endpoint, _destination_velocity = destination.state_at_time(
        tour.destination_arrival_time_s, primary_mu
    )
    endpoint_error_m = float(np.linalg.norm(states[-1, :3] - destination_endpoint))
    assessment = OrbitTourAssessment(
        ordered_events_pass=all(
            left <= right
            for left, right in zip(
                (epoch for _name, epoch in event_epochs),
                (epoch for _name, epoch in event_epochs[1:]),
                strict=False,
            )
        ),
        sphere_of_influence_pass=all(
            patch.sphere_of_influence_radius_m > patch.periapsis_radius_m
            and patch.time_periapsis_to_soi_s > 0.0
            for patch in (
                tour.departure_patch,
                tour.assist_arrival_patch,
                tour.assist_departure_patch,
                tour.destination_patch,
            )
        ),
        parking_revolutions_pass=(
            abs(
                (tour.assist_departure_time_s - tour.assist_arrival_time_s)
                / tour.assist_orbit_period_s
                - configuration.assist_dwell_revolutions
            )
            <= 1.0e-12
        ),
        delta_v_pass=tour.total_delta_v_mps <= configuration.maximum_total_delta_v_mps,
        final_mass_pass=tour.final_mass_kg >= configuration.minimum_final_mass_kg,
        dry_mass_pass=tour.final_mass_kg >= configuration.dry_mass_kg
        and float(np.min(mass_kg)) >= configuration.dry_mass_kg,
        lambert_endpoint_pass=endpoint_error_m <= 0.1,
    )
    maximum_summary: dict[str, dict[str, float | str]] = {
        "total_ideal_delta_v": {
            "value": tour.total_delta_v_mps,
            "time_days": tour.destination_arrival_time_s / 86_400.0,
            "unit": "m/s",
        },
        "final_mass": {
            "value": tour.final_mass_kg,
            "time_days": tour.destination_arrival_time_s / 86_400.0,
            "unit": "kg",
        },
        "assist_alignment_angle": {
            "value": float(np.rad2deg(tour.asymptote_alignment_angle_rad)),
            "time_days": tour.assist_arrival_time_s / 86_400.0,
            "unit": "deg",
        },
        "assist_departure_oberth_energy_gain": {
            "value": tour.departure_oberth_energy_gain_jpkg,
            "time_days": tour.assist_departure_time_s / 86_400.0,
            "unit": "J/kg",
        },
        "destination_lambert_endpoint_error": {
            "value": endpoint_error_m,
            "time_days": tour.destination_arrival_time_s / 86_400.0,
            "unit": "m",
        },
    }
    result = SimulationResult(
        scenario_name=configuration.name,
        time_s=time_s,
        columns=columns,
        events=events,
        event_summary=tuple(event_summary),
        maximum_summary=maximum_summary,
        execution_time_s=0.0,
    )
    return OrbitTourSimulation(configuration, tour, result, assessment)


def orbit_tour_payload(simulation: OrbitTourSimulation) -> dict[str, object]:
    """Build a stable, machine-readable orbit-tour verification report."""
    tour = simulation.tour
    assessment = simulation.assessment
    return {
        "scenario": simulation.configuration.name,
        "safety_scope": simulation.configuration.safety_scope,
        "model": "preliminary two-body patched conics with impulsive burns",
        "route": {
            "departure": tour.departure_body.name,
            "orbit_assist": tour.assist_body.name,
            "destination": tour.destination_body.name,
            "departure_day": tour.departure_time_s / 86_400.0,
            "assist_arrival_day": tour.assist_arrival_time_s / 86_400.0,
            "assist_departure_day": tour.assist_departure_time_s / 86_400.0,
            "destination_arrival_day": tour.destination_arrival_time_s / 86_400.0,
        },
        "assist_orbit": {
            "radius_m": tour.assist_parking_radius_m,
            "period_s": tour.assist_orbit_period_s,
            "revolutions": tour.dwell_revolutions,
            "incoming_excess_speed_mps": tour.assist_arrival_patch.excess_speed_mps,
            "outgoing_excess_speed_mps": tour.assist_departure_patch.excess_speed_mps,
            "alignment_angle_deg": float(np.rad2deg(tour.asymptote_alignment_angle_rad)),
            "departure_oberth_energy_gain_jpkg": tour.departure_oberth_energy_gain_jpkg,
        },
        "sphere_of_influence_patches": {
            name: {
                "soi_radius_m": patch.sphere_of_influence_radius_m,
                "periapsis_radius_m": patch.periapsis_radius_m,
                "eccentricity": patch.eccentricity,
                "time_periapsis_to_soi_s": patch.time_periapsis_to_soi_s,
            }
            for name, patch in (
                ("departure", tour.departure_patch),
                ("assist_arrival", tour.assist_arrival_patch),
                ("assist_departure", tour.assist_departure_patch),
                ("destination", tour.destination_patch),
            )
        },
        "burns": [
            {
                "name": burn.name,
                "epoch_day": burn.epoch_s / 86_400.0,
                "delta_v_mps": burn.delta_v_mps,
                "mass_before_kg": burn.mass_before_kg,
                "propellant_used_kg": burn.propellant_used_kg,
                "mass_after_kg": burn.mass_after_kg,
            }
            for burn in tour.burns
        ],
        "summary": {
            "total_delta_v_mps": tour.total_delta_v_mps,
            "initial_mass_kg": simulation.configuration.initial_mass_kg,
            "final_mass_kg": tour.final_mass_kg,
            "dry_mass_kg": tour.dry_mass_kg,
        },
        "requirements": {
            "ordered_events_pass": assessment.ordered_events_pass,
            "sphere_of_influence_pass": assessment.sphere_of_influence_pass,
            "parking_revolutions_pass": assessment.parking_revolutions_pass,
            "delta_v_pass": assessment.delta_v_pass,
            "final_mass_pass": assessment.final_mass_pass,
            "dry_mass_pass": assessment.dry_mass_pass,
            "lambert_endpoint_pass": assessment.lambert_endpoint_pass,
            "all_pass": assessment.all_pass,
        },
        "limitations": [
            "All bodies, vehicle values, and epochs are fictional and synthetic.",
            "Lambert arcs patch at body centres; finite SOI transit times are reported separately.",
            "Burns are ideal impulses and the parking-orbit alignment is conservative.",
            "No launch, atmosphere, low-thrust spiral, entry, landing, or "
            "operational navigation is modeled.",
        ],
    }


def write_orbit_tour_results(
    simulation: OrbitTourSimulation,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic trajectory CSV and detailed JSON report."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = write_result_csv(simulation.result, output / "orbit_assisted_tour.csv")
    report_path = output / "orbit_assisted_tour_report.json"
    report_path.write_text(
        json.dumps(orbit_tour_payload(simulation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, report_path
