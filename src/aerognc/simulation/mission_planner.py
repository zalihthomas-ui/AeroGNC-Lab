"""High-level patched-conic mission planning for the desktop Mission Designer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.astrodynamics.dynamics import RestrictedNBodyModel
from aerognc.astrodynamics.kepler import propagate_universal
from aerognc.astrodynamics.maneuvers import ImpulsiveManeuver, apply_impulsive_maneuver
from aerognc.astrodynamics.mission_design import (
    GravityAssistDesign,
    TransferOpportunity,
    design_gravity_assist,
    evaluate_lambert_transfer,
    injection_delta_v_mps,
)
from aerognc.configuration.interplanetary_loader import (
    InterplanetaryConfiguration,
    SpacecraftInjection,
)
from aerognc.configuration.planetary_catalog import PlanetaryCatalog
from aerognc.mathematics.integrators import EventOccurrence
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.interplanetary import (
    InterplanetaryMission,
    body_column_prefix,
)
from aerognc.simulation.logging import SimulationResult

MissionMethod = Literal["direct", "gravity_assist"]


@dataclass(frozen=True, slots=True)
class MissionPlanRequest:
    """Understandable mission-design inputs; all epochs are relative catalog days."""

    method: MissionMethod
    departure_body: str
    destination_body: str
    departure_day: float
    arrival_day: float
    assist_body: str | None = None
    assist_day: float | None = None
    parking_altitude_m: float = 200_000.0
    destination_parking_altitude_m: float = 200_000.0
    minimum_flyby_altitude_m: float = 200_000.0
    initial_mass_kg: float = 1_450.0
    dry_mass_kg: float = 200.0
    specific_impulse_s: float = 450.0
    sample_count: int = 900
    maneuvers: tuple[ImpulsiveManeuver, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in {"direct", "gravity_assist"}:
            raise ValueError("mission method must be direct or gravity_assist")
        if not self.departure_body.strip() or not self.destination_body.strip():
            raise ValueError("departure and destination must be selected")
        if self.departure_body == self.destination_body:
            raise ValueError("departure and destination must differ")
        scalar_values = np.array(
            [
                self.departure_day,
                self.arrival_day,
                self.parking_altitude_m,
                self.destination_parking_altitude_m,
                self.minimum_flyby_altitude_m,
                self.initial_mass_kg,
                self.dry_mass_kg,
                self.specific_impulse_s,
            ]
        )
        if not np.all(np.isfinite(scalar_values)):
            raise ValueError("mission inputs must be finite")
        if self.departure_day < 0.0 or self.arrival_day <= self.departure_day:
            raise ValueError("arrival day must be later than a nonnegative departure day")
        if np.any(scalar_values[2:] <= 0.0) or self.dry_mass_kg > self.initial_mass_kg:
            raise ValueError(
                "altitudes/masses/Isp must be positive and dry mass cannot exceed mass"
            )
        if not 100 <= self.sample_count <= 5_000:
            raise ValueError("trajectory sample count must be between 100 and 5000")
        if self.method == "gravity_assist":
            if self.assist_body is None or self.assist_day is None:
                raise ValueError("gravity-assist method requires an assist body and encounter day")
            if self.assist_body in {self.departure_body, self.destination_body}:
                raise ValueError("assist body must differ from departure and destination")
            if not self.departure_day < self.assist_day < self.arrival_day:
                raise ValueError("assist day must lie strictly between departure and arrival")
            if self.maneuvers:
                raise ValueError(
                    "manual midcourse maneuvers are supported for direct plans; "
                    "the gravity-assist planner determines its own leg transition"
                )
        mission_duration_s = (self.arrival_day - self.departure_day) * 86_400.0
        if any(maneuver.epoch_s > mission_duration_s for maneuver in self.maneuvers):
            raise ValueError("every manual maneuver must occur before mission arrival")


@dataclass(frozen=True, slots=True)
class MissionDesignMetrics:
    """Primary launch, arrival, flyby, and propellant-planning quantities."""

    departure_c3_m2_s2: float
    departure_excess_speed_mps: float
    injection_delta_v_mps: float
    arrival_excess_speed_mps: float
    ideal_capture_delta_v_mps: float
    midcourse_delta_v_mps: float
    ideal_total_delta_v_mps: float
    ideal_propellant_required_kg: float
    flyby_altitude_m: float | None
    powered_flyby_delta_v_mps: float | None
    destination_miss_distance_m: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class PlannedMission:
    """Mission-design solution and an immediately playable patched-conic trajectory."""

    request: MissionPlanRequest
    metrics: MissionDesignMetrics
    mission: InterplanetaryMission
    direct_transfer: TransferOpportunity | None = None
    gravity_assist: GravityAssistDesign | None = None


def _shift_body_epoch(
    body: CircularOrbitBody, departure_epoch_s: float, primary_mu_m3_s2: float
) -> CircularOrbitBody:
    shifted_mean_anomaly = (
        body.phase_at_epoch_rad + body.mean_motion_rad_s(primary_mu_m3_s2) * departure_epoch_s
    )
    return replace(body, phase_at_epoch_rad=float(np.mod(shifted_mean_anomaly, 2.0 * np.pi)))


def _propagate_arc(
    position_m: FloatArray,
    velocity_mps: FloatArray,
    elapsed_times_s: FloatArray,
    primary_mu_m3_s2: float,
) -> FloatArray:
    states = np.empty((elapsed_times_s.size, 6), dtype=np.float64)
    for index, elapsed_time_s in enumerate(elapsed_times_s):
        propagated = propagate_universal(
            position_m,
            velocity_mps,
            float(elapsed_time_s),
            primary_mu_m3_s2,
        )
        states[index, :3] = propagated.position_m
        states[index, 3:6] = propagated.velocity_mps
    return states


def _propagate_arc_with_impulses(
    position_m: FloatArray,
    velocity_mps: FloatArray,
    elapsed_times_s: FloatArray,
    primary_mu_m3_s2: float,
    initial_mass_kg: float,
    dry_mass_kg: float,
    maneuvers: tuple[ImpulsiveManeuver, ...],
) -> tuple[FloatArray, FloatArray, tuple[EventOccurrence, ...]]:
    """Propagate a direct conic in exact segments around user-entered impulses."""
    ordered = sorted(maneuvers, key=lambda maneuver: maneuver.epoch_s)
    segment_epoch_s = 0.0
    segment_state = np.concatenate((position_m, velocity_mps, [initial_mass_kg]))
    states = np.empty((elapsed_times_s.size, 6), dtype=np.float64)
    masses = np.empty(elapsed_times_s.size, dtype=np.float64)
    occurrences: list[EventOccurrence] = []
    maneuver_index = 0
    for sample_index, sample_time_s in enumerate(elapsed_times_s):
        while (
            maneuver_index < len(ordered)
            and ordered[maneuver_index].epoch_s <= sample_time_s + 1.0e-9
        ):
            maneuver = ordered[maneuver_index]
            propagated = propagate_universal(
                segment_state[:3],
                segment_state[3:6],
                maneuver.epoch_s - segment_epoch_s,
                primary_mu_m3_s2,
            )
            segment_state[:3] = propagated.position_m
            segment_state[3:6] = propagated.velocity_mps
            segment_state = apply_impulsive_maneuver(segment_state, maneuver, dry_mass_kg)
            segment_epoch_s = maneuver.epoch_s
            occurrences.append(
                EventOccurrence(
                    f"maneuver_{body_column_prefix(maneuver.name)}",
                    maneuver.epoch_s,
                    segment_state[:6].copy(),
                )
            )
            maneuver_index += 1
        propagated = propagate_universal(
            segment_state[:3],
            segment_state[3:6],
            float(sample_time_s - segment_epoch_s),
            primary_mu_m3_s2,
        )
        states[sample_index, :3] = propagated.position_m
        states[sample_index, 3:6] = propagated.velocity_mps
        masses[sample_index] = segment_state[6]
    return states, masses, tuple(occurrences)


def _event_summary(
    events: tuple[EventOccurrence, ...],
    configuration: InterplanetaryConfiguration,
    model: RestrictedNBodyModel,
) -> tuple[dict[str, float | str], ...]:
    assist = configuration.body_with_role("assist")
    destination = configuration.body_with_role("destination")
    rows: list[dict[str, float | str]] = []
    for event in events:
        if event.name.startswith("assist_"):
            reference_body_name = assist.name
            position, _velocity = model.body_state(reference_body_name, event.time_s)
            distance = float(np.linalg.norm(event.state[:3] - position))
        elif event.name == "destination_arrival":
            reference_body_name = destination.name
            position, _velocity = model.body_state(reference_body_name, event.time_s)
            distance = float(np.linalg.norm(event.state[:3] - position))
        else:
            reference_body_name = configuration.primary.name
            distance = float(np.linalg.norm(event.state[:3]))
        rows.append(
            {
                "name": event.name,
                "time_days": event.time_s / 86_400.0,
                "reference_body": reference_body_name,
                "distance_m": distance,
                "heliocentric_speed_mps": float(np.linalg.norm(event.state[3:6])),
            }
        )
    return tuple(rows)


def _build_playable_mission(
    catalog: PlanetaryCatalog,
    request: MissionPlanRequest,
    time_s: FloatArray,
    states: FloatArray,
    mass_kg: FloatArray,
    assist_local_time_s: float | None,
    maneuver_events: tuple[EventOccurrence, ...],
) -> InterplanetaryMission:
    primary_mu = catalog.primary.gravitational_parameter_m3_s2
    departure_epoch_s = request.departure_day * 86_400.0
    departure = _shift_body_epoch(
        catalog.body(request.departure_body, role="departure"), departure_epoch_s, primary_mu
    )
    destination = _shift_body_epoch(
        catalog.body(request.destination_body, role="destination"), departure_epoch_s, primary_mu
    )
    if request.method == "gravity_assist":
        if request.assist_body is None:
            raise RuntimeError("gravity-assist plan lost its assist body")
        assist_name = request.assist_body
    else:
        assist_name = next(
            body.name
            for body in catalog.bodies
            if body.name not in {request.departure_body, request.destination_body}
        )
    assist = _shift_body_epoch(
        catalog.body(assist_name, role="assist"), departure_epoch_s, primary_mu
    )
    backgrounds = tuple(
        _shift_body_epoch(body, departure_epoch_s, primary_mu)
        for body in catalog.bodies
        if body.name not in {departure.name, destination.name, assist.name}
    )
    bodies = (departure, assist, destination, *backgrounds)
    departure_body_position, departure_body_velocity = departure.state_at_time(0.0, primary_mu)
    departure_excess = states[0, 3:6] - departure_body_velocity
    radial = departure_body_position / np.linalg.norm(departure_body_position)
    position_offset_array = radial * 2.0 * departure.radius_m
    position_offset = (
        float(position_offset_array[0]),
        float(position_offset_array[1]),
        float(position_offset_array[2]),
    )
    velocity_offset = (
        float(departure_excess[0]),
        float(departure_excess[1]),
        float(departure_excess[2]),
    )
    configuration = InterplanetaryConfiguration(
        source_path=Path("mission-designer-generated"),
        name=f"{request.departure_body}_to_{request.destination_body}_{request.method}",
        description="Mission Designer patched-conic preliminary trajectory",
        safety_scope=(
            "Fictional civilian research spacecraft with synthetic parameters for education."
        ),
        primary=catalog.primary,
        bodies=bodies,
        spacecraft=SpacecraftInjection(
            name="Selene Design Vehicle",
            mass_kg=request.initial_mass_kg,
            reference_body=departure.name,
            position_offset_rtn_m=position_offset,
            velocity_offset_rtn_mps=velocity_offset,
            dry_mass_kg=request.dry_mass_kg,
        ),
        duration_s=float(time_s[-1]),
        step_s=float(np.median(np.diff(time_s))),
        assist_encounter_radius_m=50.0 * assist.radius_m,
        destination_arrival_radius_m=50.0 * destination.radius_m,
        output_directory=Path("results/mission_designer"),
        snapshot_time_s=0.5 * float(time_s[-1]),
    )
    model = RestrictedNBodyModel(configuration.primary, configuration.bodies)
    columns: dict[str, FloatArray] = {
        "position_x_m": states[:, 0].copy(),
        "position_y_m": states[:, 1].copy(),
        "position_z_m": states[:, 2].copy(),
        "velocity_x_mps": states[:, 3].copy(),
        "velocity_y_mps": states[:, 4].copy(),
        "velocity_z_mps": states[:, 5].copy(),
        "heliocentric_distance_m": np.linalg.norm(states[:, :3], axis=1),
        "heliocentric_speed_mps": np.linalg.norm(states[:, 3:6], axis=1),
        "mass_kg": mass_kg.copy(),
        "propellant_remaining_kg": mass_kg - request.dry_mass_kg,
    }
    columns["central_specific_energy_jkg"] = (
        0.5 * columns["heliocentric_speed_mps"] ** 2
        - primary_mu / columns["heliocentric_distance_m"]
    )
    distances: dict[str, FloatArray] = {}
    relative_speeds: dict[str, FloatArray] = {}
    for body in configuration.bodies:
        prefix = body_column_prefix(body.name)
        positions = np.empty((time_s.size, 3))
        velocities = np.empty((time_s.size, 3))
        for index, current_time_s in enumerate(time_s):
            positions[index], velocities[index] = body.state_at_time(
                float(current_time_s), primary_mu
            )
        distances[body.name] = np.linalg.norm(states[:, :3] - positions, axis=1)
        relative_speeds[body.name] = np.linalg.norm(states[:, 3:6] - velocities, axis=1)
        for component, axis_name in enumerate("xyz"):
            columns[f"{prefix}_position_{axis_name}_m"] = positions[:, component]
            columns[f"{prefix}_velocity_{axis_name}_mps"] = velocities[:, component]
        columns[f"distance_to_{prefix}_m"] = distances[body.name]
        columns[f"relative_speed_to_{prefix}_mps"] = relative_speeds[body.name]
    events: list[EventOccurrence] = [EventOccurrence("departure_injection", 0.0, states[0].copy())]
    events.extend(maneuver_events)
    if assist_local_time_s is not None:
        assist_index = int(np.argmin(np.abs(time_s - assist_local_time_s)))
        for name, offset_s in (
            ("assist_entry", -86_400.0),
            ("assist_closest_approach", 0.0),
            ("assist_exit", 86_400.0),
        ):
            event_time = float(np.clip(assist_local_time_s + offset_s, 0.0, time_s[-1]))
            event_state = np.array(
                [np.interp(event_time, time_s, states[:, component]) for component in range(6)]
            )
            if name == "assist_closest_approach":
                event_state = states[assist_index].copy()
            events.append(EventOccurrence(name, event_time, event_state))
    events.extend(
        [
            EventOccurrence("destination_arrival", float(time_s[-1]), states[-1].copy()),
            EventOccurrence("mission_end", float(time_s[-1]), states[-1].copy()),
        ]
    )
    event_tuple = tuple(sorted(events, key=lambda event: event.time_s))
    assist_distance = distances[assist.name]
    destination_distance = distances[destination.name]
    speed = columns["heliocentric_speed_mps"]
    maximum_summary: dict[str, dict[str, float | str]] = {
        "maximum_heliocentric_speed": {
            "value": float(np.max(speed)),
            "time_days": float(time_s[int(np.argmax(speed))] / 86_400.0),
            "unit": "m/s",
        },
        "assist_closest_approach": {
            "value": float(np.min(assist_distance)),
            "time_days": float(time_s[int(np.argmin(assist_distance))] / 86_400.0),
            "unit": "m",
        },
        "assist_heliocentric_speed_gain": {"value": 0.0, "time_days": 0.0, "unit": "m/s"},
        "assist_central_energy_gain": {"value": 0.0, "time_days": 0.0, "unit": "J/kg"},
        "assist_relative_speed_change": {"value": 0.0, "time_days": 0.0, "unit": "m/s"},
        "destination_closest_approach": {
            "value": float(np.min(destination_distance)),
            "time_days": float(time_s[int(np.argmin(destination_distance))] / 86_400.0),
            "unit": "m",
        },
        "destination_arrival": {
            "value": 1.0,
            "time_days": float(time_s[-1] / 86_400.0),
            "unit": "boolean",
        },
        "propellant_used": {
            "value": float(request.initial_mass_kg - mass_kg[-1]),
            "time_days": float(time_s[-1] / 86_400.0),
            "unit": "kg",
        },
        "commanded_impulsive_delta_v": {
            "value": float(sum(maneuver.magnitude_mps for maneuver in request.maneuvers)),
            "time_days": 0.0,
            "unit": "m/s",
        },
    }
    result = SimulationResult(
        scenario_name=configuration.name,
        time_s=time_s,
        columns=columns,
        events=event_tuple,
        event_summary=_event_summary(event_tuple, configuration, model),
        maximum_summary=maximum_summary,
        execution_time_s=0.0,
    )
    return InterplanetaryMission(configuration, model, result)


def plan_mission(catalog: PlanetaryCatalog, request: MissionPlanRequest) -> PlannedMission:
    """Solve a direct or single-assist preliminary mission and create 3D playback data."""
    primary_mu = catalog.primary.gravitational_parameter_m3_s2
    departure = catalog.body(request.departure_body, role="departure")
    destination = catalog.body(request.destination_body, role="destination")
    departure_time_s = request.departure_day * 86_400.0
    arrival_time_s = request.arrival_day * 86_400.0
    direct_transfer: TransferOpportunity | None = None
    gravity_assist: GravityAssistDesign | None = None
    assist_local_time_s: float | None = None
    if request.method == "direct":
        direct_transfer = evaluate_lambert_transfer(
            departure, destination, primary_mu, departure_time_s, arrival_time_s
        )
        local_time_s = np.linspace(0.0, arrival_time_s - departure_time_s, request.sample_count)
        departure_position, _departure_velocity = departure.state_at_time(
            departure_time_s, primary_mu
        )
        states, mass_kg, maneuver_events = _propagate_arc_with_impulses(
            departure_position,
            direct_transfer.lambert.departure_velocity_mps,
            local_time_s,
            primary_mu,
            request.initial_mass_kg,
            request.dry_mass_kg,
            request.maneuvers,
        )
        departure_excess_speed = float(np.sqrt(direct_transfer.departure_c3_m2_s2))
        arrival_excess_speed = direct_transfer.arrival_excess_speed_mps
        feasible = True
        flyby_altitude = None
        flyby_delta_v = None
    else:
        if request.assist_body is None or request.assist_day is None:
            raise RuntimeError("gravity-assist request lost required fields")
        assist = catalog.body(request.assist_body, role="assist")
        assist_time_s = request.assist_day * 86_400.0
        gravity_assist = design_gravity_assist(
            departure,
            assist,
            destination,
            primary_mu,
            departure_time_s,
            assist_time_s,
            arrival_time_s,
            minimum_flyby_altitude_m=request.minimum_flyby_altitude_m,
            excess_speed_tolerance_mps=250.0,
        )
        first_count = max(
            50,
            int(
                request.sample_count
                * (assist_time_s - departure_time_s)
                / (arrival_time_s - departure_time_s)
            ),
        )
        second_count = request.sample_count - first_count + 1
        first_elapsed = np.linspace(0.0, assist_time_s - departure_time_s, first_count)
        second_elapsed = np.linspace(0.0, arrival_time_s - assist_time_s, second_count)
        departure_position, _departure_velocity = departure.state_at_time(
            departure_time_s, primary_mu
        )
        assist_position, _assist_velocity = assist.state_at_time(assist_time_s, primary_mu)
        first_states = _propagate_arc(
            departure_position,
            gravity_assist.first_leg.lambert.departure_velocity_mps,
            first_elapsed,
            primary_mu,
        )
        second_states = _propagate_arc(
            assist_position,
            gravity_assist.second_leg.lambert.departure_velocity_mps,
            second_elapsed,
            primary_mu,
        )
        local_time_s = np.concatenate((first_elapsed, second_elapsed[1:] + first_elapsed[-1]))
        states = np.vstack((first_states, second_states[1:]))
        mass_kg = np.full(local_time_s.shape, request.initial_mass_kg)
        maneuver_events = ()
        assist_local_time_s = assist_time_s - departure_time_s
        departure_excess_speed = float(np.sqrt(gravity_assist.first_leg.departure_c3_m2_s2))
        arrival_excess_speed = gravity_assist.second_leg.arrival_excess_speed_mps
        feasible = gravity_assist.feasible
        flyby_altitude = gravity_assist.flyby.periapsis_altitude_m
        flyby_delta_v = gravity_assist.flyby.powered_flyby_delta_v_mps
    injection_delta_v = injection_delta_v_mps(
        departure_excess_speed,
        departure.gravitational_parameter_m3_s2,
        departure.radius_m + request.parking_altitude_m,
    )
    destination_position, destination_velocity = destination.state_at_time(
        arrival_time_s, primary_mu
    )
    destination_miss_distance = float(np.linalg.norm(states[-1, :3] - destination_position))
    arrival_excess_speed = float(np.linalg.norm(states[-1, 3:6] - destination_velocity))
    capture_delta_v = injection_delta_v_mps(
        arrival_excess_speed,
        destination.gravitational_parameter_m3_s2,
        destination.radius_m + request.destination_parking_altitude_m,
    )
    midcourse_delta_v = float(sum(maneuver.magnitude_mps for maneuver in request.maneuvers))
    total_delta_v = injection_delta_v + capture_delta_v + midcourse_delta_v + (flyby_delta_v or 0.0)
    final_mass = request.initial_mass_kg * np.exp(
        -total_delta_v / (request.specific_impulse_s * 9.80665)
    )
    propellant_required = request.initial_mass_kg - final_mass
    propellant_feasible = bool(final_mass >= request.dry_mass_kg)
    metrics = MissionDesignMetrics(
        departure_c3_m2_s2=departure_excess_speed**2,
        departure_excess_speed_mps=departure_excess_speed,
        injection_delta_v_mps=injection_delta_v,
        arrival_excess_speed_mps=arrival_excess_speed,
        ideal_capture_delta_v_mps=capture_delta_v,
        midcourse_delta_v_mps=midcourse_delta_v,
        ideal_total_delta_v_mps=total_delta_v,
        ideal_propellant_required_kg=float(propellant_required),
        flyby_altitude_m=flyby_altitude,
        powered_flyby_delta_v_mps=flyby_delta_v,
        destination_miss_distance_m=destination_miss_distance,
        feasible=(
            feasible
            and propellant_feasible
            and destination_miss_distance <= 50.0 * destination.radius_m
        ),
    )
    mission = _build_playable_mission(
        catalog,
        request,
        local_time_s,
        states,
        mass_kg,
        assist_local_time_s,
        maneuver_events,
    )
    return PlannedMission(request, metrics, mission, direct_transfer, gravity_assist)
