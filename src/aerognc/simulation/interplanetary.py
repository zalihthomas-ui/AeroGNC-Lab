"""Configured restricted N-body mission propagation and encounter analysis."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.astrodynamics.dynamics import RestrictedNBodyModel
from aerognc.astrodynamics.maneuvers import (
    FiniteBurn,
    ImpulsiveManeuver,
    apply_impulsive_maneuver,
    finite_burn_derivative,
)
from aerognc.astrodynamics.perturbations import optional_perturbation_acceleration_mps2
from aerognc.configuration.interplanetary_loader import (
    SECONDS_PER_DAY,
    InterplanetaryConfiguration,
)
from aerognc.mathematics.integrators import EventOccurrence, rk4_step
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.logging import SimulationResult


@dataclass(frozen=True, slots=True)
class InterplanetaryMission:
    """Numerical mission result paired with its validated model and configuration."""

    configuration: InterplanetaryConfiguration
    model: RestrictedNBodyModel
    result: SimulationResult


def body_column_prefix(name: str) -> str:
    """Convert a configured body name into a stable lower-case log prefix."""
    prefix = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    if not prefix:
        raise ValueError("body name cannot produce an empty column prefix")
    return prefix


def _rtn_basis(position_m: FloatArray, velocity_mps: FloatArray) -> FloatArray:
    radial = position_m / np.linalg.norm(position_m)
    normal = np.cross(position_m, velocity_mps)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        raise ValueError("reference-body state cannot define an RTN frame")
    normal /= normal_norm
    transverse = np.cross(normal, radial)
    return np.column_stack((radial, transverse, normal))


def initial_spacecraft_state(configuration: InterplanetaryConfiguration) -> FloatArray:
    """Construct the primary-centred injection state from configured RTN offsets."""
    model = RestrictedNBodyModel(configuration.primary, configuration.bodies)
    body_position, body_velocity = model.body_state(configuration.spacecraft.reference_body, 0.0)
    transform = _rtn_basis(body_position, body_velocity)
    position = body_position + transform @ np.asarray(
        configuration.spacecraft.position_offset_rtn_m, dtype=np.float64
    )
    velocity = body_velocity + transform @ np.asarray(
        configuration.spacecraft.velocity_offset_rtn_mps, dtype=np.float64
    )
    return np.concatenate((position, velocity))


def _adaptive_step_s(
    model: RestrictedNBodyModel,
    time_s: float,
    state: FloatArray,
    maximum_step_s: float,
) -> float:
    step_s = maximum_step_s
    for body in model.bodies:
        relative_position, relative_velocity = model.relative_state(body.name, time_s, state)
        encounter_time_s = float(np.linalg.norm(relative_position)) / max(
            float(np.linalg.norm(relative_velocity)), 1.0
        )
        step_s = min(step_s, max(30.0, 0.025 * encounter_time_s))
    return step_s


def _propagate(
    model: RestrictedNBodyModel,
    initial_state: FloatArray,
    duration_s: float,
    maximum_step_s: float,
) -> tuple[FloatArray, FloatArray]:
    time_values = [0.0]
    state_values = [initial_state.copy()]
    time_s = 0.0
    state = initial_state.copy()
    while time_s < duration_s:
        step_s = min(_adaptive_step_s(model, time_s, state, maximum_step_s), duration_s - time_s)
        state = rk4_step(model.derivative, time_s, state, step_s)
        time_s += step_s
        if duration_s - time_s <= 1.0e-12 * duration_s:
            time_s = duration_s
        if float(np.linalg.norm(state[:3])) <= model.primary.radius_m:
            raise FloatingPointError("spacecraft collided with the primary body")
        for body in model.bodies:
            relative_position, _relative_velocity = model.relative_state(body.name, time_s, state)
            if float(np.linalg.norm(relative_position)) <= body.radius_m:
                raise FloatingPointError(f"spacecraft collided with orbiting body {body.name}")
        time_values.append(time_s)
        state_values.append(state.copy())
    return np.asarray(time_values, dtype=np.float64), np.vstack(state_values)


def _propagate_with_maneuvers(
    model: RestrictedNBodyModel,
    initial_state: FloatArray,
    configuration: InterplanetaryConfiguration,
) -> tuple[FloatArray, FloatArray, tuple[EventOccurrence, ...]]:
    """Propagate a seven-state spacecraft and apply burns at exact step boundaries."""
    impulses = sorted(
        (item for item in configuration.maneuvers if isinstance(item, ImpulsiveManeuver)),
        key=lambda item: item.epoch_s,
    )
    finite_burns = tuple(item for item in configuration.maneuvers if isinstance(item, FiniteBurn))
    boundaries = sorted(
        {
            boundary
            for maneuver in configuration.maneuvers
            for boundary in (
                (maneuver.epoch_s,)
                if isinstance(maneuver, ImpulsiveManeuver)
                else (maneuver.start_time_s, maneuver.end_time_s)
            )
            if 0.0 <= boundary <= configuration.duration_s
        }
    )
    state = np.concatenate((initial_state, [configuration.spacecraft.mass_kg]))
    occurrences: list[EventOccurrence] = []
    applied_impulses: set[str] = set()

    def apply_impulses_at(time_s: float, current_state: FloatArray) -> FloatArray:
        updated = current_state
        for impulse in impulses:
            if impulse.name in applied_impulses or not np.isclose(
                time_s, impulse.epoch_s, rtol=0.0, atol=1.0e-7
            ):
                continue
            updated = apply_impulsive_maneuver(
                updated, impulse, configuration.spacecraft.dry_mass_kg
            )
            applied_impulses.add(impulse.name)
            occurrences.append(
                EventOccurrence(
                    f"maneuver_{body_column_prefix(impulse.name)}",
                    time_s,
                    updated[:6].copy(),
                )
            )
        return updated

    def derivative(time_s: float, current_state: FloatArray) -> FloatArray:
        output = np.zeros(7, dtype=np.float64)
        output[:3] = current_state[3:6]
        output[3:6] = model.acceleration(time_s, current_state[:3])
        output[3:6] += optional_perturbation_acceleration_mps2(
            current_state[:3],
            current_state[3:6],
            float(current_state[6]),
            model.primary.gravitational_parameter_m3_s2,
            configuration.force_model,
        )
        for burn in finite_burns:
            acceleration, mass_derivative = finite_burn_derivative(
                time_s,
                current_state,
                burn,
                configuration.spacecraft.dry_mass_kg,
            )
            output[3:6] += acceleration
            output[6] += mass_derivative
        return output

    state = apply_impulses_at(0.0, state)
    time_values = [0.0]
    state_values = [state.copy()]
    time_s = 0.0
    while time_s < configuration.duration_s:
        future_boundaries = [boundary for boundary in boundaries if boundary > time_s + 1.0e-7]
        next_boundary = future_boundaries[0] if future_boundaries else configuration.duration_s
        step_s = min(
            _adaptive_step_s(model, time_s, state[:6], configuration.step_s),
            configuration.duration_s - time_s,
            next_boundary - time_s,
        )
        if step_s <= 0.0:
            raise FloatingPointError("maneuvered propagator encountered a nonpositive step")
        state = rk4_step(derivative, time_s, state, step_s)
        state[6] = max(state[6], configuration.spacecraft.dry_mass_kg)
        time_s += step_s
        if configuration.duration_s - time_s <= 1.0e-12 * configuration.duration_s:
            time_s = configuration.duration_s
        state = apply_impulses_at(time_s, state)
        for burn in finite_burns:
            if np.isclose(time_s, burn.start_time_s, rtol=0.0, atol=1.0e-7):
                occurrences.append(
                    EventOccurrence(
                        f"burn_start_{body_column_prefix(burn.name)}",
                        time_s,
                        state[:6].copy(),
                    )
                )
            if np.isclose(time_s, burn.end_time_s, rtol=0.0, atol=1.0e-7):
                occurrences.append(
                    EventOccurrence(
                        f"burn_end_{body_column_prefix(burn.name)}",
                        time_s,
                        state[:6].copy(),
                    )
                )
        if float(np.linalg.norm(state[:3])) <= model.primary.radius_m:
            raise FloatingPointError("spacecraft collided with the primary body")
        for body in model.bodies:
            relative_position, _relative_velocity = model.relative_state(
                body.name, time_s, state[:6]
            )
            if float(np.linalg.norm(relative_position)) <= body.radius_m:
                raise FloatingPointError(f"spacecraft collided with orbiting body {body.name}")
        time_values.append(time_s)
        state_values.append(state.copy())
    return (
        np.asarray(time_values, dtype=np.float64),
        np.vstack(state_values),
        tuple(occurrences),
    )


def _interpolated_occurrence(
    name: str,
    time_s: FloatArray,
    states: FloatArray,
    values: FloatArray,
    threshold: float,
    *,
    direction: int,
) -> EventOccurrence | None:
    shifted = values - threshold
    for index in range(1, time_s.size):
        before = shifted[index - 1]
        after = shifted[index]
        crossed = (direction < 0 and before > 0.0 >= after) or (
            direction > 0 and before < 0.0 <= after
        )
        if not crossed:
            continue
        denominator = abs(before) + abs(after)
        fraction = 0.5 if denominator == 0.0 else abs(before) / denominator
        event_time_s = time_s[index - 1] + fraction * (time_s[index] - time_s[index - 1])
        event_state = states[index - 1] + fraction * (states[index] - states[index - 1])
        return EventOccurrence(name, float(event_time_s), event_state)
    return None


def _closest_approach_occurrence(
    name: str, time_s: FloatArray, states: FloatArray, distance_m: FloatArray
) -> EventOccurrence:
    index = int(np.argmin(distance_m))
    if index == 0 or index == time_s.size - 1:
        return EventOccurrence(name, float(time_s[index]), states[index].copy())
    local_time = time_s[index - 1 : index + 2] - time_s[index]
    distance_squared = distance_m[index - 1 : index + 2] ** 2
    coefficients = np.polyfit(local_time, distance_squared, 2)
    offset_s = 0.0
    if coefficients[0] > 0.0:
        candidate_offset_s = float(-coefficients[1] / (2.0 * coefficients[0]))
        if local_time[0] <= candidate_offset_s <= local_time[-1]:
            offset_s = candidate_offset_s
    event_time_s = float(time_s[index] + offset_s)
    event_state = np.array(
        [np.interp(event_time_s, time_s, states[:, component]) for component in range(6)]
    )
    return EventOccurrence(name, event_time_s, event_state)


def _sample_state(time_s: FloatArray, states: FloatArray, sample_time_s: float) -> FloatArray:
    return np.array(
        [np.interp(sample_time_s, time_s, states[:, component]) for component in range(6)]
    )


def _body_arrays(
    body: CircularOrbitBody,
    time_s: FloatArray,
    states: FloatArray,
    model: RestrictedNBodyModel,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    positions = np.empty((time_s.size, 3), dtype=np.float64)
    velocities = np.empty((time_s.size, 3), dtype=np.float64)
    for index, current_time_s in enumerate(time_s):
        positions[index], velocities[index] = body.state_at_time(
            float(current_time_s), model.primary.gravitational_parameter_m3_s2
        )
    relative_position = states[:, :3] - positions
    relative_velocity = states[:, 3:6] - velocities
    return (
        positions,
        velocities,
        np.linalg.norm(relative_position, axis=1),
        np.linalg.norm(relative_velocity, axis=1),
    )


def simulate_interplanetary(
    configuration: InterplanetaryConfiguration,
) -> InterplanetaryMission:
    """Propagate one configured fictional interplanetary gravity-assist mission."""
    start = time.perf_counter()
    model = RestrictedNBodyModel(configuration.primary, configuration.bodies)
    initial_state = initial_spacecraft_state(configuration)
    force_model_enabled = (
        configuration.force_model.j2 > 0.0
        or configuration.force_model.radiation_area_m2 > 0.0
        or configuration.force_model.include_relativity
    )
    if configuration.maneuvers or force_model_enabled:
        time_s, maneuvered_states, maneuver_events = _propagate_with_maneuvers(
            model, initial_state, configuration
        )
        states = maneuvered_states[:, :6]
        mass_kg = maneuvered_states[:, 6]
    else:
        time_s, states = _propagate(
            model,
            initial_state,
            configuration.duration_s,
            configuration.step_s,
        )
        mass_kg = np.full(time_s.shape, configuration.spacecraft.mass_kg)
        maneuver_events = ()
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
        "propellant_remaining_kg": mass_kg - configuration.spacecraft.dry_mass_kg,
    }
    columns["central_specific_energy_jkg"] = (
        0.5 * columns["heliocentric_speed_mps"] ** 2
        - configuration.primary.gravitational_parameter_m3_s2 / columns["heliocentric_distance_m"]
    )
    body_data: dict[str, tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = {}
    for body in configuration.bodies:
        prefix = body_column_prefix(body.name)
        positions, velocities, distance_m, relative_speed_mps = _body_arrays(
            body, time_s, states, model
        )
        body_data[body.name] = (positions, velocities, distance_m, relative_speed_mps)
        for component, axis in enumerate("xyz"):
            columns[f"{prefix}_position_{axis}_m"] = positions[:, component]
            columns[f"{prefix}_velocity_{axis}_mps"] = velocities[:, component]
        columns[f"distance_to_{prefix}_m"] = distance_m
        columns[f"relative_speed_to_{prefix}_mps"] = relative_speed_mps

    assist = configuration.body_with_role("assist")
    destination = configuration.body_with_role("destination")
    assist_distance = body_data[assist.name][2]
    destination_distance = body_data[destination.name][2]
    assist_entry = _interpolated_occurrence(
        "assist_entry",
        time_s,
        states,
        assist_distance,
        configuration.assist_encounter_radius_m,
        direction=-1,
    )
    assist_exit = _interpolated_occurrence(
        "assist_exit",
        time_s,
        states,
        assist_distance,
        configuration.assist_encounter_radius_m,
        direction=1,
    )
    assist_closest = _closest_approach_occurrence(
        "assist_closest_approach", time_s, states, assist_distance
    )
    destination_arrival = _interpolated_occurrence(
        "destination_arrival",
        time_s,
        states,
        destination_distance,
        configuration.destination_arrival_radius_m,
        direction=-1,
    )
    events = [EventOccurrence("departure_injection", 0.0, states[0].copy())]
    events.extend(maneuver_events)
    events.extend(event for event in (assist_entry, assist_exit) if event is not None)
    if float(np.min(assist_distance)) <= configuration.assist_encounter_radius_m:
        events.append(assist_closest)
    if destination_arrival is not None:
        events.append(destination_arrival)
    events.append(EventOccurrence("mission_end", float(time_s[-1]), states[-1].copy()))
    events.sort(key=lambda event: event.time_s)

    def event_distance(event: EventOccurrence) -> tuple[str, float]:
        if event.name.startswith("assist_"):
            body = assist
        elif event.name == "destination_arrival":
            body = destination
        else:
            return configuration.primary.name, float(np.linalg.norm(event.state[:3]))
        body_position, _body_velocity = model.body_state(body.name, event.time_s)
        return body.name, float(np.linalg.norm(event.state[:3] - body_position))

    event_summary: list[dict[str, float | str]] = []
    for event in events:
        body_name, event_distance_m = event_distance(event)
        event_summary.append(
            {
                "name": event.name,
                "time_days": event.time_s / SECONDS_PER_DAY,
                "reference_body": body_name,
                "distance_m": event_distance_m,
                "heliocentric_speed_mps": float(np.linalg.norm(event.state[3:6])),
            }
        )

    speed_gain_mps = 0.0
    energy_gain_jkg = 0.0
    relative_speed_change_mps = 0.0
    if assist_entry is not None and assist_exit is not None:
        entry_state = _sample_state(time_s, states, assist_entry.time_s)
        exit_state = _sample_state(time_s, states, assist_exit.time_s)
        speed_gain_mps = float(np.linalg.norm(exit_state[3:6]) - np.linalg.norm(entry_state[3:6]))
        energy_gain_jkg = model.central_specific_energy(exit_state) - model.central_specific_energy(
            entry_state
        )
        _entry_position, entry_relative_velocity = model.relative_state(
            assist.name, assist_entry.time_s, entry_state
        )
        _exit_position, exit_relative_velocity = model.relative_state(
            assist.name, assist_exit.time_s, exit_state
        )
        relative_speed_change_mps = float(
            np.linalg.norm(exit_relative_velocity) - np.linalg.norm(entry_relative_velocity)
        )
    closest_assist_index = int(np.argmin(assist_distance))
    closest_destination_index = int(np.argmin(destination_distance))
    maximum_speed_index = int(np.argmax(columns["heliocentric_speed_mps"]))
    maximum_summary: dict[str, dict[str, float | str]] = {
        "maximum_heliocentric_speed": {
            "value": float(columns["heliocentric_speed_mps"][maximum_speed_index]),
            "time_days": float(time_s[maximum_speed_index] / SECONDS_PER_DAY),
            "unit": "m/s",
        },
        "assist_closest_approach": {
            "value": float(assist_distance[closest_assist_index]),
            "time_days": float(time_s[closest_assist_index] / SECONDS_PER_DAY),
            "unit": "m",
        },
        "assist_heliocentric_speed_gain": {
            "value": speed_gain_mps,
            "time_days": float(assist_closest.time_s / SECONDS_PER_DAY),
            "unit": "m/s",
        },
        "assist_central_energy_gain": {
            "value": energy_gain_jkg,
            "time_days": float(assist_closest.time_s / SECONDS_PER_DAY),
            "unit": "J/kg",
        },
        "assist_relative_speed_change": {
            "value": relative_speed_change_mps,
            "time_days": float(assist_closest.time_s / SECONDS_PER_DAY),
            "unit": "m/s",
        },
        "destination_closest_approach": {
            "value": float(destination_distance[closest_destination_index]),
            "time_days": float(time_s[closest_destination_index] / SECONDS_PER_DAY),
            "unit": "m",
        },
        "destination_arrival": {
            "value": 1.0 if destination_arrival is not None else 0.0,
            "time_days": (
                float(destination_arrival.time_s / SECONDS_PER_DAY)
                if destination_arrival is not None
                else float(time_s[-1] / SECONDS_PER_DAY)
            ),
            "unit": "boolean",
        },
        "propellant_used": {
            "value": float(configuration.spacecraft.mass_kg - mass_kg[-1]),
            "time_days": float(time_s[-1] / SECONDS_PER_DAY),
            "unit": "kg",
        },
        "commanded_impulsive_delta_v": {
            "value": float(
                sum(
                    maneuver.magnitude_mps
                    for maneuver in configuration.maneuvers
                    if isinstance(maneuver, ImpulsiveManeuver)
                )
            ),
            "time_days": float(time_s[-1] / SECONDS_PER_DAY),
            "unit": "m/s",
        },
    }
    result = SimulationResult(
        scenario_name=configuration.name,
        time_s=time_s,
        columns=columns,
        events=tuple(events),
        event_summary=tuple(event_summary),
        maximum_summary=maximum_summary,
        execution_time_s=time.perf_counter() - start,
    )
    return InterplanetaryMission(configuration, model, result)
