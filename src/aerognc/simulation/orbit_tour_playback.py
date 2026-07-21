"""Adapt a patched-conic orbit tour to the shared interactive mission player."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aerognc.astrodynamics.dynamics import RestrictedNBodyModel
from aerognc.configuration.interplanetary_loader import (
    InterplanetaryConfiguration,
    SpacecraftInjection,
)
from aerognc.mathematics.integrators import EventOccurrence
from aerognc.simulation.interplanetary import InterplanetaryMission
from aerognc.simulation.orbit_assisted_tour import OrbitTourSimulation


def orbit_tour_playback_mission(simulation: OrbitTourSimulation) -> InterplanetaryMission:
    """Return an immutable playback-compatible view without changing tour evidence."""
    tour_configuration = simulation.configuration
    departure = tour_configuration.catalog.body(
        tour_configuration.departure_body,
        role="departure",
    )
    assist = tour_configuration.catalog.body(tour_configuration.assist_body, role="assist")
    destination = tour_configuration.catalog.body(
        tour_configuration.destination_body,
        role="destination",
    )
    bodies = (departure, assist, destination)
    primary = tour_configuration.catalog.primary
    initial_epoch_s = float(simulation.result.time_s[0])
    elapsed_time_s = simulation.result.time_s - initial_epoch_s
    time_steps = np.diff(elapsed_time_s)
    representative_step_s = float(np.median(time_steps[time_steps > 0.0]))
    configuration = InterplanetaryConfiguration(
        source_path=tour_configuration.source_path,
        name=tour_configuration.name,
        description=(
            "Fictional capture, parking-orbit dwell, powered departure, and "
            "destination-capture playback"
        ),
        safety_scope=tour_configuration.safety_scope,
        primary=primary,
        bodies=bodies,
        spacecraft=SpacecraftInjection(
            name=tour_configuration.spacecraft_name,
            mass_kg=tour_configuration.initial_mass_kg,
            reference_body=departure.name,
            position_offset_rtn_m=(0.0, 0.0, 0.0),
            velocity_offset_rtn_mps=(0.0, 0.0, 0.0),
            dry_mass_kg=tour_configuration.dry_mass_kg,
        ),
        duration_s=float(elapsed_time_s[-1]),
        step_s=representative_step_s,
        assist_encounter_radius_m=5.0 * simulation.tour.assist_parking_radius_m,
        destination_arrival_radius_m=(
            5.0 * (destination.radius_m + tour_configuration.destination_parking_altitude_m)
        ),
        output_directory=tour_configuration.output_directory,
        snapshot_time_s=simulation.tour.assist_arrival_time_s - initial_epoch_s,
    )
    columns = {name: values.copy() for name, values in simulation.result.columns.items()}
    radius_m = np.maximum(columns["heliocentric_distance_m"], 1.0)
    columns["central_specific_energy_jkg"] = (
        0.5 * columns["heliocentric_speed_mps"] ** 2
        - primary.gravitational_parameter_m3_s2 / radius_m
    )
    shifted_events = tuple(
        EventOccurrence(event.name, event.time_s - initial_epoch_s, event.state.copy())
        for event in simulation.result.events
    )
    shifted_summary: list[dict[str, float | str]] = []
    for record in simulation.result.event_summary:
        shifted = dict(record)
        if "time_days" in shifted:
            shifted["time_days"] = float(shifted["time_days"]) - initial_epoch_s / 86_400.0
        shifted_summary.append(shifted)
    result = replace(
        simulation.result,
        time_s=elapsed_time_s.copy(),
        columns=columns,
        events=shifted_events,
        event_summary=tuple(shifted_summary),
    )
    model = RestrictedNBodyModel(primary, bodies)
    return InterplanetaryMission(configuration, model, result)
