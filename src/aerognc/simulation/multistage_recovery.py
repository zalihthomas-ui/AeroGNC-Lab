"""Transparent vertical multistage-ascent and recovery demonstration."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from aerognc.mathematics.integrators import EventOccurrence, EventSpec, integrate_fixed_step
from aerognc.simulation.logging import SimulationResult
from aerognc.vehicle.recovery import RecoveryDevice
from aerognc.vehicle.staging import MultistageVehicle

if TYPE_CHECKING:
    from aerognc.configuration.multistage_recovery_loader import (
        MultistageRecoveryConfiguration,
    )


def _maximum_record(time_s: np.ndarray, values: np.ndarray, unit: str) -> dict[str, float | str]:
    index = int(np.argmax(values))
    return {"value": float(values[index]), "unit": unit, "time_s": float(time_s[index])}


def simulate_multistage_recovery(
    vehicle: MultistageVehicle,
    recovery: RecoveryDevice,
    *,
    scenario_name: str = "fictional_multistage_recovery",
    initial_altitude_m: float = 0.0,
    initial_velocity_down_mps: float = -1.0,
    density_kgpm3: float = 1.225,
    gravity_mps2: float = 9.80665,
    body_drag_area_m2: float = 0.015,
    body_drag_coefficient: float = 0.45,
    step_s: float = 0.005,
    maximum_time_s: float = 180.0,
) -> SimulationResult:
    """Run a public-safe one-axis staging/recovery benchmark in SI units."""
    if not scenario_name.strip():
        raise ValueError("scenario_name cannot be empty")
    values = np.array(
        [
            initial_altitude_m,
            initial_velocity_down_mps,
            density_kgpm3,
            gravity_mps2,
            body_drag_area_m2,
            body_drag_coefficient,
            step_s,
            maximum_time_s,
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("multistage recovery inputs must be finite")
    if initial_altitude_m < 0.0 or density_kgpm3 < 0.0:
        raise ValueError("initial altitude and density must be non-negative")
    if np.any(values[3:] <= 0.0):
        raise ValueError("gravity, areas, coefficients, step, and horizon must be positive")

    def derivative(time_s: float, state: np.ndarray) -> np.ndarray:
        _altitude_m, velocity_down_mps = state
        mass_kg = vehicle.mass_kg(time_s)
        thrust_up_n = vehicle.thrust_n(time_s)
        drag_area_coefficient_m2 = (
            body_drag_coefficient * body_drag_area_m2
            + recovery.drag_coefficient * recovery.drag_area_m2(time_s)
        )
        drag_down_n = (
            -0.5
            * density_kgpm3
            * velocity_down_mps
            * abs(velocity_down_mps)
            * drag_area_coefficient_m2
        )
        acceleration_down_mps2 = gravity_mps2 + (drag_down_n - thrust_up_n) / mass_kg
        return np.array([-velocity_down_mps, acceleration_down_mps2])

    start = perf_counter()
    integration = integrate_fixed_step(
        derivative,
        [initial_altitude_m, initial_velocity_down_mps],
        (0.0, maximum_time_s),
        step_s,
        events=(
            EventSpec("apogee", lambda _time_s, state: float(state[1]), direction=1),
            EventSpec(
                "ground_contact",
                lambda _time_s, state: float(state[0]),
                direction=-1,
                terminal=True,
            ),
        ),
    )
    execution_time_s = perf_counter() - start
    if not any(event.name == "ground_contact" for event in integration.events):
        raise RuntimeError("multistage recovery did not reach ground before maximum_time_s")

    time_s = integration.time_s
    altitude_m = integration.state[:, 0]
    velocity_down_mps = integration.state[:, 1]
    mass_kg = np.array([vehicle.mass_kg(float(value)) for value in time_s])
    dry_floor_kg = np.array([vehicle.retained_dry_mass_floor_kg(float(value)) for value in time_s])
    thrust_n = np.array([vehicle.thrust_n(float(value)) for value in time_s])
    recovery_area_m2 = np.array([recovery.drag_area_m2(float(value)) for value in time_s])
    total_drag_n = (
        0.5
        * density_kgpm3
        * np.square(velocity_down_mps)
        * (body_drag_coefficient * body_drag_area_m2 + recovery.drag_coefficient * recovery_area_m2)
    )

    extra_events: list[EventOccurrence] = []
    for event in vehicle.events():
        if event.time_s <= time_s[-1]:
            event_state = np.array(
                [
                    np.interp(event.time_s, time_s, altitude_m),
                    np.interp(event.time_s, time_s, velocity_down_mps),
                ]
            )
            extra_events.append(
                EventOccurrence(f"{event.stage_name}_{event.kind}", event.time_s, event_state)
            )
    recovery_event_times = (
        ("recovery_deployment_start", recovery.deployment_time_s),
        ("recovery_reefed", recovery.reefed_time_s),
        ("recovery_full_inflation_start", recovery.full_inflation_start_time_s),
        ("recovery_fully_inflated", recovery.fully_inflated_time_s),
    )
    for name, event_time_s in recovery_event_times:
        if event_time_s <= time_s[-1]:
            extra_events.append(
                EventOccurrence(
                    name,
                    event_time_s,
                    np.array(
                        [
                            np.interp(event_time_s, time_s, altitude_m),
                            np.interp(event_time_s, time_s, velocity_down_mps),
                        ]
                    ),
                )
            )
    all_events = tuple(sorted((*integration.events, *extra_events), key=lambda event: event.time_s))
    event_summary: tuple[dict[str, float | str], ...] = tuple(
        {
            "name": event.name,
            "time_s": event.time_s,
            "altitude_m": float(event.state[0]),
            "ground_range_m": 0.0,
            "speed_mps": abs(float(event.state[1])),
        }
        for event in all_events
    )
    columns = {
        "altitude_m": altitude_m.copy(),
        "velocity_down_mps": velocity_down_mps.copy(),
        "vertical_velocity_mps": -velocity_down_mps.copy(),
        "total_velocity_mps": np.abs(velocity_down_mps),
        "mass_kg": mass_kg,
        "retained_dry_mass_floor_kg": dry_floor_kg,
        "thrust_n": thrust_n,
        "drag_n": total_drag_n,
        "recovery_drag_area_m2": recovery_area_m2,
        "opening_load_n": 0.5
        * density_kgpm3
        * np.square(velocity_down_mps)
        * recovery.drag_coefficient
        * recovery_area_m2,
    }
    maximum_summary: dict[str, dict[str, float | str]] = {
        "altitude": _maximum_record(time_s, altitude_m, "m"),
        "speed": _maximum_record(time_s, np.abs(velocity_down_mps), "m/s"),
        "opening_load": _maximum_record(time_s, columns["opening_load_n"], "N"),
        "dry_mass_margin": {
            "value": float(np.min(mass_kg - dry_floor_kg)),
            "unit": "kg",
            "time_s": float(time_s[int(np.argmin(mass_kg - dry_floor_kg))]),
        },
    }
    return SimulationResult(
        scenario_name=scenario_name,
        time_s=time_s,
        columns=columns,
        events=all_events,
        event_summary=event_summary,
        maximum_summary=maximum_summary,
        execution_time_s=execution_time_s,
    )


def simulate_configured_multistage_recovery(
    configuration: MultistageRecoveryConfiguration,
) -> SimulationResult:
    """Run a validated YAML-backed multistage/recovery scenario."""
    return simulate_multistage_recovery(
        configuration.vehicle,
        configuration.recovery,
        scenario_name=configuration.name,
        initial_altitude_m=configuration.initial_altitude_m,
        initial_velocity_down_mps=configuration.initial_velocity_down_mps,
        density_kgpm3=configuration.density_kgpm3,
        gravity_mps2=configuration.gravity_mps2,
        body_drag_area_m2=configuration.body_drag_area_m2,
        body_drag_coefficient=configuration.body_drag_coefficient,
        step_s=configuration.step_s,
        maximum_time_s=configuration.maximum_time_s,
    )
