"""Configured point-mass simulation orchestration."""

from time import perf_counter

import numpy as np

from aerognc.configuration.models import ThreeDofConfiguration
from aerognc.dynamics.three_dof import PointMassModel
from aerognc.mathematics.integrators import EventOccurrence, integrate_fixed_step
from aerognc.simulation.events import flight_event_specs
from aerognc.simulation.logging import SimulationResult


def _maximum_record(time_s: np.ndarray, values: np.ndarray, unit: str) -> dict[str, float | str]:
    index = int(np.argmax(values))
    return {"value": float(values[index]), "unit": unit, "time_s": float(time_s[index])}


def simulate_three_dof(
    configuration: ThreeDofConfiguration,
    *,
    thrust_misalignment_pitch_rad: float = 0.0,
    thrust_misalignment_yaw_rad: float = 0.0,
) -> SimulationResult:
    """Run one deterministic, event-driven fictional ascent simulation."""
    model = PointMassModel(
        configuration.vehicle,
        configuration.environment,
        configuration.launch,
        thrust_misalignment_pitch_rad=thrust_misalignment_pitch_rad,
        thrust_misalignment_yaw_rad=thrust_misalignment_yaw_rad,
    )
    start = perf_counter()
    integration = integrate_fixed_step(
        model.derivative,
        model.initial_state(),
        (0.0, configuration.simulation.maximum_time_s),
        configuration.simulation.step_s,
        events=flight_event_specs(configuration.vehicle.propulsion.burnout_time_s),
    )
    execution_time_s = perf_counter() - start

    count = integration.time_s.size
    position = integration.state[:, :3]
    velocity = integration.state[:, 3:6]
    diagnostic_names = (
        "altitude_m",
        "ground_range_m",
        "vertical_velocity_up_mps",
        "total_velocity_mps",
        "airspeed_mps",
        "acceleration_north_mps2",
        "acceleration_east_mps2",
        "acceleration_down_mps2",
        "acceleration_magnitude_mps2",
        "mach",
        "dynamic_pressure_pa",
        "mass_kg",
        "thrust_n",
        "drag_n",
        "flight_path_angle_deg",
        "wind_north_mps",
        "wind_east_mps",
        "wind_down_mps",
    )
    diagnostic_data = {name: np.empty(count, dtype=np.float64) for name in diagnostic_names}
    for index, (time_s, state) in enumerate(
        zip(integration.time_s, integration.state, strict=True)
    ):
        diagnostic = model.diagnostics(float(time_s), state)
        diagnostic_data["altitude_m"][index] = diagnostic.altitude_m
        diagnostic_data["ground_range_m"][index] = diagnostic.ground_range_m
        diagnostic_data["vertical_velocity_up_mps"][index] = diagnostic.vertical_velocity_up_mps
        diagnostic_data["total_velocity_mps"][index] = diagnostic.total_velocity_mps
        diagnostic_data["airspeed_mps"][index] = diagnostic.airspeed_mps
        diagnostic_data["acceleration_north_mps2"][index] = diagnostic.acceleration_ned_mps2[0]
        diagnostic_data["acceleration_east_mps2"][index] = diagnostic.acceleration_ned_mps2[1]
        diagnostic_data["acceleration_down_mps2"][index] = diagnostic.acceleration_ned_mps2[2]
        diagnostic_data["acceleration_magnitude_mps2"][index] = (
            diagnostic.acceleration_magnitude_mps2
        )
        diagnostic_data["mach"][index] = diagnostic.mach
        diagnostic_data["dynamic_pressure_pa"][index] = diagnostic.dynamic_pressure_pa
        diagnostic_data["mass_kg"][index] = diagnostic.mass_kg
        diagnostic_data["thrust_n"][index] = diagnostic.thrust_n
        diagnostic_data["drag_n"][index] = diagnostic.drag_n
        diagnostic_data["flight_path_angle_deg"][index] = np.rad2deg(
            diagnostic.flight_path_angle_rad
        )
        diagnostic_data["wind_north_mps"][index] = diagnostic.wind_velocity_ned_mps[0]
        diagnostic_data["wind_east_mps"][index] = diagnostic.wind_velocity_ned_mps[1]
        diagnostic_data["wind_down_mps"][index] = diagnostic.wind_velocity_ned_mps[2]

    columns = {
        "north_m": position[:, 0].copy(),
        "east_m": position[:, 1].copy(),
        "down_m": position[:, 2].copy(),
        "velocity_north_mps": velocity[:, 0].copy(),
        "velocity_east_mps": velocity[:, 1].copy(),
        "velocity_down_mps": velocity[:, 2].copy(),
        **diagnostic_data,
    }
    event_summary = _event_summary(model, integration.events)
    maximum_summary = {
        "altitude": _maximum_record(integration.time_s, columns["altitude_m"], "m"),
        "ground_range": _maximum_record(integration.time_s, columns["ground_range_m"], "m"),
        "speed": _maximum_record(integration.time_s, columns["total_velocity_mps"], "m/s"),
        "mach": _maximum_record(integration.time_s, columns["mach"], "1"),
        "dynamic_pressure": _maximum_record(
            integration.time_s, columns["dynamic_pressure_pa"], "Pa"
        ),
        "acceleration": _maximum_record(
            integration.time_s, columns["acceleration_magnitude_mps2"], "m/s^2"
        ),
    }
    return SimulationResult(
        scenario_name=configuration.simulation.name,
        time_s=integration.time_s,
        columns=columns,
        events=integration.events,
        event_summary=event_summary,
        maximum_summary=maximum_summary,
        execution_time_s=execution_time_s,
    )


def _event_summary(
    model: PointMassModel, events: tuple[EventOccurrence, ...]
) -> tuple[dict[str, float | str], ...]:
    records: list[dict[str, float | str]] = []
    for event in events:
        diagnostic = model.diagnostics(event.time_s, event.state)
        records.append(
            {
                "name": event.name,
                "time_s": event.time_s,
                "altitude_m": diagnostic.altitude_m,
                "ground_range_m": diagnostic.ground_range_m,
                "speed_mps": diagnostic.total_velocity_mps,
            }
        )
    return tuple(records)
