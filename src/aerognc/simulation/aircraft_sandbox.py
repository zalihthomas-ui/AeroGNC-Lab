"""Batch orchestration for the coefficient-driven fictional aircraft sandbox."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.mathematics.integrators import EventOccurrence, rk4_step
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.aircraft_telemetry import aircraft_telemetry
from aerognc.simulation.logging import SimulationResult, write_result_csv, write_summary_json
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    AircraftState,
    FixedWingFlightModel,
    aircraft_initial_state,
    longitudinal_trim_command,
    project_aircraft_state,
)

KARMAN_LINE_ALTITUDE_M = 100_000.0
AircraftCommandFunction = Callable[[float, AircraftState], AircraftControlCommand]


@dataclass(frozen=True, slots=True)
class AircraftSandboxSimulation:
    """Trajectory plus explicit interpretation of ground, stall, and space events."""

    configuration: AircraftSandboxConfiguration
    result: SimulationResult
    final_state: FloatArray
    reached_space: bool
    impacted_ground: bool
    stalled_duration_s: float
    interpretation: str


def _constant_trim_provider(
    configuration: AircraftSandboxConfiguration,
) -> AircraftCommandFunction:
    command = longitudinal_trim_command(configuration)

    def provider(_time_s: float, _state: AircraftState) -> AircraftControlCommand:
        return command

    return provider


def simulate_aircraft(
    configuration: AircraftSandboxConfiguration,
    command_function: AircraftCommandFunction | None = None,
) -> AircraftSandboxSimulation:
    """Propagate one aircraft case with custom fixed-step RK4 and quaternion projection."""
    started = time.perf_counter()
    model = FixedWingFlightModel(configuration)
    provider = command_function or _constant_trim_provider(configuration)
    state = aircraft_initial_state(configuration)
    time_s = 0.0
    step_index = 0
    output_stride = round(configuration.output_step_s / configuration.integration_step_s)
    time_values = [0.0]
    state_values = [state.copy()]
    event_values: list[EventOccurrence] = []
    reached_space = False
    impacted_ground = False
    fuel_depletion_recorded = False
    previous_altitude = configuration.initial.altitude_m

    while time_s < configuration.duration_s:
        step_s = min(configuration.integration_step_s, configuration.duration_s - time_s)

        def derivative(stage_time_s: float, stage_values: FloatArray) -> FloatArray:
            stage_state = AircraftState.from_array(stage_values, normalize=True)
            return model.derivative(stage_time_s, stage_values, provider(stage_time_s, stage_state))

        next_state = project_aircraft_state(
            rk4_step(derivative, time_s, state, step_s), configuration
        )
        next_time = time_s + step_s
        next_altitude = float(np.linalg.norm(next_state[:3])) - configuration.planet.radius_m

        if not reached_space and previous_altitude < KARMAN_LINE_ALTITUDE_M <= next_altitude:
            fraction = (KARMAN_LINE_ALTITUDE_M - previous_altitude) / max(
                next_altitude - previous_altitude, 1.0e-12
            )
            event_values.append(
                EventOccurrence(
                    "karman_line_ascent",
                    time_s + fraction * step_s,
                    state + fraction * (next_state - state),
                )
            )
            reached_space = True

        if (
            not fuel_depletion_recorded
            and state[13] > configuration.mass.dry_mass_kg
            and next_state[13] <= configuration.mass.dry_mass_kg + 1.0e-9
        ):
            event_values.append(EventOccurrence("fuel_depleted", next_time, next_state.copy()))
            fuel_depletion_recorded = True

        if previous_altitude > 0.0 >= next_altitude:
            fraction = previous_altitude / max(previous_altitude - next_altitude, 1.0e-12)
            time_s += fraction * step_s
            state = project_aircraft_state(state + fraction * (next_state - state), configuration)
            event_values.append(EventOccurrence("ground_impact", time_s, state.copy()))
            if time_values[-1] < time_s:
                time_values.append(float(time_s))
                state_values.append(state.copy())
            impacted_ground = True
            break

        state = next_state
        time_s = next_time
        step_index += 1
        if step_index % output_stride == 0 or np.isclose(time_s, configuration.duration_s):
            time_values.append(float(time_s))
            state_values.append(state.copy())
        previous_altitude = next_altitude

    if time_values[-1] < time_s:
        time_values.append(float(time_s))
        state_values.append(state.copy())

    times = np.asarray(time_values, dtype=np.float64)
    states = np.vstack(state_values)
    initial_position = states[0, :3]
    sample_count = times.size
    columns = {
        name: np.zeros(sample_count, dtype=np.float64)
        for name in (
            "north_m",
            "east_m",
            "down_m",
            "x_inertial_m",
            "y_inertial_m",
            "z_inertial_m",
            "altitude_m",
            "latitude_deg",
            "longitude_ground_deg",
            "true_airspeed_mps",
            "ground_speed_mps",
            "vertical_speed_mps",
            "mach",
            "dynamic_pressure_pa",
            "angle_of_attack_deg",
            "sideslip_angle_deg",
            "stall_fraction",
            "stall_speed_mps",
            "stall_margin_mps",
            "lift_coefficient",
            "drag_coefficient",
            "pitch_moment_coefficient",
            "lift_n",
            "drag_n",
            "load_factor_g",
            "specific_force_g",
            "lift_over_weight",
            "roll_deg",
            "pitch_deg",
            "heading_deg",
            "flight_path_angle_deg",
            "roll_rate_degps",
            "pitch_rate_degps",
            "yaw_rate_degps",
            "aileron_deg",
            "elevator_deg",
            "rudder_deg",
            "throttle",
            "air_breathing_thrust_n",
            "rocket_thrust_n",
            "mass_kg",
            "fuel_fraction",
        )
    }
    stall_flags = np.zeros(sample_count, dtype=bool)
    for index, (sample_time, state_row) in enumerate(zip(times, states, strict=True)):
        typed_state = AircraftState.from_array(state_row, normalize=True)
        command = provider(float(sample_time), typed_state)
        loads = model.loads(float(sample_time), state_row, command)
        telemetry = aircraft_telemetry(
            model,
            float(sample_time),
            state_row,
            command,
            initial_position_inertial_m=initial_position,
        )
        lift = (
            loads.aerodynamic.dynamic_pressure_pa
            * configuration.geometry.wing_area_m2
            * loads.aerodynamic.lift_coefficient
        )
        drag = (
            loads.aerodynamic.dynamic_pressure_pa
            * configuration.geometry.wing_area_m2
            * loads.aerodynamic.drag_coefficient
        )
        columns["north_m"][index] = telemetry.north_m
        columns["east_m"][index] = telemetry.east_m
        columns["down_m"][index] = telemetry.down_m
        columns["x_inertial_m"][index] = typed_state.position_inertial_m[0]
        columns["y_inertial_m"][index] = typed_state.position_inertial_m[1]
        columns["z_inertial_m"][index] = typed_state.position_inertial_m[2]
        columns["altitude_m"][index] = telemetry.altitude_m
        columns["latitude_deg"][index] = np.rad2deg(loads.latitude_rad)
        columns["longitude_ground_deg"][index] = np.rad2deg(loads.longitude_ground_rad)
        columns["true_airspeed_mps"][index] = telemetry.true_airspeed_mps
        columns["ground_speed_mps"][index] = telemetry.ground_speed_mps
        columns["vertical_speed_mps"][index] = telemetry.vertical_speed_mps
        columns["mach"][index] = telemetry.mach
        columns["dynamic_pressure_pa"][index] = telemetry.dynamic_pressure_pa
        columns["angle_of_attack_deg"][index] = telemetry.angle_of_attack_deg
        columns["sideslip_angle_deg"][index] = telemetry.sideslip_angle_deg
        columns["stall_fraction"][index] = telemetry.stall_fraction
        columns["stall_speed_mps"][index] = telemetry.stall_speed_1g_mps
        columns["stall_margin_mps"][index] = telemetry.stall_margin_mps
        columns["lift_coefficient"][index] = telemetry.lift_coefficient
        columns["drag_coefficient"][index] = telemetry.drag_coefficient
        columns["pitch_moment_coefficient"][index] = telemetry.pitch_moment_coefficient
        columns["lift_n"][index] = lift
        columns["drag_n"][index] = drag
        columns["load_factor_g"][index] = telemetry.normal_load_g
        columns["specific_force_g"][index] = telemetry.specific_force_g
        columns["lift_over_weight"][index] = telemetry.lift_over_weight
        columns["roll_deg"][index] = telemetry.roll_deg
        columns["pitch_deg"][index] = telemetry.pitch_deg
        columns["heading_deg"][index] = telemetry.heading_deg
        columns["flight_path_angle_deg"][index] = telemetry.flight_path_angle_deg
        columns["roll_rate_degps"][index] = telemetry.roll_rate_degps
        columns["pitch_rate_degps"][index] = telemetry.pitch_rate_degps
        columns["yaw_rate_degps"][index] = telemetry.yaw_rate_degps
        columns["aileron_deg"][index] = np.rad2deg(typed_state.control_surface_rad[0])
        columns["elevator_deg"][index] = np.rad2deg(typed_state.control_surface_rad[1])
        columns["rudder_deg"][index] = np.rad2deg(typed_state.control_surface_rad[2])
        columns["throttle"][index] = telemetry.throttle
        columns["air_breathing_thrust_n"][index] = telemetry.air_breathing_thrust_n
        columns["rocket_thrust_n"][index] = telemetry.rocket_thrust_n
        columns["mass_kg"][index] = telemetry.mass_kg
        columns["fuel_fraction"][index] = telemetry.fuel_fraction
        stall_flags[index] = telemetry.stalled

    heading_unwrapped_rad = np.unwrap(np.deg2rad(columns["heading_deg"]))
    columns["turn_rate_degps"] = np.rad2deg(np.gradient(heading_unwrapped_rad, times))
    stalled_duration = float(np.trapezoid(stall_flags.astype(np.float64), times))
    if np.any(stall_flags):
        first_stall = int(np.flatnonzero(stall_flags)[0])
        event_values.append(
            EventOccurrence("stall_onset", float(times[first_stall]), states[first_stall].copy())
        )

    event_summary: list[dict[str, float | str]] = []
    for event in sorted(event_values, key=lambda item: item.time_s):
        event_summary.append(
            {
                "name": event.name,
                "time_s": event.time_s,
                "altitude_m": float(np.linalg.norm(event.state[:3]))
                - configuration.planet.radius_m,
            }
        )
    maximum_summary: dict[str, dict[str, float | str]] = {
        "maximum_altitude": {"value": float(np.max(columns["altitude_m"])), "unit": "m"},
        "maximum_airspeed": {
            "value": float(np.max(columns["true_airspeed_mps"])),
            "unit": "m/s",
        },
        "maximum_mach": {"value": float(np.max(columns["mach"])), "unit": "1"},
        "maximum_load_factor": {
            "value": float(np.max(np.abs(columns["load_factor_g"]))),
            "unit": "g",
        },
        "maximum_turn_rate": {
            "value": float(np.max(np.abs(columns["turn_rate_degps"]))),
            "unit": "deg/s",
        },
        "stall_duration": {"value": stalled_duration, "unit": "s"},
    }
    result = SimulationResult(
        configuration.name,
        times,
        columns,
        tuple(sorted(event_values, key=lambda item: item.time_s)),
        tuple(event_summary),
        maximum_summary,
        time.perf_counter() - started,
    )
    if reached_space:
        interpretation = (
            "The simulated vehicle crossed 100 km in the configured local physics model. "
            "This is a modeled boundary crossing, not proof of orbital insertion or design "
            "feasibility."
        )
    elif impacted_ground:
        interpretation = "The simulated trajectory ended at the spherical ground-impact boundary."
    else:
        interpretation = (
            "The aircraft remained above ground for the configured finite simulation horizon; "
            "no claim is made beyond that horizon."
        )
    return AircraftSandboxSimulation(
        configuration,
        result,
        states[-1].copy(),
        reached_space,
        impacted_ground,
        stalled_duration,
        interpretation,
    )


def aircraft_sandbox_payload(simulation: AircraftSandboxSimulation) -> dict[str, object]:
    """Return a deterministic, public-safe JSON interpretation report."""
    configuration = simulation.configuration
    return {
        "schema_version": "1.0",
        "scenario": configuration.name,
        "safety_scope": configuration.safety_scope,
        "vehicle": "Aquila-X1 fictional civilian research aircraft",
        "reached_100_km": simulation.reached_space,
        "ground_impact": simulation.impacted_ground,
        "stalled_duration_s": simulation.stalled_duration_s,
        "interpretation": simulation.interpretation,
        "events": list(simulation.result.event_summary),
        "maxima": simulation.result.maximum_summary,
        "limitations": [
            "All vehicle coefficients and mass properties are synthetic, not certification data.",
            "Stall is a deterministic coefficient-break model; spin and separated-flow CFD "
            "are omitted.",
            "The spherical-planet atmosphere is a fixed reference, not measured weather or "
            "space weather.",
            "Ground contact, landing gear, structural flexibility, heating, and ablation "
            "are omitted.",
            "Crossing 100 km does not imply a sustainable orbit; orbital energy must be "
            "assessed separately.",
        ],
    }


def write_aircraft_results(
    simulation: AircraftSandboxSimulation, output_directory: str | Path
) -> tuple[Path, Path, Path]:
    """Write trajectory CSV, standard summary, and scoped model report."""
    output = Path(output_directory)
    csv_path = write_result_csv(simulation.result, output / "aircraft_trajectory.csv")
    summary_path = write_summary_json(simulation.result, output / "aircraft_summary.json")
    report_path = output / "aircraft_model_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(aircraft_sandbox_payload(simulation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path, report_path
