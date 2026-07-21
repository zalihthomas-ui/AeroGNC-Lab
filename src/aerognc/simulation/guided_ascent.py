"""Pitch-plane ascent simulation and deterministic offline reference optimization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from aerognc.configuration.ascent_guidance_loader import AscentGuidanceConfiguration
from aerognc.environment.atmosphere import dynamic_pressure_pa, mach_number
from aerognc.gnc.ascent_guidance import (
    STANDARD_GRAVITY_MPS2,
    AscentGuidanceCommand,
    AscentGuidanceDecision,
    AscentGuidanceInputs,
    ConstraintAwareAscentGuidance,
    decision_from_vector,
    decision_vector,
)
from aerognc.mathematics.integrators import EventSpec, integrate_fixed_step
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.logging import SimulationResult


@dataclass(frozen=True, slots=True)
class GuidedAscentDiagnostics:
    """Instantaneous states, loads, commands, and limit flags in SI units."""

    altitude_m: float
    ground_range_m: float
    velocity_north_mps: float
    vertical_velocity_up_mps: float
    speed_mps: float
    acceleration_north_mps2: float
    acceleration_down_mps2: float
    mach: float
    dynamic_pressure_pa: float
    mass_kg: float
    propellant_mass_kg: float
    thrust_n: float
    drag_n: float
    normal_force_n: float
    proper_load_factor: float
    air_flight_path_angle_rad: float
    angle_of_attack_rad: float
    actual_elevation_rad: float
    command: AscentGuidanceCommand
    predicted_ballistic_apogee_m: float


@dataclass(frozen=True, slots=True)
class GuidedAscentRun:
    """One reference/governed simulation with quantitative constraint metrics."""

    decision: AscentGuidanceDecision
    governor_enabled: bool
    result: SimulationResult
    objective: float
    apogee_m: float
    apogee_error_m: float
    maximum_dynamic_pressure_pa: float
    maximum_proper_load_factor: float
    maximum_absolute_angle_of_attack_rad: float
    all_constraints_satisfied: bool


@dataclass(frozen=True, slots=True)
class GuidanceOptimizationEvaluation:
    """One cached deterministic direct-search evaluation."""

    evaluation_index: int
    decision: AscentGuidanceDecision
    objective: float
    apogee_m: float
    all_constraints_satisfied: bool


@dataclass(frozen=True, slots=True)
class AscentGuidanceOptimizationResult:
    """Reference comparison, optimum, and complete search audit trail."""

    configuration: AscentGuidanceConfiguration
    reference_run: GuidedAscentRun
    optimized_run: GuidedAscentRun
    evaluations: tuple[GuidanceOptimizationEvaluation, ...]


class GuidedAscentModel:
    """Fictional pitch-plane variable-mass plant with a pure guidance interface."""

    def __init__(
        self,
        configuration: AscentGuidanceConfiguration,
        decision: AscentGuidanceDecision,
        *,
        governor_enabled: bool,
    ) -> None:
        self.configuration = configuration
        self.decision = decision
        self.governor_enabled = governor_enabled
        self.guidance = ConstraintAwareAscentGuidance(configuration)
        self.vehicle = configuration.base_scenario.vehicle
        self.environment = configuration.base_scenario.environment

    def initial_state(self) -> FloatArray:
        """Return north/down position, velocity, propellant, and pitch elevation."""
        launch = self.configuration.base_scenario.launch
        elevation_rad = float(np.deg2rad(launch.elevation_deg))
        velocity_north = launch.initial_speed_mps * np.cos(elevation_rad)
        velocity_down = -launch.initial_speed_mps * np.sin(elevation_rad)
        return np.array(
            [
                launch.position_ned_m[0],
                launch.position_ned_m[2],
                velocity_north,
                velocity_down,
                self.vehicle.propulsion.propellant_mass_kg,
                self.configuration.reference_elevation_rad[0],
            ],
            dtype=np.float64,
        )

    def project_state(self, state: FloatArray) -> FloatArray:
        """Preserve physical propellant and pitch-elevation bounds after each step."""
        projected = state.copy()
        projected[4] = np.clip(projected[4], 0.0, self.vehicle.propulsion.propellant_mass_kg)
        projected[5] = np.clip(projected[5], 0.0, 0.5 * np.pi)
        return projected

    def _command(
        self,
        time_s: float,
        dynamic_pressure: float,
        air_flight_path_angle: float,
        mass_kg: float,
        nominal_thrust_n: float,
        aerodynamic_force_magnitude_n: float,
        predicted_apogee_m: float,
    ) -> AscentGuidanceCommand:
        inputs = AscentGuidanceInputs(
            time_s=time_s,
            dynamic_pressure_pa=dynamic_pressure,
            air_flight_path_angle_rad=air_flight_path_angle,
            mass_kg=mass_kg,
            nominal_thrust_n=nominal_thrust_n,
            aerodynamic_force_magnitude_n=aerodynamic_force_magnitude_n,
            predicted_ballistic_apogee_m=predicted_apogee_m,
        )
        if self.governor_enabled:
            return self.guidance.command(inputs, self.decision)
        elevation, throttle = self.guidance.reference(time_s, self.decision)
        return AscentGuidanceCommand(
            elevation_rad=elevation,
            throttle=throttle,
            reference_elevation_rad=elevation,
            reference_throttle=throttle,
            angle_of_attack_limited=False,
            dynamic_pressure_limited=False,
            proper_load_limited=False,
            apogee_limited=False,
        )

    def diagnostics(self, time_s: float, state: FloatArray) -> GuidedAscentDiagnostics:
        """Evaluate loads, commands, and accelerations without hidden state."""
        north_m, down_m, velocity_north, velocity_down, propellant_kg, elevation_rad = state
        altitude_m = -float(down_m)
        atmosphere = self.environment.atmosphere.properties(altitude_m)
        wind_ned = self.environment.wind.velocity_ned_mps(time_s, altitude_m)
        air_velocity = np.array(
            [velocity_north - wind_ned[0], velocity_down - wind_ned[2]],
            dtype=np.float64,
        )
        airspeed = float(np.linalg.norm(air_velocity))
        if airspeed > 5.0:
            air_direction = air_velocity / airspeed
            lift_direction = np.array([air_direction[1], -air_direction[0]])
            air_flight_path_angle = float(np.arctan2(-air_velocity[1], air_velocity[0]))
        else:
            air_direction = np.array([np.cos(elevation_rad), -np.sin(elevation_rad)])
            lift_direction = np.array([air_direction[1], -air_direction[0]])
            air_flight_path_angle = float(elevation_rad)
        mach = mach_number(airspeed, atmosphere.speed_of_sound_mps)
        dynamic_pressure = dynamic_pressure_pa(atmosphere.density_kgpm3, airspeed)
        alpha_rad = float(elevation_rad - air_flight_path_angle)
        coefficients = self.vehicle.aerodynamics.coefficients(mach, alpha_rad, 0.0)
        area_m2 = self.vehicle.aerodynamics.reference_area_m2
        drag_n = dynamic_pressure * area_m2 * max(0.0, coefficients.drag)
        normal_force_n = -dynamic_pressure * area_m2 * coefficients.normal
        aerodynamic_force = -drag_n * air_direction + normal_force_n * lift_direction
        aerodynamic_force_magnitude = float(np.linalg.norm(aerodynamic_force))

        propellant_kg = float(
            np.clip(propellant_kg, 0.0, self.vehicle.propulsion.propellant_mass_kg)
        )
        mass_kg = self.vehicle.mass_properties.dry_mass_kg + propellant_kg
        nominal_thrust = (
            self.vehicle.propulsion.thrust_at_time_n(time_s) if propellant_kg > 1.0e-10 else 0.0
        )
        gravity = self.environment.gravity.acceleration_ned_mps2(altitude_m)
        vertical_velocity_up = -float(velocity_down)
        predicted_apogee = altitude_m + max(0.0, vertical_velocity_up) ** 2 / (
            2.0 * max(gravity[2], 1.0e-6)
        )
        command = self._command(
            time_s,
            dynamic_pressure,
            air_flight_path_angle,
            mass_kg,
            nominal_thrust,
            aerodynamic_force_magnitude,
            predicted_apogee,
        )
        thrust_n = command.throttle * nominal_thrust
        thrust_direction = np.array([np.cos(elevation_rad), -np.sin(elevation_rad)])
        proper_force = thrust_n * thrust_direction + aerodynamic_force
        acceleration = proper_force / mass_kg + gravity[[0, 2]]
        return GuidedAscentDiagnostics(
            altitude_m=altitude_m,
            ground_range_m=float(abs(north_m)),
            velocity_north_mps=float(velocity_north),
            vertical_velocity_up_mps=vertical_velocity_up,
            speed_mps=float(np.hypot(velocity_north, velocity_down)),
            acceleration_north_mps2=float(acceleration[0]),
            acceleration_down_mps2=float(acceleration[1]),
            mach=mach,
            dynamic_pressure_pa=dynamic_pressure,
            mass_kg=mass_kg,
            propellant_mass_kg=propellant_kg,
            thrust_n=float(thrust_n),
            drag_n=float(drag_n),
            normal_force_n=float(normal_force_n),
            proper_load_factor=float(
                np.linalg.norm(proper_force) / (mass_kg * STANDARD_GRAVITY_MPS2)
            ),
            air_flight_path_angle_rad=air_flight_path_angle,
            angle_of_attack_rad=alpha_rad,
            actual_elevation_rad=float(elevation_rad),
            command=command,
            predicted_ballistic_apogee_m=float(predicted_apogee),
        )

    def derivative(self, time_s: float, state: FloatArray) -> FloatArray:
        """Return the six-state pitch-plane derivative."""
        diagnostic = self.diagnostics(time_s, state)
        nominal_mass_flow = self.vehicle.propulsion.mass_flow_rate_kgps(time_s)
        propellant_rate = (
            -diagnostic.command.throttle * nominal_mass_flow
            if diagnostic.propellant_mass_kg > 1.0e-10
            else 0.0
        )
        elevation_rate = np.clip(
            (diagnostic.command.elevation_rad - diagnostic.actual_elevation_rad)
            / self.configuration.pitch_time_constant_s,
            -self.configuration.pitch_rate_limit_radps,
            self.configuration.pitch_rate_limit_radps,
        )
        return np.array(
            [
                diagnostic.velocity_north_mps,
                -diagnostic.vertical_velocity_up_mps,
                diagnostic.acceleration_north_mps2,
                diagnostic.acceleration_down_mps2,
                propellant_rate,
                elevation_rate,
            ],
            dtype=np.float64,
        )


def _maximum_record(time_s: FloatArray, values: FloatArray, unit: str) -> dict[str, float | str]:
    index = int(np.argmax(values))
    return {"value": float(values[index]), "unit": unit, "time_s": float(time_s[index])}


def _objective(
    configuration: AscentGuidanceConfiguration,
    time_s: FloatArray,
    columns: dict[str, FloatArray],
) -> float:
    apogee = float(np.max(columns["altitude_m"]))
    powered_ascent = time_s <= configuration.base_scenario.vehicle.propulsion.burnout_time_s
    alpha_constraint_active = powered_ascent & (
        columns["dynamic_pressure_pa"] >= configuration.minimum_alpha_constraint_dynamic_pressure_pa
    )
    q_max = float(np.max(columns["dynamic_pressure_pa"][powered_ascent]))
    load_max = float(np.max(columns["proper_load_factor"][powered_ascent]))
    alpha_max = float(np.max(np.abs(columns["angle_of_attack_deg"][alpha_constraint_active])))
    alpha_limit_deg = float(np.rad2deg(configuration.maximum_angle_of_attack_rad))
    performance_cost = (
        (apogee - configuration.desired_apogee_m) / configuration.apogee_tolerance_m
    ) ** 2
    violation_cost = 200.0 * (
        max(0.0, q_max / configuration.maximum_dynamic_pressure_pa - 1.0) ** 2
        + max(0.0, load_max / configuration.maximum_proper_load_factor - 1.0) ** 2
        + max(0.0, alpha_max / alpha_limit_deg - 1.0) ** 2
    )
    return float(performance_cost + violation_cost)


def simulate_guided_ascent(
    configuration: AscentGuidanceConfiguration,
    decision: AscentGuidanceDecision,
    *,
    governor_enabled: bool = True,
) -> GuidedAscentRun:
    """Simulate one open-reference or online-governed ascent case."""
    model = GuidedAscentModel(
        configuration,
        decision,
        governor_enabled=governor_enabled,
    )
    burnout_time = model.vehicle.propulsion.burnout_time_s
    events = (
        EventSpec("motor_window_end", lambda time_s, _state: time_s - burnout_time, 1),
        EventSpec("apogee", lambda _time_s, state: float(state[3]), 1),
        EventSpec("ground_impact", lambda _time_s, state: -float(state[1]), -1, True),
    )
    start = perf_counter()
    integration = integrate_fixed_step(
        model.derivative,
        model.initial_state(),
        (0.0, configuration.maximum_time_s),
        configuration.simulation_step_s,
        events=events,
        state_projection=model.project_state,
    )
    execution_time_s = perf_counter() - start
    count = integration.time_s.size
    names = (
        "altitude_m",
        "ground_range_m",
        "velocity_north_mps",
        "vertical_velocity_up_mps",
        "speed_mps",
        "acceleration_north_mps2",
        "acceleration_down_mps2",
        "mach",
        "dynamic_pressure_pa",
        "mass_kg",
        "propellant_mass_kg",
        "thrust_n",
        "drag_n",
        "normal_force_n",
        "proper_load_factor",
        "flight_path_angle_deg",
        "angle_of_attack_deg",
        "actual_elevation_deg",
        "command_elevation_deg",
        "reference_elevation_deg",
        "throttle",
        "reference_throttle",
        "predicted_ballistic_apogee_m",
        "alpha_limiter_active",
        "max_q_limiter_active",
        "load_limiter_active",
        "apogee_limiter_active",
    )
    columns = {name: np.empty(count, dtype=np.float64) for name in names}
    for index, (time_s, state) in enumerate(
        zip(integration.time_s, integration.state, strict=True)
    ):
        item = model.diagnostics(float(time_s), state)
        columns["altitude_m"][index] = item.altitude_m
        columns["ground_range_m"][index] = item.ground_range_m
        columns["velocity_north_mps"][index] = item.velocity_north_mps
        columns["vertical_velocity_up_mps"][index] = item.vertical_velocity_up_mps
        columns["speed_mps"][index] = item.speed_mps
        columns["acceleration_north_mps2"][index] = item.acceleration_north_mps2
        columns["acceleration_down_mps2"][index] = item.acceleration_down_mps2
        columns["mach"][index] = item.mach
        columns["dynamic_pressure_pa"][index] = item.dynamic_pressure_pa
        columns["mass_kg"][index] = item.mass_kg
        columns["propellant_mass_kg"][index] = item.propellant_mass_kg
        columns["thrust_n"][index] = item.thrust_n
        columns["drag_n"][index] = item.drag_n
        columns["normal_force_n"][index] = item.normal_force_n
        columns["proper_load_factor"][index] = item.proper_load_factor
        columns["flight_path_angle_deg"][index] = np.rad2deg(item.air_flight_path_angle_rad)
        columns["angle_of_attack_deg"][index] = np.rad2deg(item.angle_of_attack_rad)
        columns["actual_elevation_deg"][index] = np.rad2deg(item.actual_elevation_rad)
        columns["command_elevation_deg"][index] = np.rad2deg(item.command.elevation_rad)
        columns["reference_elevation_deg"][index] = np.rad2deg(item.command.reference_elevation_rad)
        columns["throttle"][index] = item.command.throttle
        columns["reference_throttle"][index] = item.command.reference_throttle
        columns["predicted_ballistic_apogee_m"][index] = item.predicted_ballistic_apogee_m
        columns["alpha_limiter_active"][index] = float(item.command.angle_of_attack_limited)
        columns["max_q_limiter_active"][index] = float(item.command.dynamic_pressure_limited)
        columns["load_limiter_active"][index] = float(item.command.proper_load_limited)
        columns["apogee_limiter_active"][index] = float(item.command.apogee_limited)

    event_summary: list[dict[str, float | str]] = []
    for event in integration.events:
        item = model.diagnostics(event.time_s, event.state)
        event_summary.append(
            {
                "name": event.name,
                "time_s": event.time_s,
                "altitude_m": item.altitude_m,
                "ground_range_m": item.ground_range_m,
                "speed_mps": item.speed_mps,
            }
        )
    powered_ascent = integration.time_s <= burnout_time
    alpha_constraint_active = powered_ascent & (
        columns["dynamic_pressure_pa"] >= configuration.minimum_alpha_constraint_dynamic_pressure_pa
    )
    maximum_summary = {
        "altitude": _maximum_record(integration.time_s, columns["altitude_m"], "m"),
        "powered_ascent_dynamic_pressure": _maximum_record(
            integration.time_s[powered_ascent],
            columns["dynamic_pressure_pa"][powered_ascent],
            "Pa",
        ),
        "powered_ascent_proper_load": _maximum_record(
            integration.time_s[powered_ascent],
            columns["proper_load_factor"][powered_ascent],
            "g0",
        ),
        "powered_ascent_absolute_angle_of_attack": _maximum_record(
            integration.time_s[alpha_constraint_active],
            np.abs(columns["angle_of_attack_deg"][alpha_constraint_active]),
            "deg",
        ),
    }
    result = SimulationResult(
        scenario_name=configuration.name,
        time_s=integration.time_s,
        columns=columns,
        events=integration.events,
        event_summary=tuple(event_summary),
        maximum_summary=maximum_summary,
        execution_time_s=execution_time_s,
    )
    apogee = float(np.max(columns["altitude_m"]))
    q_max = float(np.max(columns["dynamic_pressure_pa"][powered_ascent]))
    load_max = float(np.max(columns["proper_load_factor"][powered_ascent]))
    alpha_max_rad = float(
        np.deg2rad(np.max(np.abs(columns["angle_of_attack_deg"][alpha_constraint_active])))
    )
    constraints_satisfied = bool(
        q_max <= configuration.maximum_dynamic_pressure_pa * (1.0 + 1.0e-6)
        and load_max <= configuration.maximum_proper_load_factor * (1.0 + 1.0e-6)
        and alpha_max_rad <= configuration.maximum_angle_of_attack_rad * (1.0 + 1.0e-6)
    )
    return GuidedAscentRun(
        decision=decision,
        governor_enabled=governor_enabled,
        result=result,
        objective=_objective(configuration, integration.time_s, columns),
        apogee_m=apogee,
        apogee_error_m=apogee - configuration.desired_apogee_m,
        maximum_dynamic_pressure_pa=q_max,
        maximum_proper_load_factor=load_max,
        maximum_absolute_angle_of_attack_rad=alpha_max_rad,
        all_constraints_satisfied=constraints_satisfied,
    )


def optimize_ascent_guidance(
    configuration: AscentGuidanceConfiguration,
) -> AscentGuidanceOptimizationResult:
    """Run bounded coordinate search over two documented reference parameters."""
    reference_decision = AscentGuidanceDecision(0.0, 1.0)
    reference_run = simulate_guided_ascent(
        configuration, reference_decision, governor_enabled=False
    )
    lower = np.array(
        [
            configuration.elevation_offset_bounds_rad[0],
            configuration.throttle_scale_bounds[0],
        ]
    )
    upper = np.array(
        [
            configuration.elevation_offset_bounds_rad[1],
            configuration.throttle_scale_bounds[1],
        ]
    )
    best_vector = np.clip(decision_vector(reference_decision), lower, upper)
    best_run = simulate_guided_ascent(
        configuration, decision_from_vector(best_vector), governor_enabled=True
    )
    best_key = (float(best_vector[0]), float(best_vector[1]))
    cache: dict[tuple[float, float], GuidedAscentRun] = {best_key: best_run}
    history: list[GuidanceOptimizationEvaluation] = [
        GuidanceOptimizationEvaluation(
            0,
            best_run.decision,
            best_run.objective,
            best_run.apogee_m,
            best_run.all_constraints_satisfied,
        )
    ]
    steps = np.array(
        [configuration.initial_elevation_step_rad, configuration.initial_throttle_step]
    )
    for _iteration in range(configuration.maximum_optimizer_iterations):
        improved = False
        for dimension in range(2):
            for direction in (-1.0, 1.0):
                candidate_vector = best_vector.copy()
                candidate_vector[dimension] += direction * steps[dimension]
                candidate_vector = np.clip(candidate_vector, lower, upper)
                key = (float(candidate_vector[0]), float(candidate_vector[1]))
                if key in cache:
                    candidate_run = cache[key]
                else:
                    candidate_run = simulate_guided_ascent(
                        configuration,
                        decision_from_vector(candidate_vector),
                        governor_enabled=True,
                    )
                    cache[key] = candidate_run
                    history.append(
                        GuidanceOptimizationEvaluation(
                            len(history),
                            candidate_run.decision,
                            candidate_run.objective,
                            candidate_run.apogee_m,
                            candidate_run.all_constraints_satisfied,
                        )
                    )
                if candidate_run.objective + 1.0e-12 < best_run.objective:
                    best_vector = candidate_vector
                    best_run = candidate_run
                    improved = True
        if not improved:
            steps *= 0.5
    return AscentGuidanceOptimizationResult(
        configuration=configuration,
        reference_run=reference_run,
        optimized_run=best_run,
        evaluations=tuple(history),
    )
