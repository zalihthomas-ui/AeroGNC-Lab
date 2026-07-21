"""Near-planet free, two-body, restricted, full N-body, and decay simulations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.astrodynamics.bodies import CircularOrbitBody, PrimaryBody
from aerognc.astrodynamics.dynamics import RestrictedNBodyModel
from aerognc.astrodynamics.full_n_body import (
    GRAVITATIONAL_CONSTANT_M3_KG_S2,
    FullNBodyModel,
    MassiveBody,
)
from aerognc.astrodynamics.orbital_elements import (
    ClassicalOrbitalElements,
    elements_to_state,
    state_to_elements,
)
from aerognc.astrodynamics.perturbations import j2_acceleration_mps2
from aerognc.configuration.orbit_sandbox_loader import (
    OrbitModelName,
    OrbitSandboxConfiguration,
)
from aerognc.environment.orbital_atmosphere import ReferenceOrbitalAtmosphere
from aerognc.mathematics.integrators import DerivativeFunction, EventOccurrence, rk4_step
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.logging import SimulationResult, write_result_csv, write_summary_json

STANDARD_GRAVITY_MPS2 = 9.80665
SECONDS_PER_DAY = 86_400.0

ORBIT_MODEL_DESCRIPTIONS: dict[OrbitModelName, str] = {
    "free": (
        "One moving body with no applied force: position changes linearly and velocity remains "
        "constant. This is the control case, not an orbit."
    ),
    "two_body": (
        "Conventional two-body relative orbit: a point satellite moves under one fixed spherical "
        "primary's gravity."
    ),
    "restricted_three_body": (
        "Restricted three-body problem: the satellite has negligible mass while the primary and "
        "first configured moon follow prescribed motion."
    ),
    "full_n_body": (
        "Full N-body problem: the primary, every configured moon, and the finite-mass satellite "
        "all accelerate one another through pairwise Newtonian gravity."
    ),
    "perturbed_decay": (
        "Perturbed low orbit: central gravity, J2, a rotating reference atmosphere, drag, mass, "
        "area, Cd, and optional idealized correction impulses affect the path."
    ),
}


@dataclass(frozen=True, slots=True)
class OrbitCorrectionBurn:
    """One idealized recircularization impulse and mass-accounting record."""

    time_s: float
    altitude_m: float
    delta_v_mps: float
    mass_before_kg: float
    mass_after_kg: float


@dataclass(frozen=True, slots=True)
class OrbitSandboxSimulation:
    """Configuration, numerical trajectory, and interpretation metadata."""

    configuration: OrbitSandboxConfiguration
    result: SimulationResult
    initial_speed_mps: float
    nominal_period_s: float | None
    correction_burns: tuple[OrbitCorrectionBurn, ...]
    reentered: bool
    escaped: bool
    survival_statement: str


def circular_speed_mps(gravitational_parameter_m3_s2: float, radius_m: float) -> float:
    """Return circular-orbit speed for a positive radius."""
    if not np.isfinite([gravitational_parameter_m3_s2, radius_m]).all():
        raise ValueError("circular-speed inputs must be finite")
    if gravitational_parameter_m3_s2 <= 0.0 or radius_m <= 0.0:
        raise ValueError("circular-speed inputs must be positive")
    return float(np.sqrt(gravitational_parameter_m3_s2 / radius_m))


def escape_speed_mps(gravitational_parameter_m3_s2: float, radius_m: float) -> float:
    """Return two-body escape speed for a positive radius."""
    return float(np.sqrt(2.0) * circular_speed_mps(gravitational_parameter_m3_s2, radius_m))


def _initial_relative_state(configuration: OrbitSandboxConfiguration) -> tuple[FloatArray, float]:
    radius_m = configuration.primary.radius_m + configuration.initial.altitude_m
    circular = circular_speed_mps(configuration.primary.gravitational_parameter_m3_s2, radius_m)
    if configuration.initial.speed_mode == "circular":
        speed_mps = circular
    elif configuration.initial.speed_mode == "escape":
        speed_mps = np.sqrt(2.0) * circular
    else:
        speed_mps = configuration.initial.custom_speed_mps
    position, circular_velocity = elements_to_state(
        ClassicalOrbitalElements(
            semi_major_axis_m=radius_m,
            eccentricity=0.0,
            inclination_rad=configuration.initial.inclination_rad,
            ascending_node_rad=configuration.initial.ascending_node_rad,
            argument_of_periapsis_rad=0.0,
            true_anomaly_rad=configuration.initial.phase_rad,
        ),
        configuration.primary.gravitational_parameter_m3_s2,
    )
    velocity = circular_velocity * (speed_mps / circular)
    return np.concatenate((position, velocity)), float(speed_mps)


def _secondary_body(configuration: OrbitSandboxConfiguration, index: int) -> CircularOrbitBody:
    secondary = configuration.secondaries[index]
    return CircularOrbitBody(
        name=secondary.name,
        role="background",
        gravitational_parameter_m3_s2=secondary.gravitational_parameter_m3_s2,
        radius_m=secondary.radius_m,
        semi_major_axis_m=secondary.orbital_radius_m,
        phase_at_epoch_rad=secondary.phase_at_epoch_rad,
        inclination_rad=secondary.inclination_rad,
        color=secondary.color,
    )


def _restricted_model(configuration: OrbitSandboxConfiguration) -> RestrictedNBodyModel:
    return RestrictedNBodyModel(
        PrimaryBody(
            configuration.primary.name,
            configuration.primary.gravitational_parameter_m3_s2,
            configuration.primary.radius_m,
            configuration.primary.color,
        ),
        (_secondary_body(configuration, 0),),
    )


def _full_n_body_initial_state(
    configuration: OrbitSandboxConfiguration,
    satellite_relative_state: FloatArray,
) -> tuple[FullNBodyModel, FloatArray]:
    primary_mass = (
        configuration.primary.gravitational_parameter_m3_s2 / GRAVITATIONAL_CONSTANT_M3_KG_S2
    )
    bodies = [MassiveBody(configuration.primary.name, primary_mass, configuration.primary.radius_m)]
    states = [np.zeros(6, dtype=np.float64)]
    for index, secondary in enumerate(configuration.secondaries):
        mass = secondary.gravitational_parameter_m3_s2 / GRAVITATIONAL_CONSTANT_M3_KG_S2
        bodies.append(MassiveBody(secondary.name, mass, secondary.radius_m))
        position, velocity = _secondary_body(configuration, index).state_at_time(
            0.0, configuration.primary.gravitational_parameter_m3_s2
        )
        states.append(np.concatenate((position, velocity)))
    bodies.append(
        MassiveBody(configuration.satellite.name, configuration.satellite.initial_mass_kg, 1.0)
    )
    states.append(satellite_relative_state.copy())
    masses = np.array([body.mass_kg for body in bodies], dtype=np.float64)
    matrix = np.vstack(states)
    matrix[:, :3] -= np.sum(masses[:, None] * matrix[:, :3], axis=0) / np.sum(masses)
    matrix[:, 3:] -= np.sum(masses[:, None] * matrix[:, 3:], axis=0) / np.sum(masses)
    return FullNBodyModel(tuple(bodies)), matrix.ravel()


def _relative_from_full(state: FloatArray) -> FloatArray:
    matrix = state.reshape(-1, 6)
    return np.concatenate((matrix[-1, :3] - matrix[0, :3], matrix[-1, 3:] - matrix[0, 3:]))


def _secondary_positions(
    configuration: OrbitSandboxConfiguration,
    model_name: OrbitModelName,
    time_s: float,
    internal_state: FloatArray,
) -> tuple[FloatArray, ...]:
    if model_name == "full_n_body":
        matrix = internal_state.reshape(-1, 6)
        return tuple(matrix[index, :3] - matrix[0, :3] for index in range(1, len(matrix) - 1))
    return tuple(
        _secondary_body(configuration, index).state_at_time(
            time_s, configuration.primary.gravitational_parameter_m3_s2
        )[0]
        for index in range(len(configuration.secondaries))
    )


def _central_derivative(mu_m3_s2: float, _time_s: float, state: FloatArray) -> FloatArray:
    radius = float(np.linalg.norm(state[:3]))
    if radius <= 0.0:
        raise FloatingPointError("central gravity reached zero radius")
    return np.concatenate((state[3:], -mu_m3_s2 * state[:3] / radius**3))


def _free_derivative(_time_s: float, state: FloatArray) -> FloatArray:
    return np.concatenate((state[3:], np.zeros(3, dtype=np.float64)))


def _orbital_diagnostics(
    position_m: FloatArray,
    velocity_mps: FloatArray,
    configuration: OrbitSandboxConfiguration,
) -> tuple[float, float, float, float, float]:
    radius = float(np.linalg.norm(position_m))
    altitude = radius - configuration.primary.radius_m
    speed = float(np.linalg.norm(velocity_mps))
    if configuration.model == "free":
        return altitude, speed, 0.0, altitude, altitude
    try:
        elements = state_to_elements(
            position_m,
            velocity_mps,
            configuration.primary.gravitational_parameter_m3_s2,
        )
    except ValueError:
        return altitude, speed, 1.0, altitude, altitude
    eccentricity = elements.eccentricity
    if elements.semi_major_axis_m > 0.0 and eccentricity < 1.0:
        perigee = elements.semi_major_axis_m * (1.0 - eccentricity) - configuration.primary.radius_m
        apogee = elements.semi_major_axis_m * (1.0 + eccentricity) - configuration.primary.radius_m
    else:
        perigee = altitude
        apogee = altitude
    return altitude, speed, eccentricity, float(perigee), float(apogee)


def _revolutions(positions_m: FloatArray) -> FloatArray:
    result = np.zeros(positions_m.shape[0], dtype=np.float64)
    for index in range(1, positions_m.shape[0]):
        before = positions_m[index - 1]
        after = positions_m[index]
        denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denominator > 0.0:
            angle = float(np.arccos(np.clip(np.dot(before, after) / denominator, -1.0, 1.0)))
            result[index] = result[index - 1] + angle / (2.0 * np.pi)
        else:
            result[index] = result[index - 1]
    return result


def simulate_orbit_sandbox(configuration: OrbitSandboxConfiguration) -> OrbitSandboxSimulation:
    """Propagate one validated orbit-sandbox configuration with custom RK4."""
    started = time.perf_counter()
    relative_initial, initial_speed = _initial_relative_state(configuration)
    mu = configuration.primary.gravitational_parameter_m3_s2
    atmosphere = ReferenceOrbitalAtmosphere(configuration.atmosphere_density_scale)
    current_mass_kg = configuration.satellite.initial_mass_kg
    restricted = _restricted_model(configuration)
    full_model: FullNBodyModel | None = None

    if configuration.model == "full_n_body":
        full_model, internal_state = _full_n_body_initial_state(configuration, relative_initial)
        derivative_function: DerivativeFunction = full_model.derivative
    else:
        internal_state = relative_initial.copy()
        if configuration.model == "free":
            derivative_function = _free_derivative
        elif configuration.model == "restricted_three_body":
            derivative_function = restricted.derivative
        elif configuration.model == "perturbed_decay":

            def perturbed_derivative(_time_s: float, state: FloatArray) -> FloatArray:
                position = state[:3]
                velocity = state[3:]
                radius = float(np.linalg.norm(position))
                altitude = radius - configuration.primary.radius_m
                acceleration = -mu * position / radius**3
                if configuration.primary.j2 > 0.0 and radius > configuration.primary.radius_m:
                    acceleration += j2_acceleration_mps2(
                        position,
                        mu,
                        configuration.primary.radius_m,
                        configuration.primary.j2,
                    )
                density = atmosphere.density_kgpm3(max(-500.0, altitude))
                rotation = np.array(
                    [0.0, 0.0, configuration.primary.rotation_rate_radps], dtype=np.float64
                )
                air_velocity = velocity - np.cross(rotation, position)
                airspeed = float(np.linalg.norm(air_velocity))
                if airspeed > 0.0 and density > 0.0:
                    ballistic_factor = (
                        0.5
                        * density
                        * configuration.satellite.drag_coefficient
                        * configuration.satellite.drag_area_m2
                        / current_mass_kg
                    )
                    acceleration -= ballistic_factor * airspeed * air_velocity
                return np.concatenate((velocity, acceleration))

            derivative_function = perturbed_derivative

        else:

            def two_body_derivative(time_s: float, state: FloatArray) -> FloatArray:
                return _central_derivative(mu, time_s, state)

            derivative_function = two_body_derivative

    logged_time: list[float] = []
    logged_relative: list[FloatArray] = []
    logged_mass: list[float] = []
    logged_secondaries: list[tuple[FloatArray, ...]] = []
    corrections: list[OrbitCorrectionBurn] = []
    events: list[EventOccurrence] = []
    output_stride = round(configuration.output_step_s / configuration.integration_step_s)
    time_s = 0.0
    step_index = 0

    def relative_state(state: FloatArray) -> FloatArray:
        return _relative_from_full(state) if configuration.model == "full_n_body" else state.copy()

    def log_sample(sample_time_s: float, state: FloatArray) -> None:
        relative = relative_state(state)
        if logged_time and np.isclose(sample_time_s, logged_time[-1]):
            logged_time[-1] = float(sample_time_s)
            logged_relative[-1] = relative
            logged_mass[-1] = current_mass_kg
            logged_secondaries[-1] = _secondary_positions(
                configuration, configuration.model, sample_time_s, state
            )
            return
        logged_time.append(float(sample_time_s))
        logged_relative.append(relative)
        logged_mass.append(current_mass_kg)
        logged_secondaries.append(
            _secondary_positions(configuration, configuration.model, sample_time_s, state)
        )

    log_sample(0.0, internal_state)
    previous_relative = relative_state(internal_state)
    previous_altitude = (
        float(np.linalg.norm(previous_relative[:3])) - configuration.primary.radius_m
    )
    previous_radial_rate = float(
        np.dot(previous_relative[:3], previous_relative[3:]) / np.linalg.norm(previous_relative[:3])
    )
    reentered = False
    escaped = False

    while time_s < configuration.duration_s:
        step_s = min(configuration.integration_step_s, configuration.duration_s - time_s)
        next_internal = rk4_step(derivative_function, time_s, internal_state, step_s)
        next_time = time_s + step_s
        next_relative = relative_state(next_internal)
        next_radius = float(np.linalg.norm(next_relative[:3]))
        next_altitude = next_radius - configuration.primary.radius_m
        next_radial_rate = float(np.dot(next_relative[:3], next_relative[3:]) / next_radius)

        if (
            configuration.model == "perturbed_decay"
            and configuration.correction.enabled
            and len(corrections) < configuration.correction.maximum_burns
            and previous_radial_rate < 0.0 <= next_radial_rate
            and next_altitude <= configuration.correction.trigger_altitude_m
        ):
            radial_unit = next_relative[:3] / next_radius
            tangent = next_relative[3:] - radial_unit * np.dot(next_relative[3:], radial_unit)
            tangent_speed = float(np.linalg.norm(tangent))
            desired_speed = circular_speed_mps(mu, next_radius)
            delta_v = max(0.0, desired_speed - tangent_speed)
            if delta_v > 1.0e-6 and tangent_speed > 0.0:
                mass_after = current_mass_kg / np.exp(
                    delta_v
                    / (
                        STANDARD_GRAVITY_MPS2
                        * configuration.satellite.correction_specific_impulse_s
                    )
                )
                if mass_after >= configuration.satellite.dry_mass_kg:
                    next_internal[3:6] += delta_v * tangent / tangent_speed
                    current_mass_before = current_mass_kg
                    current_mass_kg = float(mass_after)
                    next_relative = next_internal.copy()
                    burn = OrbitCorrectionBurn(
                        next_time,
                        next_altitude,
                        delta_v,
                        current_mass_before,
                        current_mass_kg,
                    )
                    corrections.append(burn)
                    events.append(
                        EventOccurrence(
                            f"correction_{len(corrections):02d}", next_time, next_relative.copy()
                        )
                    )

        if previous_altitude > configuration.reentry_altitude_m >= next_altitude:
            fraction = (previous_altitude - configuration.reentry_altitude_m) / max(
                previous_altitude - next_altitude, 1.0e-12
            )
            event_time = time_s + fraction * step_s
            internal_state = internal_state + fraction * (next_internal - internal_state)
            event_state = relative_state(internal_state)
            event_radius = float(np.linalg.norm(event_state[:3]))
            threshold_radius = configuration.primary.radius_m + configuration.reentry_altitude_m
            projected_position = event_state[:3] * (threshold_radius / event_radius)
            if configuration.model == "full_n_body":
                state_matrix = internal_state.reshape(-1, 6)
                state_matrix[-1, :3] = state_matrix[0, :3] + projected_position
            else:
                internal_state[:3] = projected_position
            event_state = relative_state(internal_state)
            events.append(EventOccurrence("reentry_threshold", event_time, event_state.copy()))
            time_s = event_time
            reentered = True
            log_sample(time_s, internal_state)
            break

        escape_radius = configuration.escape_radius_multiplier * configuration.primary.radius_m
        if float(np.linalg.norm(previous_relative[:3])) < escape_radius <= next_radius:
            events.append(EventOccurrence("escape_boundary", next_time, next_relative.copy()))
            internal_state = next_internal
            time_s = next_time
            escaped = True
            log_sample(time_s, internal_state)
            break

        internal_state = next_internal
        time_s = next_time
        step_index += 1
        if step_index % output_stride == 0 or np.isclose(time_s, configuration.duration_s):
            log_sample(time_s, internal_state)
        previous_relative = next_relative
        previous_altitude = next_altitude
        previous_radial_rate = next_radial_rate

    if logged_time[-1] < time_s:
        log_sample(time_s, internal_state)

    time_values = np.asarray(logged_time, dtype=np.float64)
    relative_values = np.vstack(logged_relative)
    positions = relative_values[:, :3]
    velocities = relative_values[:, 3:]
    altitudes = np.empty(time_values.size)
    speeds = np.empty(time_values.size)
    eccentricity = np.empty(time_values.size)
    perigee = np.empty(time_values.size)
    apogee = np.empty(time_values.size)
    density = np.zeros(time_values.size)
    airspeed = np.zeros(time_values.size)
    drag_acceleration = np.zeros(time_values.size)
    rotation = np.array([0.0, 0.0, configuration.primary.rotation_rate_radps])
    for index, (position, velocity) in enumerate(zip(positions, velocities, strict=True)):
        altitudes[index], speeds[index], eccentricity[index], perigee[index], apogee[index] = (
            _orbital_diagnostics(position, velocity, configuration)
        )
        if configuration.model == "perturbed_decay":
            density[index] = atmosphere.density_kgpm3(max(-500.0, altitudes[index]))
            relative_air = velocity - np.cross(rotation, position)
            airspeed[index] = np.linalg.norm(relative_air)
            drag_acceleration[index] = (
                0.5
                * density[index]
                * configuration.satellite.drag_coefficient
                * configuration.satellite.drag_area_m2
                * airspeed[index] ** 2
                / logged_mass[index]
            )
    columns: dict[str, FloatArray] = {
        "x_m": positions[:, 0],
        "y_m": positions[:, 1],
        "z_m": positions[:, 2],
        "vx_mps": velocities[:, 0],
        "vy_mps": velocities[:, 1],
        "vz_mps": velocities[:, 2],
        "radius_m": np.linalg.norm(positions, axis=1),
        "altitude_m": altitudes,
        "speed_mps": speeds,
        "eccentricity": eccentricity,
        "perigee_altitude_m": perigee,
        "apogee_altitude_m": apogee,
        "revolutions_completed": _revolutions(positions),
        "mass_kg": np.asarray(logged_mass, dtype=np.float64),
        "atmospheric_density_kgpm3": density,
        "relative_airspeed_mps": airspeed,
        "drag_acceleration_mps2": drag_acceleration,
    }
    for secondary_index, secondary in enumerate(configuration.secondaries):
        samples = np.vstack(
            [positions_at_time[secondary_index] for positions_at_time in logged_secondaries]
        )
        key = secondary.name.casefold().replace(" ", "_")
        columns[f"{key}_x_m"] = samples[:, 0]
        columns[f"{key}_y_m"] = samples[:, 1]
        columns[f"{key}_z_m"] = samples[:, 2]

    event_summary_values: list[dict[str, float | str]] = []
    for event in events:
        event_summary_values.append(
            {
                "name": event.name,
                "time_s": event.time_s,
                "time_days": event.time_s / SECONDS_PER_DAY,
                "altitude_m": float(np.linalg.norm(event.state[:3]))
                - configuration.primary.radius_m,
            }
        )
    event_summary = tuple(event_summary_values)
    maximum_summary: dict[str, dict[str, float | str]] = {
        "maximum_altitude": {"value": float(np.max(altitudes)), "unit": "m"},
        "minimum_altitude": {"value": float(np.min(altitudes)), "unit": "m"},
        "maximum_speed": {"value": float(np.max(speeds)), "unit": "m/s"},
        "maximum_eccentricity": {"value": float(np.max(eccentricity)), "unit": "1"},
        "revolutions": {"value": float(columns["revolutions_completed"][-1]), "unit": "rev"},
        "correction_delta_v": {
            "value": float(sum(item.delta_v_mps for item in corrections)),
            "unit": "m/s",
        },
    }
    result = SimulationResult(
        scenario_name=configuration.name,
        time_s=time_values,
        columns=columns,
        events=tuple(events),
        event_summary=event_summary,
        maximum_summary=maximum_summary,
        execution_time_s=time.perf_counter() - started,
    )
    simulated_days = time_values[-1] / SECONDS_PER_DAY
    if reentered:
        survival = (
            f"Reentry threshold reached after {simulated_days:.3f} modeled days under the "
            "selected reference-density assumptions."
        )
    elif escaped:
        survival = f"Escape boundary reached after {simulated_days:.3f} modeled days."
    else:
        qualifier = "without correction impulses" if not corrections else "with enabled corrections"
        survival = (
            f"No reentry occurred during {simulated_days:.3f} modeled days {qualifier}; "
            f"lifetime is therefore greater than {simulated_days:.3f} days, not proven infinite."
        )
    nominal_period = (
        None
        if configuration.model == "free"
        else 2.0
        * np.pi
        * np.sqrt((configuration.primary.radius_m + configuration.initial.altitude_m) ** 3 / mu)
    )
    return OrbitSandboxSimulation(
        configuration,
        result,
        initial_speed,
        None if nominal_period is None else float(nominal_period),
        tuple(corrections),
        reentered,
        escaped,
        survival,
    )


def orbit_sandbox_payload(simulation: OrbitSandboxSimulation) -> dict[str, object]:
    """Return a deterministic JSON-ready interpretation report."""
    configuration = simulation.configuration
    return {
        "schema_version": "1.0",
        "scenario": configuration.name,
        "model": configuration.model,
        "model_description": ORBIT_MODEL_DESCRIPTIONS[configuration.model],
        "safety_scope": configuration.safety_scope,
        "primary": configuration.primary.name,
        "satellite": configuration.satellite.name,
        "initial_altitude_m": configuration.initial.altitude_m,
        "initial_speed_mps": simulation.initial_speed_mps,
        "nominal_period_s": simulation.nominal_period_s,
        "simulated_duration_s": float(simulation.result.time_s[-1]),
        "reentered": simulation.reentered,
        "escaped": simulation.escaped,
        "survival_statement": simulation.survival_statement,
        "correction_burns": [
            {
                "time_s": item.time_s,
                "altitude_m": item.altitude_m,
                "delta_v_mps": item.delta_v_mps,
                "mass_before_kg": item.mass_before_kg,
                "mass_after_kg": item.mass_after_kg,
            }
            for item in simulation.correction_burns
        ],
        "events": list(simulation.result.event_summary),
        "maxima": simulation.result.maximum_summary,
        "limitations": [
            "The primary, moons, satellite, and density profile are fictional/synthetic.",
            "Thermospheric density is a fixed scalable reference, not space-weather forecasting.",
            "Reentry is a threshold crossing; heating, breakup, lift, and attitude are omitted.",
            "A no-reentry result proves survival only through the configured finite horizon.",
            "Correction impulses are idealized instantaneous recircularization events.",
        ],
    }


def write_orbit_sandbox_results(
    simulation: OrbitSandboxSimulation, output_directory: str | Path
) -> tuple[Path, Path, Path]:
    """Write trajectory, standard summary, and scoped orbit report."""
    output = Path(output_directory)
    csv_path = write_result_csv(simulation.result, output / "orbit_sandbox_trajectory.csv")
    summary_path = write_summary_json(simulation.result, output / "orbit_sandbox_summary.json")
    report_path = output / "orbit_sandbox_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(orbit_sandbox_payload(simulation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path, report_path
