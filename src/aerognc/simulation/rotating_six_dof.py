"""Closed-loop quaternion 6-DOF ascent on a rotating oblate planet."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from aerognc.configuration.rotating_six_dof_loader import RotatingSixDofConfiguration
from aerognc.dynamics.rotating_six_dof import (
    RotatingRigidBodyFlightModel,
    RotatingRigidBodyState,
)
from aerognc.mathematics.geodesy import (
    body_rotation_rate_ned,
    dcm_ecef_to_ned,
    dcm_inertial_to_ecef,
    ecef_to_geodetic,
    ecef_to_inertial_state,
    ned_position_to_ecef,
    transport_rate_ned,
)
from aerognc.mathematics.integrators import EventOccurrence, rk4_step
from aerognc.mathematics.quaternion import (
    dcm_to_quaternion,
    euler321_to_quaternion,
    normalize_quaternion,
    quaternion_to_dcm,
    quaternion_to_euler321,
)
from aerognc.simulation.logging import SimulationResult
from aerognc.vehicle.actuators import FirstOrderActuator


def _maximum_record(time_s: np.ndarray, values: np.ndarray, unit: str) -> dict[str, float | str]:
    index = int(np.argmax(values))
    return {"value": float(values[index]), "unit": unit, "time_s": float(time_s[index])}


def _project_quaternion(state: np.ndarray) -> np.ndarray:
    projected = np.asarray(state, dtype=np.float64).copy()
    projected[6:10] = normalize_quaternion(projected[6:10])
    return projected


def _initial_state(configuration: RotatingSixDofConfiguration) -> np.ndarray:
    six = configuration.six_dof
    rotating = configuration.rotating_planet
    site = rotating.launch_site
    planet = rotating.planet
    position_ecef_m = ned_position_to_ecef(
        six.initial_position_ned_m, site.geodetic, planet.ellipsoid
    )
    quaternion_nb = euler321_to_quaternion(
        *(float(np.deg2rad(value)) for value in six.initial_euler321_deg)
    )
    dcm_nb = quaternion_to_dcm(quaternion_nb)
    velocity_ned_mps = six.initial_speed_mps * dcm_nb[:, 0]
    dcm_ne = dcm_ecef_to_ned(site.geodetic.latitude_rad, site.geodetic.longitude_rad)
    position_inertial_m, velocity_inertial_mps = ecef_to_inertial_state(
        position_ecef_m,
        dcm_ne.T @ velocity_ned_mps,
        time_s=0.0,
        rotation_rate_radps=planet.rotation_rate_radps,
    )
    dcm_ib = dcm_ne.T @ dcm_nb
    quaternion_ib = dcm_to_quaternion(dcm_ib)
    angular_rate_navigation_body_radps = np.deg2rad(six.initial_angular_rate_body_degps)
    angular_rate_inertial_ned_radps = body_rotation_rate_ned(
        site.geodetic.latitude_rad, planet.rotation_rate_radps
    )
    angular_rate_inertial_body_radps = (
        angular_rate_navigation_body_radps + dcm_nb.T @ angular_rate_inertial_ned_radps
    )
    return RotatingRigidBodyState(
        position_inertial_m,
        velocity_inertial_mps,
        quaternion_ib,
        angular_rate_inertial_body_radps,
    ).as_array()


def _local_attitude_and_rate(
    configuration: RotatingSixDofConfiguration,
    time_s: float,
    state: RotatingRigidBodyState,
    velocity_ned_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planet = configuration.rotating_planet.planet
    dcm_ei = dcm_inertial_to_ecef(planet.rotation_rate_radps * time_s)
    position_ecef = dcm_ei @ state.position_inertial_m
    geodetic = ecef_to_geodetic(position_ecef, planet.ellipsoid)
    dcm_ne = dcm_ecef_to_ned(geodetic.latitude_rad, geodetic.longitude_rad)
    dcm_nb = dcm_ne @ dcm_ei @ quaternion_to_dcm(state.quaternion_ib)
    quaternion_nb = dcm_to_quaternion(dcm_nb)
    angular_rate_inertial_ned_radps = body_rotation_rate_ned(
        geodetic.latitude_rad, planet.rotation_rate_radps
    ) + transport_rate_ned(geodetic, velocity_ned_mps, planet.ellipsoid)
    angular_rate_navigation_body_radps = (
        state.angular_rate_inertial_body_radps - dcm_nb.T @ angular_rate_inertial_ned_radps
    )
    return quaternion_nb, angular_rate_navigation_body_radps, dcm_nb


def _reference_quaternion_ib(
    configuration: RotatingSixDofConfiguration,
    time_s: float,
    geodetic_latitude_rad: float,
    geodetic_longitude_rad: float,
) -> np.ndarray:
    planet = configuration.rotating_planet.planet
    reference_nb = configuration.six_dof.reference_schedule.quaternion_at_time_nb(time_s)
    dcm_ne = dcm_ecef_to_ned(geodetic_latitude_rad, geodetic_longitude_rad)
    dcm_ei = dcm_inertial_to_ecef(planet.rotation_rate_radps * time_s)
    dcm_ib = dcm_ei.T @ dcm_ne.T @ quaternion_to_dcm(reference_nb)
    return dcm_to_quaternion(dcm_ib)


def simulate_rotating_six_dof(configuration: RotatingSixDofConfiguration) -> SimulationResult:
    """Propagate inertial translation and attitude with rotating-atmosphere loads."""
    six = configuration.six_dof
    rotating = configuration.rotating_planet
    sample_count = int(np.floor(six.duration_s / six.step_s + 0.5)) + 1
    time_s = np.linspace(0.0, six.duration_s, sample_count)
    states = np.empty((sample_count, 13), dtype=np.float64)
    local_quaternion = np.empty((sample_count, 4), dtype=np.float64)
    local_rate = np.empty((sample_count, 3), dtype=np.float64)
    ecef_position = np.empty((sample_count, 3), dtype=np.float64)
    ecef_velocity = np.empty((sample_count, 3), dtype=np.float64)
    ned_position = np.empty((sample_count, 3), dtype=np.float64)
    ned_velocity = np.empty((sample_count, 3), dtype=np.float64)
    acceleration_ned = np.empty((sample_count, 3), dtype=np.float64)
    geodetic_deg_m = np.empty((sample_count, 3), dtype=np.float64)
    actuator_commands = np.zeros((sample_count, 3), dtype=np.float64)
    actuator_moments = np.zeros((sample_count, 3), dtype=np.float64)
    attitude_error_deg = np.zeros(sample_count)
    alpha_deg = np.zeros(sample_count)
    beta_deg = np.zeros(sample_count)
    mach = np.zeros(sample_count)
    dynamic_pressure = np.zeros(sample_count)
    mass = np.zeros(sample_count)
    thrust = np.zeros(sample_count)
    drag = np.zeros(sample_count)
    model = RotatingRigidBodyFlightModel(
        six.base.vehicle, six.base.environment, rotating.planet, rotating.launch_site
    )
    actuators = [FirstOrderActuator(six.base.vehicle.actuator_limits) for _ in range(3)]
    state = _initial_state(configuration)
    start = perf_counter()

    for index, current_time_s in enumerate(time_s):
        current_time = float(current_time_s)
        states[index] = state
        rigid_state = RotatingRigidBodyState.from_array(state)
        loads = model.loads(current_time, state, actuator_moments[max(index - 1, 0)])
        ecef_position[index] = loads.position_ecef_m
        ecef_velocity[index] = loads.velocity_ecef_mps
        ned_position[index] = loads.position_ned_m
        ned_velocity[index] = loads.velocity_ned_mps
        geodetic_deg_m[index] = (
            np.rad2deg(loads.geodetic.latitude_rad),
            np.rad2deg(loads.geodetic.longitude_rad),
            loads.geodetic.altitude_m,
        )
        quaternion_nb, rate_nb_body, _dcm_nb = _local_attitude_and_rate(
            configuration, current_time, rigid_state, loads.velocity_ned_mps
        )
        local_quaternion[index] = quaternion_nb
        local_rate[index] = rate_nb_body
        reference_ib = _reference_quaternion_ib(
            configuration,
            current_time,
            loads.geodetic.latitude_rad,
            loads.geodetic.longitude_rad,
        )
        requested_moment = six.controller.command_moment_body_nm(
            reference_ib, rigid_state.quaternion_ib, rate_nb_body
        )
        desired_commands = six.base.vehicle.actuator_allocator.allocate(requested_moment)
        actuator_commands[index] = np.array(
            [
                actuator.update(float(desired_commands[channel]), six.step_s)
                for channel, actuator in enumerate(actuators)
            ]
        )
        actuator_moments[index] = six.base.vehicle.actuator_allocator.achieved_moment_body_nm(
            actuator_commands[index]
        )
        loads = model.loads(current_time, state, actuator_moments[index])
        alpha_deg[index] = np.rad2deg(loads.aerodynamic.alpha_rad)
        beta_deg[index] = np.rad2deg(loads.aerodynamic.beta_rad)
        mach[index] = loads.aerodynamic.mach
        dynamic_pressure[index] = loads.aerodynamic.dynamic_pressure_pa
        mass[index] = loads.mass_properties.mass_kg
        thrust[index] = float(np.linalg.norm(loads.thrust_body_n))
        drag[index] = (
            loads.aerodynamic.dynamic_pressure_pa
            * six.base.vehicle.aerodynamics.reference_area_m2
            * loads.aerodynamic.coefficients.drag
        )
        reference_nb = six.reference_schedule.quaternion_at_time_nb(current_time)
        dot = float(np.clip(abs(reference_nb @ quaternion_nb), 0.0, 1.0))
        attitude_error_deg[index] = np.rad2deg(2.0 * np.arccos(dot))
        derivative_now = model.derivative(current_time, state, actuator_moments[index])
        dcm_ei = dcm_inertial_to_ecef(rotating.planet.rotation_rate_radps * current_time)
        dcm_ne = dcm_ecef_to_ned(loads.geodetic.latitude_rad, loads.geodetic.longitude_rad)
        acceleration_ned[index] = dcm_ne @ dcm_ei @ derivative_now[3:6]
        if index == sample_count - 1:
            continue
        held_moment = actuator_moments[index].copy()

        def derivative(
            stage_time_s: float,
            stage_state: np.ndarray,
            applied_moment: np.ndarray = held_moment,
        ) -> np.ndarray:
            return model.derivative(stage_time_s, stage_state, applied_moment)

        state = _project_quaternion(rk4_step(derivative, current_time, state, six.step_s))

    execution_time_s = perf_counter() - start
    euler_rad = np.array([quaternion_to_euler321(value) for value in local_quaternion])
    reference_euler_rad = np.array(
        [six.reference_schedule.euler_at_time_rad(float(value)) for value in time_s]
    )
    altitude = geodetic_deg_m[:, 2] - rotating.launch_site.geodetic.altitude_m
    speed_inertial = np.linalg.norm(states[:, 3:6], axis=1)
    speed_ground = np.linalg.norm(ecef_velocity, axis=1)
    columns = {
        "inertial_x_m": states[:, 0].copy(),
        "inertial_y_m": states[:, 1].copy(),
        "inertial_z_m": states[:, 2].copy(),
        "velocity_inertial_x_mps": states[:, 3].copy(),
        "velocity_inertial_y_mps": states[:, 4].copy(),
        "velocity_inertial_z_mps": states[:, 5].copy(),
        "ecef_x_m": ecef_position[:, 0].copy(),
        "ecef_y_m": ecef_position[:, 1].copy(),
        "ecef_z_m": ecef_position[:, 2].copy(),
        "velocity_ecef_x_mps": ecef_velocity[:, 0].copy(),
        "velocity_ecef_y_mps": ecef_velocity[:, 1].copy(),
        "velocity_ecef_z_mps": ecef_velocity[:, 2].copy(),
        "latitude_deg": geodetic_deg_m[:, 0].copy(),
        "longitude_deg": geodetic_deg_m[:, 1].copy(),
        "ellipsoid_altitude_m": geodetic_deg_m[:, 2].copy(),
        "altitude_m": altitude.copy(),
        "north_m": ned_position[:, 0].copy(),
        "east_m": ned_position[:, 1].copy(),
        "down_m": ned_position[:, 2].copy(),
        "velocity_north_mps": ned_velocity[:, 0].copy(),
        "velocity_east_mps": ned_velocity[:, 1].copy(),
        "velocity_down_mps": ned_velocity[:, 2].copy(),
        "inertial_speed_mps": speed_inertial,
        "total_velocity_mps": speed_ground,
        "acceleration_north_mps2": acceleration_ned[:, 0].copy(),
        "acceleration_east_mps2": acceleration_ned[:, 1].copy(),
        "acceleration_down_mps2": acceleration_ned[:, 2].copy(),
        "quaternion_q0": states[:, 6].copy(),
        "quaternion_q1": states[:, 7].copy(),
        "quaternion_q2": states[:, 8].copy(),
        "quaternion_q3": states[:, 9].copy(),
        "quaternion_norm": np.linalg.norm(states[:, 6:10], axis=1),
        "roll_deg": np.rad2deg(euler_rad[:, 0]),
        "pitch_deg": np.rad2deg(euler_rad[:, 1]),
        "yaw_deg": np.rad2deg(euler_rad[:, 2]),
        "reference_roll_deg": np.rad2deg(reference_euler_rad[:, 0]),
        "reference_pitch_deg": np.rad2deg(reference_euler_rad[:, 1]),
        "reference_yaw_deg": np.rad2deg(reference_euler_rad[:, 2]),
        "roll_rate_degps": np.rad2deg(local_rate[:, 0]),
        "pitch_rate_degps": np.rad2deg(local_rate[:, 1]),
        "yaw_rate_degps": np.rad2deg(local_rate[:, 2]),
        "attitude_error_deg": attitude_error_deg,
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
        "mach": mach,
        "dynamic_pressure_pa": dynamic_pressure,
        "mass_kg": mass,
        "thrust_n": thrust,
        "drag_n": drag,
        "actuator_roll_deg": np.rad2deg(actuator_commands[:, 0]),
        "actuator_pitch_deg": np.rad2deg(actuator_commands[:, 1]),
        "actuator_yaw_deg": np.rad2deg(actuator_commands[:, 2]),
    }
    burnout_time_s = six.base.vehicle.propulsion.burnout_time_s
    events: tuple[EventOccurrence, ...] = ()
    event_summary: tuple[dict[str, float | str], ...] = ()
    if burnout_time_s <= six.duration_s:
        burnout_state = np.array(
            [np.interp(burnout_time_s, time_s, states[:, component]) for component in range(13)]
        )
        events = (EventOccurrence("burnout", burnout_time_s, burnout_state),)
        event_summary = (
            {
                "name": "burnout",
                "time_s": burnout_time_s,
                "altitude_m": float(np.interp(burnout_time_s, time_s, altitude)),
                "ground_range_m": float(
                    np.interp(burnout_time_s, time_s, np.linalg.norm(ned_position[:, :2], axis=1))
                ),
                "speed_mps": float(np.interp(burnout_time_s, time_s, speed_ground)),
            },
        )
    maximum_summary = {
        "altitude": _maximum_record(time_s, altitude, "m"),
        "ground_range": _maximum_record(time_s, np.linalg.norm(ned_position[:, :2], axis=1), "m"),
        "inertial_speed": _maximum_record(time_s, speed_inertial, "m/s"),
        "attitude_error": _maximum_record(time_s, attitude_error_deg, "deg"),
        "quaternion_norm_error": _maximum_record(
            time_s, np.abs(columns["quaternion_norm"] - 1.0), "1"
        ),
        "dynamic_pressure": _maximum_record(time_s, dynamic_pressure, "Pa"),
    }
    return SimulationResult(
        scenario_name=configuration.name,
        time_s=time_s,
        columns=columns,
        events=events,
        event_summary=event_summary,
        maximum_summary=maximum_summary,
        execution_time_s=execution_time_s,
    )
