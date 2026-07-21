"""Configured closed-loop 6-DOF fictional ascent simulation."""

from time import perf_counter

import numpy as np

from aerognc.configuration.six_dof_loader import SixDofConfiguration
from aerognc.dynamics.six_dof import RigidBodyFlightModel
from aerognc.dynamics.state import SixDofState, project_six_dof_quaternion
from aerognc.mathematics.integrators import EventOccurrence, rk4_step
from aerognc.mathematics.quaternion import (
    euler321_to_quaternion,
    quaternion_to_dcm,
    quaternion_to_euler321,
)
from aerognc.simulation.logging import SimulationResult
from aerognc.vehicle.actuators import FirstOrderActuator


def _maximum_record(time_s: np.ndarray, values: np.ndarray, unit: str) -> dict[str, float | str]:
    index = int(np.argmax(values))
    return {"value": float(values[index]), "unit": unit, "time_s": float(time_s[index])}


def simulate_six_dof(configuration: SixDofConfiguration) -> SimulationResult:
    """Run a sampled quaternion-attitude-hold ascent with bounded actuators."""
    sample_count = int(np.floor(configuration.duration_s / configuration.step_s + 0.5)) + 1
    time_s = np.linspace(0.0, configuration.duration_s, sample_count)
    initial_quaternion = euler321_to_quaternion(
        *(float(np.deg2rad(value)) for value in configuration.initial_euler321_deg)
    )
    initial_direction_ned = quaternion_to_dcm(initial_quaternion)[:, 0]
    state = SixDofState(
        position_ned_m=np.asarray(configuration.initial_position_ned_m, dtype=np.float64),
        velocity_ned_mps=configuration.initial_speed_mps * initial_direction_ned,
        quaternion_nb=initial_quaternion,
        angular_rate_body_radps=np.deg2rad(configuration.initial_angular_rate_body_degps),
    ).as_array()
    states = np.empty((sample_count, 13), dtype=np.float64)
    actuator_commands = np.zeros((sample_count, 3), dtype=np.float64)
    actuator_moments = np.zeros((sample_count, 3), dtype=np.float64)
    requested_moments = np.zeros((sample_count, 3), dtype=np.float64)
    attitude_error_deg = np.zeros(sample_count)
    alpha_deg = np.zeros(sample_count)
    beta_deg = np.zeros(sample_count)
    mach = np.zeros(sample_count)
    dynamic_pressure = np.zeros(sample_count)
    mass = np.zeros(sample_count)
    thrust = np.zeros(sample_count)
    drag = np.zeros(sample_count)
    actuators = [FirstOrderActuator(configuration.base.vehicle.actuator_limits) for _ in range(3)]
    model = RigidBodyFlightModel(configuration.base.vehicle, configuration.base.environment)
    start = perf_counter()

    for index, current_time_s in enumerate(time_s):
        states[index] = state
        reference_quaternion = configuration.reference_schedule.quaternion_at_time_nb(
            float(current_time_s)
        )
        rigid_state = SixDofState.from_array(state)
        requested_moments[index] = configuration.controller.command_moment_body_nm(
            reference_quaternion,
            rigid_state.quaternion_nb,
            rigid_state.angular_rate_body_radps,
        )
        desired_commands = configuration.base.vehicle.actuator_allocator.allocate(
            requested_moments[index]
        )
        actuator_commands[index] = np.array(
            [
                actuator.update(float(desired_commands[channel]), configuration.step_s)
                for channel, actuator in enumerate(actuators)
            ]
        )
        actuator_moments[index] = (
            configuration.base.vehicle.actuator_allocator.achieved_moment_body_nm(
                actuator_commands[index]
            )
        )
        loads = model.loads(float(current_time_s), state, actuator_moments[index])
        alpha_deg[index] = np.rad2deg(loads.aerodynamic.alpha_rad)
        beta_deg[index] = np.rad2deg(loads.aerodynamic.beta_rad)
        mach[index] = loads.aerodynamic.mach
        dynamic_pressure[index] = loads.aerodynamic.dynamic_pressure_pa
        mass[index] = loads.mass_properties.mass_kg
        thrust[index] = float(np.linalg.norm(loads.thrust_body_n))
        drag[index] = (
            loads.aerodynamic.dynamic_pressure_pa
            * configuration.base.vehicle.aerodynamics.reference_area_m2
            * loads.aerodynamic.coefficients.drag
        )
        dot = float(np.clip(abs(reference_quaternion @ rigid_state.quaternion_nb), 0.0, 1.0))
        attitude_error_deg[index] = np.rad2deg(2.0 * np.arccos(dot))
        if index == sample_count - 1:
            continue

        held_moment = actuator_moments[index].copy()

        def derivative(
            stage_time_s: float,
            stage_state: np.ndarray,
            applied_moment: np.ndarray = held_moment,
        ) -> np.ndarray:
            return model.derivative(stage_time_s, stage_state, applied_moment)

        state = project_six_dof_quaternion(
            rk4_step(derivative, float(current_time_s), state, configuration.step_s)
        )

    execution_time_s = perf_counter() - start
    position = states[:, :3]
    velocity = states[:, 3:6]
    quaternion = states[:, 6:10]
    angular_rate = states[:, 10:13]
    euler_rad = np.array([quaternion_to_euler321(value) for value in quaternion])
    reference_euler_rad = np.array(
        [configuration.reference_schedule.euler_at_time_rad(float(value)) for value in time_s]
    )
    speed = np.linalg.norm(velocity, axis=1)
    columns = {
        "north_m": position[:, 0].copy(),
        "east_m": position[:, 1].copy(),
        "down_m": position[:, 2].copy(),
        "altitude_m": -position[:, 2].copy(),
        "velocity_north_mps": velocity[:, 0].copy(),
        "velocity_east_mps": velocity[:, 1].copy(),
        "velocity_down_mps": velocity[:, 2].copy(),
        "total_velocity_mps": speed,
        "quaternion_q0": quaternion[:, 0].copy(),
        "quaternion_q1": quaternion[:, 1].copy(),
        "quaternion_q2": quaternion[:, 2].copy(),
        "quaternion_q3": quaternion[:, 3].copy(),
        "quaternion_norm": np.linalg.norm(quaternion, axis=1),
        "roll_deg": np.rad2deg(euler_rad[:, 0]),
        "pitch_deg": np.rad2deg(euler_rad[:, 1]),
        "yaw_deg": np.rad2deg(euler_rad[:, 2]),
        "reference_roll_deg": np.rad2deg(reference_euler_rad[:, 0]),
        "reference_pitch_deg": np.rad2deg(reference_euler_rad[:, 1]),
        "reference_yaw_deg": np.rad2deg(reference_euler_rad[:, 2]),
        "roll_rate_degps": np.rad2deg(angular_rate[:, 0]),
        "pitch_rate_degps": np.rad2deg(angular_rate[:, 1]),
        "yaw_rate_degps": np.rad2deg(angular_rate[:, 2]),
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
        "control_roll_moment_nm": actuator_moments[:, 0].copy(),
        "control_pitch_moment_nm": actuator_moments[:, 1].copy(),
        "control_yaw_moment_nm": actuator_moments[:, 2].copy(),
    }
    burnout_time_s = configuration.base.vehicle.propulsion.burnout_time_s
    burnout_state = np.array(
        [np.interp(burnout_time_s, time_s, states[:, component]) for component in range(13)]
    )
    burnout_event = EventOccurrence("burnout", burnout_time_s, burnout_state)
    burnout_altitude = float(np.interp(burnout_time_s, time_s, columns["altitude_m"]))
    maximum_summary = {
        "altitude": _maximum_record(time_s, columns["altitude_m"], "m"),
        "speed": _maximum_record(time_s, speed, "m/s"),
        "attitude_error": _maximum_record(time_s, attitude_error_deg, "deg"),
        "angular_rate": _maximum_record(
            time_s, np.linalg.norm(np.rad2deg(angular_rate), axis=1), "deg/s"
        ),
        "mach": _maximum_record(time_s, mach, "1"),
        "dynamic_pressure": _maximum_record(time_s, dynamic_pressure, "Pa"),
    }
    return SimulationResult(
        scenario_name=configuration.name,
        time_s=time_s,
        columns=columns,
        events=(burnout_event,),
        event_summary=(
            {
                "name": "burnout",
                "time_s": burnout_time_s,
                "altitude_m": burnout_altitude,
                "ground_range_m": float(
                    np.hypot(
                        np.interp(burnout_time_s, time_s, columns["north_m"]),
                        np.interp(burnout_time_s, time_s, columns["east_m"]),
                    )
                ),
                "speed_mps": float(
                    np.interp(burnout_time_s, time_s, columns["total_velocity_mps"])
                ),
            },
        ),
        maximum_summary=maximum_summary,
        execution_time_s=execution_time_s,
    )
