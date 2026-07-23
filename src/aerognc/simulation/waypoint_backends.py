"""Vehicle backends for the waypoint GNC loop.

`VehicleBackend` is the swap point that lets the same GNC logic drive either internal
simulator now and optional JSBSim / ArduPilot-SITL / PX4-SITL adapters later. Physical
hardware output is deliberately outside this contract. Each backend declares the
*command level* it accepts.

`InternalFixedWingBackend` is a reduced, flat-Earth 6-DOF-lite fixed-wing model
(the autopilot-model level in Beard & McLain, extended so control-surface
deflections — not the raw commands — drive the response, which makes actuator
failures observable). It integrates with fixed-step RK4:

    p_dot     = L_da*da - L_p*p             (roll accel from aileron, roll damping)
    q_dot     = M_de*de - M_q*q             (pitch accel from elevator, pitch damping)
    phi_dot   = p,   theta_dot = q
    psi_dot   = (g/Va)*tan(phi) + N_dr*dr   (coordinated turn + rudder yaw)
    Va_dot    = (Va_trim(throttle) - Va)/tau - k*g*sin(theta)
    pn_dot    = Va*cos(theta)*cos(psi) + w_n   (ground velocity = air + wind)
    pe_dot    = Va*cos(theta)*sin(psi) + w_e
    pd_dot    = -Va*sin(theta)

The sign convention matches the autopilot/control-surface modules (positive
aileron -> roll right, positive elevator -> pitch up, positive rudder -> yaw right).
This is a control-design/mission-level model, **not** a validated flight-dynamics
plant. `InternalCoefficientFixedWingBackend` adapts the project's nonlinear
coefficient-driven 18-state plant to the same simulation-only contract.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.mathematics.geodesy import dcm_inertial_to_ecef
from aerognc.mathematics.integrators import rk4_step
from aerognc.mathematics.local_frame import wrap_to_pi
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.mathematics.vectors import FloatArray
from aerognc.navigation.state import FlightEnvironment, NavigationState
from aerognc.vehicle.control_surfaces import SurfaceDeflections
from aerognc.vehicle.fixed_wing import (
    STANDARD_GRAVITY_MPS2,
    AircraftControlCommand,
    AircraftState,
    FixedWingFlightModel,
    aircraft_initial_state,
    initial_tangent_displacement_ned_m,
    local_ned_dcm_inertial,
    project_aircraft_state,
)


class CommandLevel(StrEnum):
    """The lowest command level a backend accepts."""

    RAW_ACTUATOR = "raw_actuator"
    BODY_RATE = "body_rate"
    ATTITUDE = "attitude"
    VELOCITY = "velocity"
    POSITION = "position"
    MISSION_WAYPOINT = "mission_waypoint"


class VehicleBackendKind(StrEnum):
    """Built-in vehicle models available to the waypoint runtime."""

    INTERNAL_REDUCED = "internal_reduced"
    INTERNAL_COEFFICIENT = "internal_coefficient"


@dataclass(frozen=True, slots=True)
class ReducedFixedWingParams:
    """Coefficients for the reduced fixed-wing model (SI units)."""

    roll_from_aileron: float = 45.0  # L_da [rad/s^2 per rad]
    roll_damping: float = 5.0  # L_p [1/s]
    pitch_from_elevator: float = 28.0  # M_de [rad/s^2 per rad]
    pitch_damping: float = 6.0  # M_q [1/s]
    yaw_from_rudder: float = 2.0  # N_dr [rad/s^2 per rad] (into psi_dot)
    airspeed_min_mps: float = 8.0
    airspeed_at_zero_throttle_mps: float = 12.0
    airspeed_at_full_throttle_mps: float = 45.0
    airspeed_time_constant_s: float = 3.0
    climb_speed_coupling: float = 0.5
    max_bank_for_turn_rad: float = float(np.deg2rad(60.0))

    def __post_init__(self) -> None:
        positives = [
            self.roll_from_aileron,
            self.roll_damping,
            self.pitch_from_elevator,
            self.pitch_damping,
            self.airspeed_time_constant_s,
        ]
        if not np.all(np.isfinite(positives)) or np.any(np.asarray(positives) <= 0.0):
            raise ValueError("reduced model coefficients must be positive and finite")
        if self.airspeed_at_full_throttle_mps <= self.airspeed_at_zero_throttle_mps:
            raise ValueError("full-throttle airspeed must exceed zero-throttle airspeed")


class VehicleBackend(ABC):
    """Abstract vehicle backend the GNC loop drives."""

    command_level: CommandLevel

    @abstractmethod
    def initialize(
        self, *, position_ned_m: FloatArray, heading_rad: float, airspeed_mps: float
    ) -> None:
        """Set the initial trimmed-ish state."""

    @abstractmethod
    def read_state(self) -> NavigationState:
        """Return the current truth state."""

    @abstractmethod
    def send_actuator_commands(self, deflections: SurfaceDeflections) -> None:
        """Provide the physical actuator deflections for the next step."""

    @abstractmethod
    def step(self, dt_s: float, environment: FlightEnvironment) -> None:
        """Integrate the vehicle forward by ``dt_s`` under ``environment``."""

    def shutdown(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release any backend resources (no-op for the internal model)."""

    def provenance(self) -> Mapping[str, object]:
        """Return backend identity suitable for a structured mission record."""
        return {
            "implementation": type(self).__name__,
            "command_level": self.command_level.value,
        }


# State layout: [pn, pe, pd, phi, theta, psi, Va, p, q]
_PN, _PE, _PD, _PHI, _THETA, _PSI, _VA, _P, _Q = range(9)


class InternalFixedWingBackend(VehicleBackend):
    """Reduced flat-Earth fixed-wing model driven by surface deflections."""

    command_level = CommandLevel.RAW_ACTUATOR

    def __init__(self, params: ReducedFixedWingParams | None = None) -> None:
        self.params = params or ReducedFixedWingParams()
        self._state = np.zeros(9, dtype=np.float64)
        self._deflections = SurfaceDeflections(0.0, 0.0, 0.0, 0.5, saturated=False)
        self._wind_ned = np.zeros(3, dtype=np.float64)

    def initialize(
        self, *, position_ned_m: FloatArray, heading_rad: float, airspeed_mps: float
    ) -> None:
        position = np.asarray(position_ned_m, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("position_ned_m must have shape (3,)")
        if not np.isfinite(airspeed_mps) or airspeed_mps <= 0.0:
            raise ValueError("airspeed_mps must be positive and finite")
        self._state = np.array(
            [
                position[0],
                position[1],
                position[2],
                0.0,
                0.0,
                wrap_to_pi(heading_rad),
                airspeed_mps,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    def send_actuator_commands(self, deflections: SurfaceDeflections) -> None:
        self._deflections = deflections

    def step(self, dt_s: float, environment: FlightEnvironment) -> None:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        self._wind_ned = environment.wind_ned_mps
        gravity = environment.gravity_mps2
        state = self._state
        k1 = self._derivative(state, gravity)
        k2 = self._derivative(state + 0.5 * dt_s * k1, gravity)
        k3 = self._derivative(state + 0.5 * dt_s * k2, gravity)
        k4 = self._derivative(state + dt_s * k3, gravity)
        state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state[_PSI] = wrap_to_pi(state[_PSI])
        state[_VA] = max(state[_VA], 0.0)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("internal backend produced a non-finite state")
        self._state = state

    def read_state(self) -> NavigationState:
        return NavigationState(
            position_ned_m=self._state[:3].copy(),
            velocity_ned_mps=self._ground_velocity(self._state),
            quaternion_nb=euler321_to_quaternion(
                float(self._state[_PHI]), float(self._state[_THETA]), float(self._state[_PSI])
            ),
            angular_rate_body_radps=np.array(
                [self._state[_P], self._state[_Q], self._yaw_rate(self._state, 9.80665)]
            ),
            airspeed_mps=float(self._state[_VA]),
        )

    # -- dynamics -------------------------------------------------------------
    def _derivative(self, state: FloatArray, gravity_mps2: float) -> FloatArray:
        params = self.params
        theta = state[_THETA]
        deflections = self._deflections

        derivative = np.zeros(9, dtype=np.float64)
        derivative[_P] = (
            params.roll_from_aileron * deflections.aileron_rad - params.roll_damping * state[_P]
        )
        derivative[_Q] = (
            params.pitch_from_elevator * deflections.elevator_rad - params.pitch_damping * state[_Q]
        )
        derivative[_PHI] = state[_P]
        derivative[_THETA] = state[_Q]
        derivative[_PSI] = self._yaw_rate(state, gravity_mps2)

        airspeed_trim = params.airspeed_at_zero_throttle_mps + deflections.throttle * (
            params.airspeed_at_full_throttle_mps - params.airspeed_at_zero_throttle_mps
        )
        derivative[_VA] = (airspeed_trim - state[_VA]) / params.airspeed_time_constant_s - (
            params.climb_speed_coupling * gravity_mps2 * np.sin(theta)
        )

        horizontal_speed = state[_VA] * np.cos(theta)
        derivative[_PN] = horizontal_speed * np.cos(state[_PSI]) + self._wind_ned[0]
        derivative[_PE] = horizontal_speed * np.sin(state[_PSI]) + self._wind_ned[1]
        derivative[_PD] = -state[_VA] * np.sin(theta)
        return derivative

    def _yaw_rate(self, state: FloatArray, gravity_mps2: float) -> float:
        params = self.params
        va = max(state[_VA], params.airspeed_min_mps)
        bank = float(
            np.clip(state[_PHI], -params.max_bank_for_turn_rad, params.max_bank_for_turn_rad)
        )
        rudder_yaw = params.yaw_from_rudder * self._deflections.rudder_rad
        return float((gravity_mps2 / va) * np.tan(bank) + rudder_yaw)

    def _ground_velocity(self, state: FloatArray) -> FloatArray:
        horizontal_speed = state[_VA] * np.cos(state[_THETA])
        return np.array(
            [
                horizontal_speed * np.cos(state[_PSI]) + self._wind_ned[0],
                horizontal_speed * np.sin(state[_PSI]) + self._wind_ned[1],
                -state[_VA] * np.sin(state[_THETA]),
            ],
            dtype=np.float64,
        )


class InternalCoefficientFixedWingBackend(VehicleBackend):
    """Adapter for the nonlinear coefficient-driven 18-state fictional aircraft.

    The waypoint runtime already owns actuator position/rate dynamics. Before every
    integration step, those physical deflections are copied into the plant state and
    supplied as matching normalized targets. This prevents an accidental second
    actuator lag while retaining the nonlinear aerodynamic, propulsion, rotating-
    planet, mass, and rigid-body dynamics.
    """

    command_level = CommandLevel.RAW_ACTUATOR

    def __init__(
        self,
        configuration: AircraftSandboxConfiguration,
        *,
        steady_wind_ned_mps: tuple[float, float, float] | None = None,
        wind_horizon_s: float | None = None,
    ) -> None:
        self.configuration = configuration
        try:
            self._configuration_sha256: str | None = hashlib.sha256(
                configuration.source_path.read_bytes()
            ).hexdigest()
        except OSError:
            self._configuration_sha256 = None
        configured_wind = (
            configuration.wind_north_mps,
            configuration.wind_east_mps,
            0.0,
        )
        wind = np.asarray(
            configured_wind if steady_wind_ned_mps is None else steady_wind_ned_mps,
            dtype=np.float64,
        )
        if wind.shape != (3,) or not np.all(np.isfinite(wind)):
            raise ValueError("steady_wind_ned_mps must contain three finite values")
        if wind_horizon_s is not None and (
            not np.isfinite(wind_horizon_s) or wind_horizon_s <= 0.0
        ):
            raise ValueError("wind_horizon_s must be positive and finite")
        self._steady_wind_ned_mps = wind.copy()
        self._wind_horizon_s = wind_horizon_s
        self._runtime_configuration = configuration
        self._model = FixedWingFlightModel(
            configuration,
            wind_horizon_s=wind_horizon_s,
            steady_wind_ned_mps=wind,
        )
        self._state = aircraft_initial_state(configuration)
        self._time_s = 0.0
        self._position_offset_ned_m = np.zeros(3, dtype=np.float64)
        self._initial_position_inertial_m = self._state[:3].copy()
        self._initial_ned_dcm_inertial = local_ned_dcm_inertial(self._initial_position_inertial_m)
        self._deflections = self._trim_deflections(self._state)
        self._command = self._command_from_deflections(self._deflections)
        self._initialized = False

    def initialize(
        self, *, position_ned_m: FloatArray, heading_rad: float, airspeed_mps: float
    ) -> None:
        position = np.asarray(position_ned_m, dtype=np.float64)
        values = np.asarray([heading_rad, airspeed_mps], dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_ned_m must contain three finite values")
        if not np.all(np.isfinite(values)) or airspeed_mps <= 0.0:
            raise ValueError("heading and positive airspeed must be finite")
        altitude_m = -float(position[2])
        if altitude_m < 0.0:
            raise ValueError("coefficient backend initial altitude must be nonnegative")

        initial = replace(
            self.configuration.initial,
            altitude_m=altitude_m,
            true_airspeed_mps=float(airspeed_mps),
            heading_rad=wrap_to_pi(float(heading_rad)),
        )
        runtime = replace(
            self.configuration,
            initial=initial,
            wind_north_mps=float(self._steady_wind_ned_mps[0]),
            wind_east_mps=float(self._steady_wind_ned_mps[1]),
        )
        model = FixedWingFlightModel(
            runtime,
            wind_horizon_s=self._wind_horizon_s,
            steady_wind_ned_mps=self._steady_wind_ned_mps,
        )
        state = aircraft_initial_state(runtime)
        if self._steady_wind_ned_mps[2] != 0.0:
            state[3:6] += local_ned_dcm_inertial(state[:3]) @ np.array(
                [0.0, 0.0, self._steady_wind_ned_mps[2]], dtype=np.float64
            )

        self._runtime_configuration = runtime
        self._model = model
        self._state = state
        self._time_s = 0.0
        self._position_offset_ned_m = position.copy()
        self._initial_position_inertial_m = state[:3].copy()
        self._initial_ned_dcm_inertial = local_ned_dcm_inertial(state[:3])
        self._deflections = self._trim_deflections(state)
        self._command = self._command_from_deflections(self._deflections)
        self._initialized = True

    def read_state(self) -> NavigationState:
        self._require_initialized()
        typed = AircraftState.from_array(self._state, normalize=True)
        displacement = initial_tangent_displacement_ned_m(
            typed.position_inertial_m,
            self._time_s,
            self._initial_position_inertial_m,
            self._runtime_configuration.planet.rotation_rate_radps,
        )
        rotation = dcm_inertial_to_ecef(
            self._runtime_configuration.planet.rotation_rate_radps * self._time_s
        )
        planet_rate = np.array([0.0, 0.0, self._runtime_configuration.planet.rotation_rate_radps])
        velocity_fixed = rotation @ (
            typed.velocity_inertial_mps - np.cross(planet_rate, typed.position_inertial_m)
        )
        velocity_ned = self._initial_ned_dcm_inertial.T @ velocity_fixed
        roll, pitch, heading = self._model.local_attitude_rad(self._state)
        loads = self._model.loads(self._time_s, self._state, self._command)
        return NavigationState(
            position_ned_m=self._position_offset_ned_m + displacement,
            velocity_ned_mps=velocity_ned,
            quaternion_nb=euler321_to_quaternion(roll, pitch, heading),
            angular_rate_body_radps=typed.angular_rate_body_radps.copy(),
            airspeed_mps=loads.aerodynamic.true_airspeed_mps,
        )

    def send_actuator_commands(self, deflections: SurfaceDeflections) -> None:
        limits = self._surface_limits_rad
        physical = np.array(
            [
                deflections.aileron_rad,
                deflections.elevator_rad,
                deflections.rudder_rad,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(physical)) or np.any(np.abs(physical) > limits + 1.0e-12):
            raise ValueError("surface deflection exceeds coefficient-plant limits")
        if not np.isfinite(deflections.throttle) or not 0.0 <= deflections.throttle <= 1.0:
            raise ValueError("coefficient-backend throttle must lie in [0, 1]")
        self._deflections = deflections
        self._command = self._command_from_deflections(deflections)

    def step(self, dt_s: float, environment: FlightEnvironment) -> None:
        self._require_initialized()
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if not np.allclose(
            environment.wind_ned_mps,
            self._steady_wind_ned_mps,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("coefficient backend environment wind differs from its fixed model")
        if not np.isclose(
            environment.gravity_mps2,
            STANDARD_GRAVITY_MPS2,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "coefficient backend derives gravity from its planet; interface gravity must be "
                "standard"
            )

        state = self._state.copy()
        state[14:17] = np.array(
            [
                self._deflections.aileron_rad,
                -self._deflections.elevator_rad,
                -self._deflections.rudder_rad,
            ]
        )
        state[17] = self._deflections.throttle

        def derivative(stage_time_s: float, stage_state: FloatArray) -> FloatArray:
            return self._model.derivative(stage_time_s, stage_state, self._command)

        next_state = project_aircraft_state(
            rk4_step(derivative, self._time_s, state, dt_s),
            self._runtime_configuration,
        )
        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError("coefficient backend produced a non-finite state")
        self._state = next_state
        self._time_s += dt_s

    def provenance(self) -> Mapping[str, object]:
        details = dict(super().provenance())
        details.update(
            {
                "model": "coefficient_driven_18_state",
                "aircraft_name": self.configuration.name,
                "aircraft_configuration": self.configuration.source_path.name,
                "aircraft_configuration_sha256": self._configuration_sha256,
                "aerodynamic_backend": self.configuration.aerodynamic_backend,
                "steady_wind_ned_mps": self._steady_wind_ned_mps.tolist(),
            }
        )
        return details

    @property
    def _surface_limits_rad(self) -> FloatArray:
        geometry = self._runtime_configuration.geometry
        return np.array(
            [
                geometry.aileron_limit_rad,
                geometry.elevator_limit_rad,
                geometry.rudder_limit_rad,
            ],
            dtype=np.float64,
        )

    def _command_from_deflections(self, deflections: SurfaceDeflections) -> AircraftControlCommand:
        limits = self._surface_limits_rad
        return AircraftControlCommand(
            roll=float(deflections.aileron_rad / limits[0]),
            pitch=float(deflections.elevator_rad / limits[1]),
            yaw=float(deflections.rudder_rad / limits[2]),
            throttle=deflections.throttle,
            rocket_assist=False,
        )

    @staticmethod
    def _trim_deflections(state: FloatArray) -> SurfaceDeflections:
        return SurfaceDeflections(
            aileron_rad=float(state[14]),
            elevator_rad=float(-state[15]),
            rudder_rad=float(-state[16]),
            throttle=float(state[17]),
            saturated=False,
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("coefficient backend must be initialized before use")
