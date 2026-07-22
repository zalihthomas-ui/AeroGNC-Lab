"""Vehicle backends for the waypoint GNC loop.

`VehicleBackend` is the swap point that lets the same GNC logic drive the internal
simulator now and JSBSim / ArduPilot-SITL / PX4-SITL / hardware later. Each backend
declares the *command level* it accepts.

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
plant; the project's higher-fidelity 18-state `vehicle/fixed_wing.py` is the
intended higher-fidelity backend (integration hook, TODO Phase 10.2).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from aerognc.mathematics.local_frame import wrap_to_pi
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.mathematics.vectors import FloatArray
from aerognc.navigation.state import FlightEnvironment, NavigationState
from aerognc.vehicle.control_surfaces import SurfaceDeflections


class CommandLevel(StrEnum):
    """The lowest command level a backend accepts."""

    RAW_ACTUATOR = "raw_actuator"
    BODY_RATE = "body_rate"
    ATTITUDE = "attitude"
    VELOCITY = "velocity"
    POSITION = "position"
    MISSION_WAYPOINT = "mission_waypoint"


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
