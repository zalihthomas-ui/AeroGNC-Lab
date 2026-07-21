"""Vehicle navigation state and flight environment used across the GNC chain.

:class:`NavigationState` is the single state object guidance, control, safety, and
logging consume. It is frame-explicit (NED navigation, FRD body, Hamilton
``quaternion_nb``) and stores everything in SI units. Derived quantities the
controllers need (Euler angles, course, heading, groundspeed, climb rate,
altitude above the home datum) are computed on access so producers only need to
supply the primitive state.
"""

from dataclasses import dataclass

import numpy as np

from aerognc.dynamics.state import SixDofState
from aerognc.mathematics.quaternion import normalize_quaternion, quaternion_to_euler321
from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class NavigationState:
    """Estimated or true vehicle state in NED/FRD with SI units."""

    position_ned_m: FloatArray
    velocity_ned_mps: FloatArray
    quaternion_nb: FloatArray
    angular_rate_body_radps: FloatArray
    airspeed_mps: float
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_ned_m", as_vector(self.position_ned_m, 3, name="position_ned_m")
        )
        object.__setattr__(
            self, "velocity_ned_mps", as_vector(self.velocity_ned_mps, 3, name="velocity_ned_mps")
        )
        object.__setattr__(
            self, "quaternion_nb", normalize_quaternion(self.quaternion_nb)
        )
        object.__setattr__(
            self,
            "angular_rate_body_radps",
            as_vector(self.angular_rate_body_radps, 3, name="angular_rate_body_radps"),
        )
        if not np.isfinite(self.airspeed_mps) or self.airspeed_mps < 0.0:
            raise ValueError("airspeed_mps must be finite and nonnegative")

    @classmethod
    def from_six_dof(
        cls, state: SixDofState, airspeed_mps: float, *, valid: bool = True
    ) -> "NavigationState":
        """Build a navigation state from a rigid-body truth state and airspeed."""
        return cls(
            position_ned_m=state.position_ned_m,
            velocity_ned_mps=state.velocity_ned_mps,
            quaternion_nb=state.quaternion_nb,
            angular_rate_body_radps=state.angular_rate_body_radps,
            airspeed_mps=airspeed_mps,
            valid=valid,
        )

    @property
    def euler_rad(self) -> tuple[float, float, float]:
        """Return the (roll, pitch, yaw) 3-2-1 Euler angles [rad]."""
        roll, pitch, yaw = quaternion_to_euler321(self.quaternion_nb)
        return float(roll), float(pitch), float(yaw)

    @property
    def roll_rad(self) -> float:
        return self.euler_rad[0]

    @property
    def pitch_rad(self) -> float:
        return self.euler_rad[1]

    @property
    def yaw_rad(self) -> float:
        return self.euler_rad[2]

    @property
    def heading_rad(self) -> float:
        """Body yaw (nose direction), distinct from ground course [rad]."""
        return self.euler_rad[2]

    @property
    def altitude_m(self) -> float:
        """Height above the home datum (positive up) [m]."""
        return -float(self.position_ned_m[2])

    @property
    def groundspeed_mps(self) -> float:
        """Horizontal speed over the ground [m/s]."""
        return float(np.hypot(self.velocity_ned_mps[0], self.velocity_ned_mps[1]))

    @property
    def course_rad(self) -> float:
        """Direction of travel over the ground, clockwise from north [rad]."""
        return float(np.arctan2(self.velocity_ned_mps[1], self.velocity_ned_mps[0]))

    @property
    def climb_rate_mps(self) -> float:
        """Vertical speed, positive up [m/s]."""
        return -float(self.velocity_ned_mps[2])


@dataclass(frozen=True, slots=True)
class FlightEnvironment:
    """Environment the guidance/control layers read (wind + gravity)."""

    wind_ned_mps: FloatArray
    gravity_mps2: float = 9.80665

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "wind_ned_mps", as_vector(self.wind_ned_mps, 3, name="wind_ned_mps")
        )
        if not np.isfinite(self.gravity_mps2) or self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive and finite")

    @classmethod
    def calm(cls, gravity_mps2: float = 9.80665) -> "FlightEnvironment":
        """Return a calm-air environment (zero wind)."""
        return cls(wind_ned_mps=np.zeros(3, dtype=np.float64), gravity_mps2=gravity_mps2)
