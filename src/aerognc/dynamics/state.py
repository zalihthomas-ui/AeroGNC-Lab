"""Typed state views using the normative state-vector ordering."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.quaternion import normalize_quaternion
from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class ThreeDofState:
    """NED point-mass position [m] and velocity [m/s]."""

    position_ned_m: FloatArray
    velocity_ned_mps: FloatArray

    @classmethod
    def from_array(cls, state: npt.ArrayLike) -> "ThreeDofState":
        """Parse ``[r_N,r_E,r_D,v_N,v_E,v_D]``."""
        values = as_vector(state, 6, name="three_dof_state")
        return cls(values[:3], values[3:6])

    def as_array(self) -> FloatArray:
        """Return a new state vector in normative order."""
        return np.concatenate((self.position_ned_m, self.velocity_ned_mps))


@dataclass(frozen=True, slots=True)
class SixDofState:
    """NED/FRD quaternion rigid-body state."""

    position_ned_m: FloatArray
    velocity_ned_mps: FloatArray
    quaternion_nb: FloatArray
    angular_rate_body_radps: FloatArray

    @classmethod
    def from_array(cls, state: npt.ArrayLike, *, normalize: bool = False) -> "SixDofState":
        """Parse the documented 13-element rigid-body state."""
        values = as_vector(state, 13, name="six_dof_state")
        quaternion = normalize_quaternion(values[6:10]) if normalize else values[6:10]
        return cls(values[:3], values[3:6], quaternion, values[10:13])

    def as_array(self) -> FloatArray:
        """Return a new state vector in normative order."""
        return np.concatenate(
            (
                self.position_ned_m,
                self.velocity_ned_mps,
                self.quaternion_nb,
                self.angular_rate_body_radps,
            )
        )


def project_six_dof_quaternion(state: FloatArray) -> FloatArray:
    """Return a copy with elements 6:10 normalised."""
    values = as_vector(state, 13, name="six_dof_state")
    values[6:10] = normalize_quaternion(values[6:10])
    return values
