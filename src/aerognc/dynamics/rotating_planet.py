"""Translational equations in a uniformly rotating body-fixed frame."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class RotatingTranslationalState:
    """ECEF position and velocity state in SI units."""

    position_ecef_m: FloatArray
    velocity_ecef_mps: FloatArray

    def __init__(
        self,
        position_ecef_m: npt.ArrayLike,
        velocity_ecef_mps: npt.ArrayLike,
    ) -> None:
        object.__setattr__(
            self,
            "position_ecef_m",
            as_vector(position_ecef_m, 3, name="position_ecef_m"),
        )
        object.__setattr__(
            self,
            "velocity_ecef_mps",
            as_vector(velocity_ecef_mps, 3, name="velocity_ecef_mps"),
        )

    def as_array(self) -> FloatArray:
        """Return state ordering ``[r_ecef, v_ecef]``."""
        return np.concatenate((self.position_ecef_m, self.velocity_ecef_mps))

    @classmethod
    def from_array(cls, state: npt.ArrayLike) -> "RotatingTranslationalState":
        """Build from the documented six-element ordering."""
        values = as_vector(state, 6, name="state")
        return cls(values[:3], values[3:])


def rotating_translational_derivative(
    state: npt.ArrayLike,
    planet: RotatingOblatePlanet,
    specific_force_ecef_mps2: npt.ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Evaluate ECEF translational dynamics.

    ``specific_force_ecef_mps2`` excludes gravity and all rotating-frame
    apparent terms.  This makes IMU, propulsion, and aerodynamic composition
    explicit at the call boundary.
    """
    rotating_state = RotatingTranslationalState.from_array(state)
    acceleration_ecef_mps2 = as_vector(
        specific_force_ecef_mps2, 3, name="specific_force_ecef_mps2"
    ) + planet.apparent_acceleration_ecef_mps2(
        rotating_state.position_ecef_m,
        rotating_state.velocity_ecef_mps,
    )
    return np.concatenate((rotating_state.velocity_ecef_mps, acceleration_ecef_mps2))
