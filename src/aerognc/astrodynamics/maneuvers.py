"""Impulsive and finite spacecraft maneuvers with propellant accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

STANDARD_GRAVITY_MPS2 = 9.80665
ManeuverFrame = Literal["inertial", "rtn"]


def _vector3(value: npt.ArrayLike, label: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite components")
    return vector


def rtn_to_inertial_matrix(position_m: npt.ArrayLike, velocity_mps: npt.ArrayLike) -> FloatArray:
    """Return the RTN-to-inertial direction-cosine matrix for one state."""
    position = _vector3(position_m, "position")
    velocity = _vector3(velocity_mps, "velocity")
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("position must have nonzero magnitude")
    radial = position / radius
    angular_momentum = np.cross(position, velocity)
    momentum_norm = float(np.linalg.norm(angular_momentum))
    if momentum_norm <= 0.0:
        raise ValueError("position and velocity cannot define an RTN frame")
    normal = angular_momentum / momentum_norm
    transverse = np.cross(normal, radial)
    return np.column_stack((radial, transverse, normal))


def ideal_propellant_used_kg(
    initial_mass_kg: float, delta_v_mps: float, specific_impulse_s: float
) -> float:
    """Return ideal propellant use from the Tsiolkovsky equation."""
    values = np.array([initial_mass_kg, delta_v_mps, specific_impulse_s])
    if not np.all(np.isfinite(values)) or initial_mass_kg <= 0.0 or specific_impulse_s <= 0.0:
        raise ValueError("mass and specific impulse must be positive and all inputs finite")
    if delta_v_mps < 0.0:
        raise ValueError("delta-v magnitude must be nonnegative")
    final_mass = initial_mass_kg * np.exp(
        -delta_v_mps / (specific_impulse_s * STANDARD_GRAVITY_MPS2)
    )
    return float(initial_mass_kg - final_mass)


def available_delta_v_mps(
    initial_mass_kg: float, dry_mass_kg: float, specific_impulse_s: float
) -> float:
    """Return ideal total delta-v available above dry mass."""
    values = np.array([initial_mass_kg, dry_mass_kg, specific_impulse_s])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("mass and specific impulse inputs must be positive and finite")
    if dry_mass_kg > initial_mass_kg:
        raise ValueError("dry mass cannot exceed initial mass")
    return float(specific_impulse_s * STANDARD_GRAVITY_MPS2 * np.log(initial_mass_kg / dry_mass_kg))


@dataclass(frozen=True, slots=True)
class ImpulsiveManeuver:
    """Instantaneous velocity change at a configured mission epoch."""

    name: str
    epoch_s: float
    delta_velocity_mps: tuple[float, float, float]
    frame: ManeuverFrame = "rtn"
    specific_impulse_s: float = 320.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("maneuver name cannot be empty")
        values = np.array([self.epoch_s, self.specific_impulse_s, *self.delta_velocity_mps])
        if not np.all(np.isfinite(values)) or self.epoch_s < 0.0 or self.specific_impulse_s <= 0.0:
            raise ValueError("impulsive maneuver values are nonphysical")
        if self.frame not in {"inertial", "rtn"}:
            raise ValueError("maneuver frame must be inertial or rtn")

    @property
    def magnitude_mps(self) -> float:
        """Return commanded delta-v magnitude."""
        return float(np.linalg.norm(self.delta_velocity_mps))


@dataclass(frozen=True, slots=True)
class FiniteBurn:
    """Constant-thrust finite burn over a specified interval."""

    name: str
    start_time_s: float
    duration_s: float
    thrust_n: float
    direction: tuple[float, float, float]
    frame: ManeuverFrame = "rtn"
    specific_impulse_s: float = 320.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("burn name cannot be empty")
        values = np.array(
            [
                self.start_time_s,
                self.duration_s,
                self.thrust_n,
                self.specific_impulse_s,
                *self.direction,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("finite-burn values must be finite")
        if self.start_time_s < 0.0 or self.duration_s <= 0.0 or self.thrust_n <= 0.0:
            raise ValueError("burn start must be nonnegative and duration/thrust positive")
        if self.specific_impulse_s <= 0.0:
            raise ValueError("specific impulse must be positive")
        if self.frame not in {"inertial", "rtn"}:
            raise ValueError("maneuver frame must be inertial or rtn")
        if np.linalg.norm(self.direction) <= 0.0:
            raise ValueError("finite-burn direction must have nonzero magnitude")

    @property
    def end_time_s(self) -> float:
        """Return the burn stop epoch."""
        return self.start_time_s + self.duration_s

    @property
    def mass_flow_rate_kg_s(self) -> float:
        """Return positive propellant mass-flow magnitude."""
        return self.thrust_n / (self.specific_impulse_s * STANDARD_GRAVITY_MPS2)


SpacecraftManeuver = ImpulsiveManeuver | FiniteBurn


def vector_in_inertial_frame(
    vector: npt.ArrayLike,
    frame: ManeuverFrame,
    position_m: npt.ArrayLike,
    velocity_mps: npt.ArrayLike,
) -> FloatArray:
    """Resolve an inertial or local RTN vector into inertial coordinates."""
    components = _vector3(vector, "maneuver vector")
    if frame == "inertial":
        return components
    if frame == "rtn":
        return rtn_to_inertial_matrix(position_m, velocity_mps) @ components
    raise ValueError("maneuver frame must be inertial or rtn")


def apply_impulsive_maneuver(
    state: npt.ArrayLike,
    maneuver: ImpulsiveManeuver,
    dry_mass_kg: float,
) -> FloatArray:
    """Apply one impulse to a ``[r, v, mass]`` state and enforce dry mass."""
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.shape != (7,) or not np.all(np.isfinite(state_array)):
        raise ValueError("maneuver state must contain seven finite values")
    if not np.isfinite(dry_mass_kg) or dry_mass_kg <= 0.0:
        raise ValueError("dry mass must be positive and finite")
    delta_velocity = vector_in_inertial_frame(
        maneuver.delta_velocity_mps,
        maneuver.frame,
        state_array[:3],
        state_array[3:6],
    )
    propellant_used = ideal_propellant_used_kg(
        float(state_array[6]), float(np.linalg.norm(delta_velocity)), maneuver.specific_impulse_s
    )
    final_mass = float(state_array[6] - propellant_used)
    if final_mass < dry_mass_kg - 1.0e-12:
        raise FloatingPointError(
            f"maneuver {maneuver.name!r} requires more propellant than available"
        )
    output = state_array.copy()
    output[3:6] += delta_velocity
    output[6] = max(final_mass, dry_mass_kg)
    return output


def finite_burn_derivative(
    time_s: float,
    state: npt.ArrayLike,
    burn: FiniteBurn,
    dry_mass_kg: float,
) -> tuple[FloatArray, float]:
    """Return inertial acceleration and mass derivative for one active burn."""
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.shape != (7,) or not np.all(np.isfinite(state_array)):
        raise ValueError("finite-burn state must contain seven finite values")
    if not burn.start_time_s <= time_s < burn.end_time_s or state_array[6] <= dry_mass_kg:
        return np.zeros(3), 0.0
    direction = vector_in_inertial_frame(
        burn.direction, burn.frame, state_array[:3], state_array[3:6]
    )
    direction /= np.linalg.norm(direction)
    return burn.thrust_n / state_array[6] * direction, -burn.mass_flow_rate_kg_s
