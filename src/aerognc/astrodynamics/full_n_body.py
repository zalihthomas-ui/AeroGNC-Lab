"""Mutually interacting point-mass N-body dynamics for verification studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.integrators import rk4_step
from aerognc.mathematics.vectors import FloatArray

GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11


@dataclass(frozen=True, slots=True)
class MassiveBody:
    """Point-mass propagation body with a collision radius."""

    name: str
    mass_kg: float
    radius_m: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("N-body name cannot be empty")
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("N-body mass must be positive and finite")
        if not np.isfinite(self.radius_m) or self.radius_m < 0.0:
            raise ValueError("N-body radius must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class FullNBodyModel:
    """Pairwise Newtonian gravity with no fixed central-body assumption."""

    bodies: tuple[MassiveBody, ...]
    gravitational_constant_m3_kg_s2: float = GRAVITATIONAL_CONSTANT_M3_KG_S2

    def __post_init__(self) -> None:
        if len(self.bodies) < 2:
            raise ValueError("full N-body model requires at least two bodies")
        names = [body.name.casefold() for body in self.bodies]
        if len(names) != len(set(names)):
            raise ValueError("N-body names must be unique")
        if (
            not np.isfinite(self.gravitational_constant_m3_kg_s2)
            or self.gravitational_constant_m3_kg_s2 <= 0.0
        ):
            raise ValueError("gravitational constant must be positive and finite")

    @property
    def state_size(self) -> int:
        """Return the flattened Cartesian state size."""
        return 6 * len(self.bodies)

    def unpack(self, state: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return ``(positions, velocities)`` with shape ``(body_count, 3)``."""
        state_array = np.asarray(state, dtype=np.float64)
        if state_array.shape != (self.state_size,) or not np.all(np.isfinite(state_array)):
            raise ValueError(f"N-body state must contain {self.state_size} finite values")
        reshaped = state_array.reshape(len(self.bodies), 6)
        return reshaped[:, :3], reshaped[:, 3:6]

    def derivative(self, _time_s: float, state: FloatArray) -> FloatArray:
        """Return pairwise Newtonian state derivatives."""
        positions, velocities = self.unpack(state)
        accelerations = np.zeros_like(positions)
        for first_index in range(len(self.bodies)):
            for second_index in range(first_index + 1, len(self.bodies)):
                separation = positions[second_index] - positions[first_index]
                distance = float(np.linalg.norm(separation))
                collision_distance = (
                    self.bodies[first_index].radius_m + self.bodies[second_index].radius_m
                )
                if distance <= max(collision_distance, 0.0):
                    raise FloatingPointError("N-body collision or coincident positions detected")
                inverse_cube = 1.0 / distance**3
                accelerations[first_index] += (
                    self.gravitational_constant_m3_kg_s2
                    * self.bodies[second_index].mass_kg
                    * separation
                    * inverse_cube
                )
                accelerations[second_index] -= (
                    self.gravitational_constant_m3_kg_s2
                    * self.bodies[first_index].mass_kg
                    * separation
                    * inverse_cube
                )
        derivative = np.empty((len(self.bodies), 6), dtype=np.float64)
        derivative[:, :3] = velocities
        derivative[:, 3:6] = accelerations
        return derivative.ravel()

    def total_linear_momentum_kg_mps(self, state: npt.ArrayLike) -> FloatArray:
        """Return system linear momentum."""
        _positions, velocities = self.unpack(state)
        masses = np.array([body.mass_kg for body in self.bodies])
        return np.sum(masses[:, None] * velocities, axis=0)

    def barycentre_m(self, state: npt.ArrayLike) -> FloatArray:
        """Return the system centre of mass."""
        positions, _velocities = self.unpack(state)
        masses = np.array([body.mass_kg for body in self.bodies])
        return np.asarray(
            np.sum(masses[:, None] * positions, axis=0) / float(np.sum(masses)),
            dtype=np.float64,
        )

    def total_energy_j(self, state: npt.ArrayLike) -> float:
        """Return Newtonian kinetic plus pairwise potential energy."""
        positions, velocities = self.unpack(state)
        kinetic = sum(
            0.5 * body.mass_kg * float(np.dot(velocities[index], velocities[index]))
            for index, body in enumerate(self.bodies)
        )
        potential = 0.0
        for first_index in range(len(self.bodies)):
            for second_index in range(first_index + 1, len(self.bodies)):
                distance = float(np.linalg.norm(positions[second_index] - positions[first_index]))
                potential -= (
                    self.gravitational_constant_m3_kg_s2
                    * self.bodies[first_index].mass_kg
                    * self.bodies[second_index].mass_kg
                    / distance
                )
        return float(kinetic + potential)


@dataclass(frozen=True, slots=True)
class FullNBodyPropagation:
    """Fixed-step full N-body result."""

    time_s: FloatArray
    state: FloatArray


def propagate_full_n_body(
    model: FullNBodyModel,
    initial_state: npt.ArrayLike,
    duration_s: float,
    step_s: float,
) -> FullNBodyPropagation:
    """Propagate a full N-body state using the project's custom fixed-step RK4."""
    state = np.asarray(initial_state, dtype=np.float64)
    model.unpack(state)
    if not np.isfinite(duration_s) or not np.isfinite(step_s) or duration_s <= 0.0 or step_s <= 0.0:
        raise ValueError("N-body duration and step must be positive and finite")
    time_values = [0.0]
    state_values = [state.copy()]
    time_s = 0.0
    while time_s < duration_s:
        current_step_s = min(step_s, duration_s - time_s)
        state = rk4_step(model.derivative, time_s, state, current_step_s)
        time_s += current_step_s
        time_values.append(time_s)
        state_values.append(state.copy())
    return FullNBodyPropagation(np.asarray(time_values), np.vstack(state_values))
