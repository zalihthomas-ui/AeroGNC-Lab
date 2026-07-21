"""Nonlinear quaternion rigid-body dynamics in NED navigation and FRD body axes."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.configuration.models import EnvironmentDefinition, VehicleDefinition
from aerognc.dynamics.state import SixDofState
from aerognc.mathematics.coordinates import navigation_to_body
from aerognc.mathematics.quaternion import quaternion_multiply, quaternion_to_dcm
from aerognc.mathematics.vectors import FloatArray, as_matrix3, as_vector
from aerognc.vehicle.aerodynamics import AerodynamicLoads
from aerognc.vehicle.mass_properties import MassProperties


@dataclass(frozen=True, slots=True)
class RigidBodyInputs:
    """Explicit force/moment, mass properties, and gravity for the 6-DOF EOM."""

    mass_kg: float
    inertia_body_kgm2: FloatArray
    inertia_rate_body_kgm2ps: FloatArray
    force_body_n: FloatArray
    moment_body_nm: FloatArray
    gravity_ned_mps2: FloatArray

    def __init__(
        self,
        *,
        mass_kg: float,
        inertia_body_kgm2: npt.ArrayLike,
        force_body_n: npt.ArrayLike,
        moment_body_nm: npt.ArrayLike,
        gravity_ned_mps2: npt.ArrayLike,
        inertia_rate_body_kgm2ps: npt.ArrayLike | None = None,
    ) -> None:
        if not np.isfinite(mass_kg) or mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive and finite")
        inertia = as_matrix3(inertia_body_kgm2, name="inertia_body_kgm2")
        if not np.allclose(inertia, inertia.T, atol=1.0e-12):
            raise ValueError("inertia_body_kgm2 must be symmetric")
        if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
            raise ValueError("inertia_body_kgm2 must be positive definite")
        inertia_rate = (
            np.zeros((3, 3), dtype=np.float64)
            if inertia_rate_body_kgm2ps is None
            else as_matrix3(inertia_rate_body_kgm2ps, name="inertia_rate_body_kgm2ps")
        )
        if not np.allclose(inertia_rate, inertia_rate.T, atol=1.0e-12):
            raise ValueError("inertia_rate_body_kgm2ps must be symmetric")
        object.__setattr__(self, "mass_kg", float(mass_kg))
        object.__setattr__(self, "inertia_body_kgm2", inertia)
        object.__setattr__(self, "inertia_rate_body_kgm2ps", inertia_rate)
        object.__setattr__(self, "force_body_n", as_vector(force_body_n, 3, name="force_body_n"))
        object.__setattr__(
            self, "moment_body_nm", as_vector(moment_body_nm, 3, name="moment_body_nm")
        )
        object.__setattr__(
            self,
            "gravity_ned_mps2",
            as_vector(gravity_ned_mps2, 3, name="gravity_ned_mps2"),
        )


def six_dof_derivative(state: npt.ArrayLike, inputs: RigidBodyInputs) -> FloatArray:
    """Evaluate the documented nonlinear 13-state rigid-body equations.

    Applied body force excludes gravity. The rotational equation includes the
    prescribed inertia-rate term ``-I_dot @ omega``.
    """
    rigid_state = SixDofState.from_array(state)
    dcm_nb = quaternion_to_dcm(rigid_state.quaternion_nb)
    acceleration_ned_mps2 = dcm_nb @ inputs.force_body_n / inputs.mass_kg + inputs.gravity_ned_mps2
    angular_momentum_body = inputs.inertia_body_kgm2 @ rigid_state.angular_rate_body_radps
    rotational_rhs = (
        inputs.moment_body_nm
        - np.cross(rigid_state.angular_rate_body_radps, angular_momentum_body)
        - inputs.inertia_rate_body_kgm2ps @ rigid_state.angular_rate_body_radps
    )
    angular_acceleration_body_radps2 = np.linalg.solve(inputs.inertia_body_kgm2, rotational_rhs)
    quaternion_rate = 0.5 * quaternion_multiply(
        rigid_state.quaternion_nb,
        np.concatenate(([0.0], rigid_state.angular_rate_body_radps)),
    )
    return np.concatenate(
        (
            rigid_state.velocity_ned_mps,
            acceleration_ned_mps2,
            quaternion_rate,
            angular_acceleration_body_radps2,
        )
    )


@dataclass(frozen=True, slots=True)
class RigidBodyLoads:
    """Configured full-flight loads and supporting diagnostics."""

    mass_properties: MassProperties
    aerodynamic: AerodynamicLoads
    thrust_body_n: FloatArray
    total_force_body_n: FloatArray
    total_moment_body_nm: FloatArray
    gravity_ned_mps2: FloatArray
    wind_ned_mps: FloatArray


class RigidBodyFlightModel:
    """Compose environment and synthetic vehicle models around the 6-DOF EOM."""

    def __init__(self, vehicle: VehicleDefinition, environment: EnvironmentDefinition) -> None:
        self.vehicle = vehicle
        self.environment = environment

    def loads(
        self,
        time_s: float,
        state_array: npt.ArrayLike,
        actuator_moment_body_nm: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> RigidBodyLoads:
        """Evaluate thrust, aerodynamic, actuator, and gravity loads."""
        state = SixDofState.from_array(state_array)
        altitude_m = -float(state.position_ned_m[2])
        atmosphere = self.environment.atmosphere.properties(altitude_m)
        wind_ned_mps = self.environment.wind.velocity_ned_mps(time_s, altitude_m)
        air_velocity_ned_mps = state.velocity_ned_mps - wind_ned_mps
        air_velocity_body_mps = navigation_to_body(air_velocity_ned_mps, state.quaternion_nb)
        aerodynamic = self.vehicle.aerodynamics.loads(
            air_velocity_body_mps,
            density_kgpm3=atmosphere.density_kgpm3,
            speed_of_sound_mps=atmosphere.speed_of_sound_mps,
            angular_rate_body_radps=state.angular_rate_body_radps,
        )
        thrust_body_n = np.array(
            [self.vehicle.propulsion.thrust_at_time_n(time_s), 0.0, 0.0],
            dtype=np.float64,
        )
        actuator_moment = as_vector(actuator_moment_body_nm, 3, name="actuator_moment_body_nm")
        return RigidBodyLoads(
            mass_properties=self.vehicle.mass_properties.at_time(time_s),
            aerodynamic=aerodynamic,
            thrust_body_n=thrust_body_n,
            total_force_body_n=thrust_body_n + aerodynamic.force_body_n,
            total_moment_body_nm=aerodynamic.moment_body_nm + actuator_moment,
            gravity_ned_mps2=self.environment.gravity.acceleration_ned_mps2(altitude_m),
            wind_ned_mps=wind_ned_mps,
        )

    def derivative(
        self,
        time_s: float,
        state_array: FloatArray,
        actuator_moment_body_nm: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> FloatArray:
        """Evaluate configured rigid-body derivative."""
        loads = self.loads(time_s, state_array, actuator_moment_body_nm)
        inputs = RigidBodyInputs(
            mass_kg=loads.mass_properties.mass_kg,
            inertia_body_kgm2=loads.mass_properties.inertia_body_kgm2,
            inertia_rate_body_kgm2ps=loads.mass_properties.inertia_rate_body_kgm2ps,
            force_body_n=loads.total_force_body_n,
            moment_body_nm=loads.total_moment_body_nm,
            gravity_ned_mps2=loads.gravity_ned_mps2,
        )
        return six_dof_derivative(state_array, inputs)
