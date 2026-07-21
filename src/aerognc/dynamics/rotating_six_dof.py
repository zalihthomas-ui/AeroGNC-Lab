"""Quaternion rigid-body dynamics in a planet-centred inertial frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.configuration.models import EnvironmentDefinition, VehicleDefinition
from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    LaunchSite,
    dcm_ecef_to_ned,
    dcm_inertial_to_ecef,
    ecef_position_to_ned,
    ecef_to_geodetic,
    inertial_to_ecef_state,
)
from aerognc.mathematics.quaternion import quaternion_multiply, quaternion_to_dcm
from aerognc.mathematics.vectors import FloatArray, as_matrix3, as_vector
from aerognc.vehicle.aerodynamics import AerodynamicLoads
from aerognc.vehicle.mass_properties import MassProperties


@dataclass(frozen=True, slots=True)
class RotatingRigidBodyState:
    """State ordered as inertial position/velocity, ``q_ib``, and ``omega_ib^b``."""

    position_inertial_m: FloatArray
    velocity_inertial_mps: FloatArray
    quaternion_ib: FloatArray
    angular_rate_inertial_body_radps: FloatArray

    @classmethod
    def from_array(cls, state: npt.ArrayLike) -> RotatingRigidBodyState:
        """Parse the documented 13-element inertial rigid-body state."""
        values = as_vector(state, 13, name="rotating_six_dof_state")
        return cls(values[:3], values[3:6], values[6:10], values[10:13])

    def as_array(self) -> FloatArray:
        """Return a new vector in the normative ordering."""
        return np.concatenate(
            (
                self.position_inertial_m,
                self.velocity_inertial_mps,
                self.quaternion_ib,
                self.angular_rate_inertial_body_radps,
            )
        )


@dataclass(frozen=True, slots=True)
class RotatingRigidBodyInputs:
    """Explicit body loads and inertial gravity for the rotating-planet EOM."""

    mass_kg: float
    inertia_body_kgm2: FloatArray
    inertia_rate_body_kgm2ps: FloatArray
    force_body_n: FloatArray
    moment_body_nm: FloatArray
    gravity_inertial_mps2: FloatArray

    def __init__(
        self,
        *,
        mass_kg: float,
        inertia_body_kgm2: npt.ArrayLike,
        force_body_n: npt.ArrayLike,
        moment_body_nm: npt.ArrayLike,
        gravity_inertial_mps2: npt.ArrayLike,
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
            "gravity_inertial_mps2",
            as_vector(gravity_inertial_mps2, 3, name="gravity_inertial_mps2"),
        )


def rotating_six_dof_derivative(
    state: npt.ArrayLike, inputs: RotatingRigidBodyInputs
) -> FloatArray:
    """Evaluate inertial translation and body-resolved Euler rotation equations."""
    rigid_state = RotatingRigidBodyState.from_array(state)
    dcm_ib = quaternion_to_dcm(rigid_state.quaternion_ib)
    acceleration_inertial_mps2 = (
        dcm_ib @ inputs.force_body_n / inputs.mass_kg + inputs.gravity_inertial_mps2
    )
    angular_momentum_body = inputs.inertia_body_kgm2 @ rigid_state.angular_rate_inertial_body_radps
    rotational_rhs = (
        inputs.moment_body_nm
        - np.cross(rigid_state.angular_rate_inertial_body_radps, angular_momentum_body)
        - inputs.inertia_rate_body_kgm2ps @ rigid_state.angular_rate_inertial_body_radps
    )
    angular_acceleration_body_radps2 = np.linalg.solve(inputs.inertia_body_kgm2, rotational_rhs)
    quaternion_rate = 0.5 * quaternion_multiply(
        rigid_state.quaternion_ib,
        np.concatenate(([0.0], rigid_state.angular_rate_inertial_body_radps)),
    )
    return np.concatenate(
        (
            rigid_state.velocity_inertial_mps,
            acceleration_inertial_mps2,
            quaternion_rate,
            angular_acceleration_body_radps2,
        )
    )


@dataclass(frozen=True, slots=True)
class RotatingRigidBodyLoads:
    """Loads and frame-resolved diagnostics at one inertial state."""

    geodetic: GeodeticPosition
    position_ecef_m: FloatArray
    velocity_ecef_mps: FloatArray
    position_ned_m: FloatArray
    velocity_ned_mps: FloatArray
    wind_ned_mps: FloatArray
    mass_properties: MassProperties
    aerodynamic: AerodynamicLoads
    thrust_body_n: FloatArray
    total_force_body_n: FloatArray
    total_moment_body_nm: FloatArray
    gravity_inertial_mps2: FloatArray


class RotatingRigidBodyFlightModel:
    """Compose atmosphere, wind, vehicle loads, and inertial rotating-planet flight."""

    def __init__(
        self,
        vehicle: VehicleDefinition,
        environment: EnvironmentDefinition,
        planet: RotatingOblatePlanet,
        launch_site: LaunchSite,
    ) -> None:
        self.vehicle = vehicle
        self.environment = environment
        self.planet = planet
        self.launch_site = launch_site

    def loads(
        self,
        time_s: float,
        state_array: npt.ArrayLike,
        actuator_moment_body_nm: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> RotatingRigidBodyLoads:
        """Evaluate forces in FRD body axes and gravity in inertial axes."""
        state = RotatingRigidBodyState.from_array(state_array)
        position_ecef_m, velocity_ecef_mps = inertial_to_ecef_state(
            state.position_inertial_m,
            state.velocity_inertial_mps,
            time_s=time_s,
            rotation_rate_radps=self.planet.rotation_rate_radps,
        )
        geodetic = ecef_to_geodetic(position_ecef_m, self.planet.ellipsoid)
        dcm_ne = dcm_ecef_to_ned(geodetic.latitude_rad, geodetic.longitude_rad)
        dcm_ei = dcm_inertial_to_ecef(self.planet.rotation_rate_radps * time_s)
        position_ned_m = ecef_position_to_ned(
            position_ecef_m, self.launch_site.geodetic, self.planet.ellipsoid
        )
        velocity_ned_mps = dcm_ne @ velocity_ecef_mps
        altitude_above_site_m = geodetic.altitude_m - self.launch_site.geodetic.altitude_m
        atmosphere = self.environment.atmosphere.properties(altitude_above_site_m)
        wind_ned_mps = self.environment.wind.velocity_ned_mps(time_s, altitude_above_site_m)
        air_velocity_ecef_mps = velocity_ecef_mps - dcm_ne.T @ wind_ned_mps
        air_velocity_inertial_mps = dcm_ei.T @ air_velocity_ecef_mps
        air_velocity_body_mps = quaternion_to_dcm(state.quaternion_ib).T @ air_velocity_inertial_mps
        aerodynamic = self.vehicle.aerodynamics.loads(
            air_velocity_body_mps,
            density_kgpm3=atmosphere.density_kgpm3,
            speed_of_sound_mps=atmosphere.speed_of_sound_mps,
            angular_rate_body_radps=state.angular_rate_inertial_body_radps,
        )
        thrust_body_n = np.array(
            [self.vehicle.propulsion.thrust_at_time_n(time_s), 0.0, 0.0], dtype=np.float64
        )
        actuator_moment = as_vector(actuator_moment_body_nm, 3, name="actuator_moment_body_nm")
        gravity_inertial_mps2 = dcm_ei.T @ self.planet.gravity_ecef_mps2(position_ecef_m)
        return RotatingRigidBodyLoads(
            geodetic=geodetic,
            position_ecef_m=position_ecef_m,
            velocity_ecef_mps=velocity_ecef_mps,
            position_ned_m=position_ned_m,
            velocity_ned_mps=velocity_ned_mps,
            wind_ned_mps=wind_ned_mps,
            mass_properties=self.vehicle.mass_properties.at_time(time_s),
            aerodynamic=aerodynamic,
            thrust_body_n=thrust_body_n,
            total_force_body_n=thrust_body_n + aerodynamic.force_body_n,
            total_moment_body_nm=aerodynamic.moment_body_nm + actuator_moment,
            gravity_inertial_mps2=gravity_inertial_mps2,
        )

    def derivative(
        self,
        time_s: float,
        state_array: FloatArray,
        actuator_moment_body_nm: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> FloatArray:
        """Evaluate the configured 13-state inertial derivative."""
        loads = self.loads(time_s, state_array, actuator_moment_body_nm)
        return rotating_six_dof_derivative(
            state_array,
            RotatingRigidBodyInputs(
                mass_kg=loads.mass_properties.mass_kg,
                inertia_body_kgm2=loads.mass_properties.inertia_body_kgm2,
                inertia_rate_body_kgm2ps=loads.mass_properties.inertia_rate_body_kgm2ps,
                force_body_n=loads.total_force_body_n,
                moment_body_nm=loads.total_moment_body_nm,
                gravity_inertial_mps2=loads.gravity_inertial_mps2,
            ),
        )
