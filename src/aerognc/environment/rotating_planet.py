"""Rotating oblate-planet gravity and apparent accelerations."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    ReferenceEllipsoid,
    dcm_ecef_to_ned,
    geodetic_to_ecef,
)
from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class RotatingOblatePlanet:
    """Synthetic axisymmetric planet model for flight-mechanics studies."""

    name: str
    ellipsoid: ReferenceEllipsoid
    gravitational_parameter_m3ps2: float
    rotation_rate_radps: float
    j2: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("planet name must be nonempty")
        values = np.array([self.gravitational_parameter_m3ps2, self.rotation_rate_radps, self.j2])
        if not np.all(np.isfinite(values)):
            raise ValueError("planet dynamics values must be finite")
        if self.gravitational_parameter_m3ps2 <= 0.0:
            raise ValueError("gravitational_parameter_m3ps2 must be positive")
        if self.rotation_rate_radps < 0.0:
            raise ValueError("rotation_rate_radps must be nonnegative")
        if abs(self.j2) >= 0.1:
            raise ValueError("j2 magnitude must be below 0.1")

    @property
    def rotation_vector_ecef_radps(self) -> FloatArray:
        """Return constant eastward rotation vector expressed ECEF."""
        return np.array([0.0, 0.0, self.rotation_rate_radps], dtype=np.float64)

    def gravity_ecef_mps2(self, position_ecef_m: npt.ArrayLike) -> FloatArray:
        """Return central plus first-order J2 gravitational acceleration."""
        position = as_vector(position_ecef_m, 3, name="position_ecef_m")
        radius_m = float(np.linalg.norm(position))
        if radius_m <= 0.0:
            raise ValueError("gravity is undefined at the planet centre")
        x_m, y_m, z_m = position
        z_ratio_squared = (z_m / radius_m) ** 2
        j2_scale = 1.5 * self.j2 * (self.ellipsoid.semi_major_axis_m / radius_m) ** 2
        equatorial_factor = 1.0 - j2_scale * (5.0 * z_ratio_squared - 1.0)
        polar_factor = 1.0 - j2_scale * (5.0 * z_ratio_squared - 3.0)
        common = -self.gravitational_parameter_m3ps2 / radius_m**3
        return common * np.array(
            [x_m * equatorial_factor, y_m * equatorial_factor, z_m * polar_factor],
            dtype=np.float64,
        )

    def coriolis_ecef_mps2(self, velocity_ecef_mps: npt.ArrayLike) -> FloatArray:
        """Return rotating-frame Coriolis acceleration ``-2 omega x v``."""
        velocity = as_vector(velocity_ecef_mps, 3, name="velocity_ecef_mps")
        return -2.0 * np.cross(self.rotation_vector_ecef_radps, velocity)

    def centrifugal_ecef_mps2(self, position_ecef_m: npt.ArrayLike) -> FloatArray:
        """Return rotating-frame centrifugal acceleration ``-omega x (omega x r)``."""
        position = as_vector(position_ecef_m, 3, name="position_ecef_m")
        rotation = self.rotation_vector_ecef_radps
        return -np.cross(rotation, np.cross(rotation, position))

    def apparent_acceleration_ecef_mps2(
        self,
        position_ecef_m: npt.ArrayLike,
        velocity_ecef_mps: npt.ArrayLike,
    ) -> FloatArray:
        """Return gravity plus Coriolis and centrifugal acceleration in ECEF."""
        return (
            self.gravity_ecef_mps2(position_ecef_m)
            + self.coriolis_ecef_mps2(velocity_ecef_mps)
            + self.centrifugal_ecef_mps2(position_ecef_m)
        )

    def surface_gravity_down_mps2(self, geodetic: GeodeticPosition) -> float:
        """Return positive apparent-gravity component along local geodetic down."""
        position_ecef_m = geodetic_to_ecef(geodetic, self.ellipsoid)
        acceleration_ecef_mps2 = self.apparent_acceleration_ecef_mps2(position_ecef_m, np.zeros(3))
        acceleration_ned_mps2 = (
            dcm_ecef_to_ned(geodetic.latitude_rad, geodetic.longitude_rad) @ acceleration_ecef_mps2
        )
        return float(acceleration_ned_mps2[2])
