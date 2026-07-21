"""Oblate-body geodesy and rotating-frame transformations in SI units."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class ReferenceEllipsoid:
    """Axisymmetric reference ellipsoid.

    ``flattening`` is dimensionless and follows ``f = (a - b) / a``.  The
    class contains geometry only; gravity and rotation belong to the planet
    model so that frame conversion never silently applies dynamics.
    """

    semi_major_axis_m: float
    flattening: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.semi_major_axis_m) or self.semi_major_axis_m <= 0.0:
            raise ValueError("semi_major_axis_m must be positive and finite")
        if not np.isfinite(self.flattening) or not 0.0 <= self.flattening < 1.0:
            raise ValueError("flattening must be finite and in [0, 1)")

    @property
    def semi_minor_axis_m(self) -> float:
        """Return polar semi-axis ``b`` [m]."""
        return self.semi_major_axis_m * (1.0 - self.flattening)

    @property
    def first_eccentricity_squared(self) -> float:
        """Return first eccentricity squared ``e²`` [-]."""
        return self.flattening * (2.0 - self.flattening)


@dataclass(frozen=True, slots=True)
class GeodeticPosition:
    """Geodetic latitude, east longitude, and ellipsoidal altitude."""

    latitude_rad: float
    longitude_rad: float
    altitude_m: float

    def __post_init__(self) -> None:
        values = np.array([self.latitude_rad, self.longitude_rad, self.altitude_m])
        if not np.all(np.isfinite(values)):
            raise ValueError("geodetic coordinates must be finite")
        if not -0.5 * np.pi <= self.latitude_rad <= 0.5 * np.pi:
            raise ValueError("latitude_rad must be in [-pi/2, pi/2]")


@dataclass(frozen=True, slots=True)
class LaunchSite:
    """Fixed launch site on a reference ellipsoid."""

    geodetic: GeodeticPosition
    azimuth_rad: float
    elevation_rad: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.azimuth_rad, self.elevation_rad])):
            raise ValueError("launch-site pointing angles must be finite")
        if not -0.5 * np.pi <= self.elevation_rad <= 0.5 * np.pi:
            raise ValueError("elevation_rad must be in [-pi/2, pi/2]")


def prime_vertical_radius_m(latitude_rad: float, ellipsoid: ReferenceEllipsoid) -> float:
    """Return the prime-vertical radius of curvature ``N`` [m]."""
    if not np.isfinite(latitude_rad):
        raise ValueError("latitude_rad must be finite")
    sin_latitude = np.sin(latitude_rad)
    denominator = np.sqrt(1.0 - ellipsoid.first_eccentricity_squared * sin_latitude**2)
    return float(ellipsoid.semi_major_axis_m / denominator)


def meridian_radius_m(latitude_rad: float, ellipsoid: ReferenceEllipsoid) -> float:
    """Return the meridian radius of curvature ``M`` [m]."""
    if not np.isfinite(latitude_rad):
        raise ValueError("latitude_rad must be finite")
    eccentricity_squared = ellipsoid.first_eccentricity_squared
    sin_latitude = np.sin(latitude_rad)
    denominator = (1.0 - eccentricity_squared * sin_latitude**2) ** 1.5
    return float(ellipsoid.semi_major_axis_m * (1.0 - eccentricity_squared) / denominator)


def geodetic_to_ecef(
    geodetic: GeodeticPosition,
    ellipsoid: ReferenceEllipsoid,
) -> FloatArray:
    """Convert geodetic coordinates to body-fixed Cartesian position [m]."""
    latitude = geodetic.latitude_rad
    longitude = geodetic.longitude_rad
    altitude = geodetic.altitude_m
    radius = prime_vertical_radius_m(latitude, ellipsoid)
    cos_latitude = np.cos(latitude)
    sin_latitude = np.sin(latitude)
    cos_longitude = np.cos(longitude)
    sin_longitude = np.sin(longitude)
    return np.array(
        [
            (radius + altitude) * cos_latitude * cos_longitude,
            (radius + altitude) * cos_latitude * sin_longitude,
            (radius * (1.0 - ellipsoid.first_eccentricity_squared) + altitude) * sin_latitude,
        ],
        dtype=np.float64,
    )


def ecef_to_geodetic(
    position_ecef_m: npt.ArrayLike,
    ellipsoid: ReferenceEllipsoid,
    *,
    tolerance_rad: float = 1.0e-13,
    maximum_iterations: int = 15,
) -> GeodeticPosition:
    """Convert body-fixed Cartesian position to geodetic coordinates.

    A bounded fixed-point iteration is used away from the poles.  Failure to
    converge is explicit; the function never returns a partially converged
    coordinate.
    """
    if tolerance_rad <= 0.0 or not np.isfinite(tolerance_rad):
        raise ValueError("tolerance_rad must be positive and finite")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    x_m, y_m, z_m = as_vector(position_ecef_m, 3, name="position_ecef_m")
    horizontal_m = float(np.hypot(x_m, y_m))
    if horizontal_m <= np.finfo(np.float64).eps * ellipsoid.semi_major_axis_m:
        if abs(z_m) <= np.finfo(np.float64).eps * ellipsoid.semi_major_axis_m:
            raise ValueError("geodetic coordinates are undefined at the body centre")
        return GeodeticPosition(
            latitude_rad=float(np.copysign(0.5 * np.pi, z_m)),
            longitude_rad=0.0,
            altitude_m=float(abs(z_m) - ellipsoid.semi_minor_axis_m),
        )

    longitude = float(np.arctan2(y_m, x_m))
    eccentricity_squared = ellipsoid.first_eccentricity_squared
    latitude = float(np.arctan2(z_m, horizontal_m * (1.0 - eccentricity_squared)))
    altitude = 0.0
    for _ in range(maximum_iterations):
        radius = prime_vertical_radius_m(latitude, ellipsoid)
        cos_latitude = np.cos(latitude)
        sin_latitude = np.sin(latitude)
        surface_projection_m = ellipsoid.semi_major_axis_m * np.sqrt(
            1.0 - eccentricity_squared * sin_latitude**2
        )
        altitude = horizontal_m * cos_latitude + z_m * sin_latitude - surface_projection_m
        correction = 1.0 - eccentricity_squared * radius / (radius + altitude)
        updated_latitude = float(np.arctan2(z_m, horizontal_m * correction))
        if abs(updated_latitude - latitude) <= tolerance_rad:
            latitude = updated_latitude
            sin_latitude = np.sin(latitude)
            cos_latitude = np.cos(latitude)
            surface_projection_m = ellipsoid.semi_major_axis_m * np.sqrt(
                1.0 - eccentricity_squared * sin_latitude**2
            )
            altitude = horizontal_m * cos_latitude + z_m * sin_latitude - surface_projection_m
            return GeodeticPosition(latitude, longitude, float(altitude))
        latitude = updated_latitude
    raise RuntimeError("ECEF-to-geodetic conversion did not converge")


def dcm_ecef_to_ned(latitude_rad: float, longitude_rad: float) -> FloatArray:
    """Return the ECEF-to-local-NED direction cosine matrix."""
    if not np.all(np.isfinite([latitude_rad, longitude_rad])):
        raise ValueError("latitude_rad and longitude_rad must be finite")
    sin_latitude = np.sin(latitude_rad)
    cos_latitude = np.cos(latitude_rad)
    sin_longitude = np.sin(longitude_rad)
    cos_longitude = np.cos(longitude_rad)
    return np.array(
        [
            [
                -sin_latitude * cos_longitude,
                -sin_latitude * sin_longitude,
                cos_latitude,
            ],
            [-sin_longitude, cos_longitude, 0.0],
            [
                -cos_latitude * cos_longitude,
                -cos_latitude * sin_longitude,
                -sin_latitude,
            ],
        ],
        dtype=np.float64,
    )


def ecef_position_to_ned(
    position_ecef_m: npt.ArrayLike,
    origin: GeodeticPosition,
    ellipsoid: ReferenceEllipsoid,
) -> FloatArray:
    """Return target position relative to a geodetic origin in local NED [m]."""
    origin_ecef_m = geodetic_to_ecef(origin, ellipsoid)
    delta_ecef_m = as_vector(position_ecef_m, 3, name="position_ecef_m") - origin_ecef_m
    return dcm_ecef_to_ned(origin.latitude_rad, origin.longitude_rad) @ delta_ecef_m


def ned_position_to_ecef(
    position_ned_m: npt.ArrayLike,
    origin: GeodeticPosition,
    ellipsoid: ReferenceEllipsoid,
) -> FloatArray:
    """Convert local-NED displacement from an origin to ECEF position [m]."""
    origin_ecef_m = geodetic_to_ecef(origin, ellipsoid)
    dcm_ne = dcm_ecef_to_ned(origin.latitude_rad, origin.longitude_rad)
    return origin_ecef_m + dcm_ne.T @ as_vector(position_ned_m, 3, name="position_ned_m")


def dcm_inertial_to_ecef(rotation_angle_rad: float) -> FloatArray:
    """Return inertial-to-body-fixed DCM for positive eastward body rotation."""
    if not np.isfinite(rotation_angle_rad):
        raise ValueError("rotation_angle_rad must be finite")
    cosine = np.cos(rotation_angle_rad)
    sine = np.sin(rotation_angle_rad)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def inertial_to_ecef_state(
    position_inertial_m: npt.ArrayLike,
    velocity_inertial_mps: npt.ArrayLike,
    *,
    time_s: float,
    rotation_rate_radps: float,
    initial_rotation_angle_rad: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Convert inertial position/velocity to rotating ECEF components."""
    values = np.array([time_s, rotation_rate_radps, initial_rotation_angle_rad])
    if not np.all(np.isfinite(values)):
        raise ValueError("time and rotation values must be finite")
    position_i = as_vector(position_inertial_m, 3, name="position_inertial_m")
    velocity_i = as_vector(velocity_inertial_mps, 3, name="velocity_inertial_mps")
    angle = initial_rotation_angle_rad + rotation_rate_radps * time_s
    dcm_ei = dcm_inertial_to_ecef(angle)
    rotation_i_radps = np.array([0.0, 0.0, rotation_rate_radps])
    return (
        dcm_ei @ position_i,
        dcm_ei @ (velocity_i - np.cross(rotation_i_radps, position_i)),
    )


def ecef_to_inertial_state(
    position_ecef_m: npt.ArrayLike,
    velocity_ecef_mps: npt.ArrayLike,
    *,
    time_s: float,
    rotation_rate_radps: float,
    initial_rotation_angle_rad: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Convert rotating ECEF position/velocity to inertial components."""
    values = np.array([time_s, rotation_rate_radps, initial_rotation_angle_rad])
    if not np.all(np.isfinite(values)):
        raise ValueError("time and rotation values must be finite")
    position_e = as_vector(position_ecef_m, 3, name="position_ecef_m")
    velocity_e = as_vector(velocity_ecef_mps, 3, name="velocity_ecef_mps")
    angle = initial_rotation_angle_rad + rotation_rate_radps * time_s
    dcm_ie = dcm_inertial_to_ecef(angle).T
    rotation_e_radps = np.array([0.0, 0.0, rotation_rate_radps])
    return (
        dcm_ie @ position_e,
        dcm_ie @ (velocity_e + np.cross(rotation_e_radps, position_e)),
    )


def body_rotation_rate_ned(
    latitude_rad: float,
    rotation_rate_radps: float,
) -> FloatArray:
    """Return planet-fixed frame rotation relative inertial, expressed NED."""
    if not np.all(np.isfinite([latitude_rad, rotation_rate_radps])):
        raise ValueError("latitude_rad and rotation_rate_radps must be finite")
    return np.array(
        [
            rotation_rate_radps * np.cos(latitude_rad),
            0.0,
            -rotation_rate_radps * np.sin(latitude_rad),
        ],
        dtype=np.float64,
    )


def transport_rate_ned(
    geodetic: GeodeticPosition,
    velocity_ned_mps: npt.ArrayLike,
    ellipsoid: ReferenceEllipsoid,
) -> FloatArray:
    """Return local NED frame transport rate relative body-fixed, expressed NED."""
    north_mps, east_mps, _ = as_vector(velocity_ned_mps, 3, name="velocity_ned_mps")
    meridian_m = meridian_radius_m(geodetic.latitude_rad, ellipsoid)
    prime_vertical_m = prime_vertical_radius_m(geodetic.latitude_rad, ellipsoid)
    meridian_distance_m = meridian_m + geodetic.altitude_m
    transverse_distance_m = prime_vertical_m + geodetic.altitude_m
    if meridian_distance_m <= 0.0 or transverse_distance_m <= 0.0:
        raise ValueError("geodetic altitude is outside the transport-rate domain")
    return np.array(
        [
            east_mps / transverse_distance_m,
            -north_mps / meridian_distance_m,
            -east_mps * np.tan(geodetic.latitude_rad) / transverse_distance_m,
        ],
        dtype=np.float64,
    )
