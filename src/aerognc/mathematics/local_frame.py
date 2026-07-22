"""Home-referenced local-tangent geometry for waypoint navigation.

This module builds on :mod:`aerognc.mathematics.geodesy` (geodetic/ECEF/NED
conversions on a reference ellipsoid) and adds the pieces a map-based waypoint
planner needs:

* a shared WGS84 ellipsoid constant,
* a :class:`LocalTangentFrame` that converts between geodetic coordinates and a
  local NED frame anchored at a fixed *home* origin,
* great-circle distance and initial bearing on a sphere,
* a flat-Earth NED offset valid for short ranges, and
* angle-wrapping helpers.

All angles are in radians and all distances in metres, matching the rest of the
package. Latitude/longitude *degrees* only appear at YAML/UI boundaries and are
converted before reaching this module.
"""

from dataclasses import dataclass

import numpy as np

from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    ReferenceEllipsoid,
    ecef_position_to_ned,
    ecef_to_geodetic,
    geodetic_to_ecef,
    meridian_radius_m,
    ned_position_to_ecef,
    prime_vertical_radius_m,
)
from aerognc.mathematics.vectors import FloatArray

# Standard WGS84 ellipsoid (metres / dimensionless flattening). The atmospheric
# plant uses a fictional Orbis-A planet, but map-based mission planning is stated
# in real geographic coordinates, so a documented Earth ellipsoid is the correct
# reference here.
WGS84 = ReferenceEllipsoid(semi_major_axis_m=6378137.0, flattening=1.0 / 298.257223563)

# Mean spherical Earth radius used only for great-circle range/bearing sanity
# geometry (never for the exact NED conversion, which uses the ellipsoid).
MEAN_EARTH_RADIUS_M = 6371008.7714


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to the half-open interval ``[-pi, pi)`` [rad]."""
    if not np.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_to_2pi(angle_rad: float) -> float:
    """Wrap an angle to the half-open interval ``[0, 2*pi)`` [rad]."""
    if not np.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    return float(angle_rad % (2.0 * np.pi))


def great_circle_distance_m(
    origin: GeodeticPosition,
    target: GeodeticPosition,
    *,
    radius_m: float = MEAN_EARTH_RADIUS_M,
) -> float:
    """Return the great-circle surface distance between two points [m].

    Uses the numerically well-conditioned haversine formula on a sphere of the
    given radius. This is intended for range readouts and short-to-medium mission
    legs, not geodesic-grade distance on the ellipsoid.
    """
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    d_lat = target.latitude_rad - origin.latitude_rad
    d_lon = target.longitude_rad - origin.longitude_rad
    sin_lat = np.sin(0.5 * d_lat)
    sin_lon = np.sin(0.5 * d_lon)
    haversine = (
        sin_lat * sin_lat
        + np.cos(origin.latitude_rad) * np.cos(target.latitude_rad) * sin_lon * sin_lon
    )
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return float(radius_m * central_angle)


def initial_bearing_rad(origin: GeodeticPosition, target: GeodeticPosition) -> float:
    """Return the initial great-circle bearing from origin to target [rad].

    Measured clockwise from true north and wrapped to ``[0, 2*pi)``. Undefined
    when the two points coincide; the caller should guard against zero range.
    """
    d_lon = target.longitude_rad - origin.longitude_rad
    y = np.sin(d_lon) * np.cos(target.latitude_rad)
    x = np.cos(origin.latitude_rad) * np.sin(target.latitude_rad) - np.sin(
        origin.latitude_rad
    ) * np.cos(target.latitude_rad) * np.cos(d_lon)
    return wrap_to_2pi(float(np.arctan2(y, x)))


def flat_earth_offset_ned_m(
    origin: GeodeticPosition,
    target: GeodeticPosition,
    ellipsoid: ReferenceEllipsoid = WGS84,
) -> FloatArray:
    """Return the target's local NED offset from origin using a flat-Earth model.

    North and East use the meridian and prime-vertical radii of curvature at the
    origin latitude; Down is the negative altitude difference. Accurate for short
    ranges (a few km); use :class:`LocalTangentFrame` for the exact conversion.
    """
    meridian_m = meridian_radius_m(origin.latitude_rad, ellipsoid) + origin.altitude_m
    transverse_m = prime_vertical_radius_m(origin.latitude_rad, ellipsoid) + origin.altitude_m
    north_m = (target.latitude_rad - origin.latitude_rad) * meridian_m
    east_m = (
        (target.longitude_rad - origin.longitude_rad) * transverse_m * np.cos(origin.latitude_rad)
    )
    down_m = -(target.altitude_m - origin.altitude_m)
    return np.array([north_m, east_m, down_m], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LocalTangentFrame:
    """Local NED frame anchored at a fixed geodetic *home* origin.

    Provides exact, invertible conversion between geodetic coordinates and the
    home-referenced local NED frame by round-tripping through ECEF on the given
    ellipsoid. The origin is immutable, so every conversion shares one reference.
    """

    origin: GeodeticPosition
    ellipsoid: ReferenceEllipsoid = WGS84

    def geodetic_to_ned(self, geodetic: GeodeticPosition) -> FloatArray:
        """Return the geodetic point expressed in home-referenced local NED [m]."""
        position_ecef_m = geodetic_to_ecef(geodetic, self.ellipsoid)
        return ecef_position_to_ned(position_ecef_m, self.origin, self.ellipsoid)

    def ned_to_geodetic(self, position_ned_m: FloatArray) -> GeodeticPosition:
        """Return the geodetic coordinates of a home-referenced local NED point."""
        position_ecef_m = ned_position_to_ecef(position_ned_m, self.origin, self.ellipsoid)
        return ecef_to_geodetic(position_ecef_m, self.ellipsoid)
