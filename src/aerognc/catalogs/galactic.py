"""ICRS and Galactic coordinate transforms implemented without Astropy.

The fixed orthonormal matrix is the standard ICRS-to-Galactic rotation used by
the Hipparcos explanatory supplement and Astropy. Cartesian axes are
heliocentric: +x points to Galactic longitude 0 deg, +y to 90 deg, and +z to
the north Galactic pole. Distances are parsecs because catalog distances are
observational rather than simulation state variables.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ICRS_TO_GALACTIC = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=np.float64,
)


def _validate_longitude_latitude(longitude_deg: float, latitude_deg: float) -> None:
    values = np.array([longitude_deg, latitude_deg], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinate angles must be finite")
    if not 0.0 <= longitude_deg < 360.0:
        raise ValueError("longitude/right ascension must be in [0, 360) deg")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude/declination must be in [-90, 90] deg")


def _unit_vector(longitude_deg: float, latitude_deg: float) -> FloatArray:
    longitude_rad = np.deg2rad(longitude_deg)
    latitude_rad = np.deg2rad(latitude_deg)
    cosine_latitude = np.cos(latitude_rad)
    return np.array(
        [
            cosine_latitude * np.cos(longitude_rad),
            cosine_latitude * np.sin(longitude_rad),
            np.sin(latitude_rad),
        ],
        dtype=np.float64,
    )


def _angles_from_unit_vector(vector: FloatArray) -> tuple[float, float]:
    longitude_deg = float(np.rad2deg(np.arctan2(vector[1], vector[0])) % 360.0)
    latitude_deg = float(np.rad2deg(np.arcsin(np.clip(vector[2], -1.0, 1.0))))
    return longitude_deg, latitude_deg


def icrs_to_galactic_deg(right_ascension_deg: float, declination_deg: float) -> tuple[float, float]:
    """Return Galactic longitude and latitude [deg] from an ICRS direction."""
    _validate_longitude_latitude(right_ascension_deg, declination_deg)
    galactic_vector = ICRS_TO_GALACTIC @ _unit_vector(right_ascension_deg, declination_deg)
    return _angles_from_unit_vector(galactic_vector)


def galactic_to_icrs_deg(longitude_deg: float, latitude_deg: float) -> tuple[float, float]:
    """Return ICRS right ascension and declination [deg] from Galactic angles."""
    _validate_longitude_latitude(longitude_deg, latitude_deg)
    icrs_vector = ICRS_TO_GALACTIC.T @ _unit_vector(longitude_deg, latitude_deg)
    return _angles_from_unit_vector(icrs_vector)


def heliocentric_galactic_xyz_pc(
    right_ascension_deg: float,
    declination_deg: float,
    distance_pc: float,
) -> FloatArray:
    """Return heliocentric Galactic Cartesian position [pc]."""
    if not np.isfinite(distance_pc) or distance_pc <= 0.0:
        raise ValueError("catalog distance must be finite and positive")
    longitude_deg, latitude_deg = icrs_to_galactic_deg(
        right_ascension_deg,
        declination_deg,
    )
    return distance_pc * _unit_vector(longitude_deg, latitude_deg)
