"""Declared inertial-frame metadata and ecliptic/equatorial J2000 rotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

FrameName = Literal["HELIOS_ECLIPJ2000", "J2000"]
MEAN_OBLIQUITY_J2000_RAD = np.deg2rad(23.439291111)


@dataclass(frozen=True, slots=True)
class ReferenceFrameDefinition:
    """Human- and machine-readable frame declaration for exported trajectories."""

    name: FrameName
    center_name: str
    x_axis: str
    z_axis: str
    handedness: str
    time_scale: str
    fidelity: str


HELIOS_ECLIPJ2000 = ReferenceFrameDefinition(
    name="HELIOS_ECLIPJ2000",
    center_name="Helios (fictional)",
    x_axis="J2000 mean-equinox direction",
    z_axis="J2000 mean ecliptic north",
    handedness="right-handed",
    time_scale="synthetic elapsed TDB-like seconds",
    fidelity="fixed mean-ecliptic orientation; not an operational ICRF realization",
)

J2000_EQUATORIAL = ReferenceFrameDefinition(
    name="J2000",
    center_name="caller-defined",
    x_axis="J2000 mean-equinox direction",
    z_axis="J2000 mean-equator north",
    handedness="right-handed",
    time_scale="TDB",
    fidelity="fixed mean-equator/equinox J2000 rotation used for interoperability tests",
)


def ecliptic_to_equatorial_j2000_matrix() -> FloatArray:
    """Return the passive component transform from mean ecliptic to J2000 equator."""
    cosine = float(np.cos(MEAN_OBLIQUITY_J2000_RAD))
    sine = float(np.sin(MEAN_OBLIQUITY_J2000_RAD))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def transform_inertial_state(
    state: npt.ArrayLike,
    source_frame: FrameName,
    destination_frame: FrameName,
) -> FloatArray:
    """Rotate one position/velocity state between the two fixed J2000 frames."""
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("inertial state must contain six finite SI components")
    if source_frame == destination_frame:
        return values.copy()
    rotation = ecliptic_to_equatorial_j2000_matrix()
    if source_frame == "J2000" and destination_frame == "HELIOS_ECLIPJ2000":
        rotation = rotation.T
    elif not (source_frame == "HELIOS_ECLIPJ2000" and destination_frame == "J2000"):
        raise ValueError("unsupported inertial frame conversion")
    return np.concatenate((rotation @ values[:3], rotation @ values[3:]))
