"""Direct line-of-sight, eclipse, occultation, and ground-access geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class SphericalBodyGeometry:
    """Spherical geometry body expressed in one named Cartesian frame."""

    name: str
    center_m: FloatArray
    radius_m: float
    frame: str

    def __init__(self, name: str, center_m: npt.ArrayLike, radius_m: float, frame: str) -> None:
        if not name.strip() or not frame.strip():
            raise ValueError("spherical body name and frame cannot be empty")
        if not np.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("spherical body radius_m must be positive and finite")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "center_m", as_vector(center_m, 3, name="body center_m"))
        object.__setattr__(self, "radius_m", float(radius_m))
        object.__setattr__(self, "frame", frame)


@dataclass(frozen=True, slots=True)
class OccultationResult:
    """Closest line-segment approach to one spherical body."""

    body_name: str
    occulted: bool
    closest_distance_m: float
    clearance_m: float
    segment_fraction: float
    frame: str


def spherical_occultation(
    observer_position_m: npt.ArrayLike,
    target_position_m: npt.ArrayLike,
    body: SphericalBodyGeometry,
    *,
    frame: str,
) -> OccultationResult:
    """Test whether a finite observer-target segment intersects a sphere."""
    if frame != body.frame:
        raise ValueError(f"geometry frame {frame!r} does not match body frame {body.frame!r}")
    observer = as_vector(observer_position_m, 3, name="observer_position_m")
    target = as_vector(target_position_m, 3, name="target_position_m")
    segment = target - observer
    length_squared = float(segment @ segment)
    if length_squared <= 0.0:
        raise ValueError("observer and target positions must be distinct")
    relative_observer = observer - body.center_m
    relative_target = target - body.center_m
    if (
        np.linalg.norm(relative_observer) < body.radius_m
        or np.linalg.norm(relative_target) < body.radius_m
    ):
        raise ValueError("occultation segment endpoint cannot lie inside the spherical body")
    fraction = float(np.clip(-np.dot(relative_observer, segment) / length_squared, 0.0, 1.0))
    closest = relative_observer + fraction * segment
    distance_m = float(np.linalg.norm(closest))
    clearance_m = distance_m - body.radius_m
    return OccultationResult(
        body.name,
        clearance_m <= 0.0,
        distance_m,
        clearance_m,
        fraction,
        frame,
    )


@dataclass(frozen=True, slots=True)
class LineOfSightResult:
    """Aggregate visibility across declared spherical occulting bodies."""

    clear: bool
    occultations: tuple[OccultationResult, ...]
    frame: str

    @property
    def occulting_body_names(self) -> tuple[str, ...]:
        """Return bodies intersecting the line segment."""
        return tuple(item.body_name for item in self.occultations if item.occulted)


def line_of_sight(
    observer_position_m: npt.ArrayLike,
    target_position_m: npt.ArrayLike,
    bodies: tuple[SphericalBodyGeometry, ...],
    *,
    frame: str,
) -> LineOfSightResult:
    """Return clear only when no declared spherical body intersects the segment."""
    if len({body.name for body in bodies}) != len(bodies):
        raise ValueError("line-of-sight body names must be unique")
    occultations = tuple(
        spherical_occultation(observer_position_m, target_position_m, body, frame=frame)
        for body in bodies
    )
    return LineOfSightResult(not any(item.occulted for item in occultations), occultations, frame)


EclipseState = Literal["sunlit", "penumbra", "umbra"]


@dataclass(frozen=True, slots=True)
class EclipseResult:
    """Apparent-disc eclipse classification at one observer."""

    state: EclipseState
    luminous_angular_radius_rad: float
    occulting_angular_radius_rad: float
    separation_angle_rad: float
    frame: str


def eclipse_state(
    observer_position_m: npt.ArrayLike,
    luminous_body: SphericalBodyGeometry,
    occulting_body: SphericalBodyGeometry,
    *,
    frame: str,
) -> EclipseResult:
    """Classify full light, partial overlap, or complete occultation by angular discs."""
    if luminous_body.frame != frame or occulting_body.frame != frame:
        raise ValueError("eclipse bodies and observer must use the same explicit frame")
    observer = as_vector(observer_position_m, 3, name="observer_position_m")
    luminous_vector = luminous_body.center_m - observer
    occulting_vector = occulting_body.center_m - observer
    luminous_distance = float(np.linalg.norm(luminous_vector))
    occulting_distance = float(np.linalg.norm(occulting_vector))
    if luminous_distance <= luminous_body.radius_m or occulting_distance <= occulting_body.radius_m:
        raise ValueError("eclipse observer cannot lie inside either spherical body")
    luminous_angle = float(np.arcsin(luminous_body.radius_m / luminous_distance))
    occulting_angle = float(np.arcsin(occulting_body.radius_m / occulting_distance))
    separation = float(
        np.arccos(
            np.clip(
                np.dot(luminous_vector, occulting_vector)
                / (luminous_distance * occulting_distance),
                -1.0,
                1.0,
            )
        )
    )
    state: EclipseState
    if separation >= luminous_angle + occulting_angle:
        state = "sunlit"
    elif occulting_angle >= luminous_angle + separation:
        state = "umbra"
    else:
        state = "penumbra"
    return EclipseResult(state, luminous_angle, occulting_angle, separation, frame)


@dataclass(frozen=True, slots=True)
class SphericalGroundStation:
    """Fixed geocentric station on a spherical body-fixed frame."""

    name: str
    latitude_rad: float
    longitude_rad: float
    altitude_m: float
    body_radius_m: float
    minimum_elevation_rad: float = 0.0
    frame: str = "BODY_FIXED"

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.latitude_rad,
                self.longitude_rad,
                self.altitude_m,
                self.body_radius_m,
                self.minimum_elevation_rad,
            ]
        )
        if not self.name.strip() or not self.frame.strip() or not np.all(np.isfinite(values)):
            raise ValueError("ground-station name, frame, and values must be finite/nonempty")
        if not -0.5 * np.pi <= self.latitude_rad <= 0.5 * np.pi:
            raise ValueError("ground-station latitude outside [-pi/2, pi/2]")
        if self.altitude_m < 0.0 or self.body_radius_m <= 0.0:
            raise ValueError("ground-station altitude must be nonnegative and radius positive")
        if not -0.5 * np.pi <= self.minimum_elevation_rad < 0.5 * np.pi:
            raise ValueError("minimum elevation outside [-pi/2, pi/2)")

    @property
    def position_body_fixed_m(self) -> FloatArray:
        """Return spherical body-fixed station position [m]."""
        radius_m = self.body_radius_m + self.altitude_m
        cosine_latitude = np.cos(self.latitude_rad)
        return radius_m * np.array(
            [
                cosine_latitude * np.cos(self.longitude_rad),
                cosine_latitude * np.sin(self.longitude_rad),
                np.sin(self.latitude_rad),
            ]
        )


def ground_station_elevation_rad(
    target_position_body_fixed_m: npt.ArrayLike,
    station: SphericalGroundStation,
    *,
    frame: str,
) -> float:
    """Return target elevation above the station's local spherical horizon."""
    if frame != station.frame:
        raise ValueError(f"target frame {frame!r} does not match station frame {station.frame!r}")
    target = as_vector(target_position_body_fixed_m, 3, name="target_position_body_fixed_m")
    station_position = station.position_body_fixed_m
    line = target - station_position
    distance_m = float(np.linalg.norm(line))
    if distance_m <= 0.0:
        raise ValueError("target cannot coincide with ground station")
    zenith = station_position / np.linalg.norm(station_position)
    return float(np.arcsin(np.clip(np.dot(line / distance_m, zenith), -1.0, 1.0)))


CrossingKind = Literal["rise", "set"]


@dataclass(frozen=True, slots=True)
class ElevationCrossing:
    """Interpolated minimum-elevation crossing."""

    kind: CrossingKind
    time_s: float
    elevation_rad: float


@dataclass(frozen=True, slots=True)
class GroundStationAccessResult:
    """Sampled elevation, crossings, and visible intervals."""

    time_s: FloatArray
    elevation_rad: FloatArray
    visible: npt.NDArray[np.bool_]
    crossings: tuple[ElevationCrossing, ...]
    access_intervals_s: tuple[tuple[float, float], ...]
    station_name: str
    frame: str


def ground_station_access(
    time_s: npt.ArrayLike,
    target_position_body_fixed_m: npt.ArrayLike,
    station: SphericalGroundStation,
    *,
    frame: str,
) -> GroundStationAccessResult:
    """Locate rise/set crossings and access intervals from body-fixed samples."""
    times = np.asarray(time_s, dtype=np.float64)
    positions = np.asarray(target_position_body_fixed_m, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("access time_s must be a finite vector with at least 2 samples")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("access time_s must be strictly increasing")
    if positions.shape != (times.size, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("access target positions must have finite shape (sample_count, 3)")
    elevation = np.array(
        [ground_station_elevation_rad(position, station, frame=frame) for position in positions]
    )
    shifted = elevation - station.minimum_elevation_rad
    visible = shifted >= 0.0
    crossings: list[ElevationCrossing] = []
    for index in range(1, times.size):
        before = shifted[index - 1]
        after = shifted[index]
        rising = before < 0.0 <= after
        setting = before >= 0.0 > after
        if not rising and not setting:
            continue
        denominator = abs(before) + abs(after)
        fraction = 0.5 if denominator == 0.0 else abs(before) / denominator
        crossing_time_s = times[index - 1] + fraction * (times[index] - times[index - 1])
        crossings.append(
            ElevationCrossing(
                "rise" if rising else "set",
                float(crossing_time_s),
                station.minimum_elevation_rad,
            )
        )
    interval_start: float | None = float(times[0]) if visible[0] else None
    intervals: list[tuple[float, float]] = []
    for crossing in crossings:
        if crossing.kind == "rise":
            interval_start = crossing.time_s
        elif interval_start is not None:
            intervals.append((interval_start, crossing.time_s))
            interval_start = None
    if interval_start is not None:
        intervals.append((interval_start, float(times[-1])))
    return GroundStationAccessResult(
        times.copy(),
        elevation,
        visible,
        tuple(crossings),
        tuple(intervals),
        station.name,
        frame,
    )
