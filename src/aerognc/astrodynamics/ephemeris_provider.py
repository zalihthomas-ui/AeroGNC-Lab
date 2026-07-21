"""Coverage-aware ephemeris providers with explicit frame and time metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.astrodynamics.ephemeris import EphemerisUnavailableError, SpiceEphemeris
from aerognc.mathematics.vectors import FloatArray, as_vector


class EphemerisCoverageError(ValueError):
    """Raised when a body or epoch lies outside declared provider coverage."""


@dataclass(frozen=True, slots=True)
class EphemerisCoverage:
    """Closed epoch interval and semantic coordinate contract."""

    start_time_s: float
    end_time_s: float
    body_names: tuple[str, ...]
    frame: str
    center: str
    time_system: str

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.start_time_s, self.end_time_s])):
            raise ValueError("ephemeris coverage epochs must be finite")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("ephemeris coverage must have positive duration")
        if not self.body_names or any(not name.strip() for name in self.body_names):
            raise ValueError("ephemeris coverage requires nonempty body names")
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("ephemeris coverage body names must be unique")
        if not self.frame.strip() or not self.center.strip() or not self.time_system.strip():
            raise ValueError("ephemeris frame, center, and time system cannot be empty")

    def require(self, body_name: str, time_s: float) -> None:
        """Raise a contextual error unless body and epoch are covered."""
        if not np.isfinite(time_s):
            raise ValueError("ephemeris query time_s must be finite")
        if body_name not in self.body_names:
            raise EphemerisCoverageError(
                f"body {body_name!r} is outside provider coverage {self.body_names}"
            )
        if not self.start_time_s <= time_s <= self.end_time_s:
            raise EphemerisCoverageError(
                f"epoch {time_s} s outside [{self.start_time_s}, {self.end_time_s}] s"
            )


@dataclass(frozen=True, slots=True)
class EphemerisState:
    """One SI Cartesian state with complete source semantics."""

    body_name: str
    time_s: float
    position_m: FloatArray
    velocity_mps: FloatArray
    frame: str
    center: str
    time_system: str
    source: str

    def __post_init__(self) -> None:
        if not self.body_name.strip() or not self.source.strip():
            raise ValueError("ephemeris state body and source cannot be empty")
        if not np.isfinite(self.time_s):
            raise ValueError("ephemeris state time_s must be finite")
        position = as_vector(self.position_m, 3, name="ephemeris position_m")
        velocity = as_vector(self.velocity_mps, 3, name="ephemeris velocity_mps")
        position.flags.writeable = False
        velocity.flags.writeable = False
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_mps", velocity)


class CoverageAwareEphemeris(Protocol):
    """Shared provider interface; callers must not substitute on failure."""

    @property
    def coverage(self) -> EphemerisCoverage:
        """Return the provider's closed coverage interval and semantics."""
        ...

    def state(self, body_name: str, time_s: float) -> EphemerisState:
        """Return a covered Cartesian state or raise explicitly."""
        ...


def _state(
    coverage: EphemerisCoverage,
    body_name: str,
    time_s: float,
    position_m: FloatArray,
    velocity_mps: FloatArray,
    source: str,
) -> EphemerisState:
    return EphemerisState(
        body_name,
        float(time_s),
        position_m,
        velocity_mps,
        coverage.frame,
        coverage.center,
        coverage.time_system,
        source,
    )


@dataclass(frozen=True, slots=True)
class AnalyticalEphemerisProvider:
    """Coverage-bounded analytical Kepler ephemeris for synthetic bodies."""

    bodies: Mapping[str, CircularOrbitBody]
    primary_mu_m3_s2: float
    coverage: EphemerisCoverage
    source: str = "configured analytical Kepler ephemeris"

    def __post_init__(self) -> None:
        if not np.isfinite(self.primary_mu_m3_s2) or self.primary_mu_m3_s2 <= 0.0:
            raise ValueError("primary_mu_m3_s2 must be positive and finite")
        if set(self.bodies) != set(self.coverage.body_names):
            raise ValueError("analytical body names must exactly match declared coverage")

    def state(self, body_name: str, time_s: float) -> EphemerisState:
        """Evaluate a covered analytical state."""
        self.coverage.require(body_name, time_s)
        position, velocity = self.bodies[body_name].state_at_time(time_s, self.primary_mu_m3_s2)
        return _state(self.coverage, body_name, time_s, position, velocity, self.source)

    def state_at_time(self, body_name: str, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Return the legacy position/velocity tuple without weakening coverage checks."""
        value = self.state(body_name, time_s)
        return value.position_m.copy(), value.velocity_mps.copy()


class TabulatedEphemerisProvider:
    """Cubic-Hermite SI state table with hard coverage boundaries."""

    def __init__(
        self,
        time_s: Sequence[float],
        states_by_body: Mapping[str, np.ndarray],
        *,
        frame: str,
        center: str,
        time_system: str,
        source: str = "in-memory tabulated ephemeris",
    ) -> None:
        times = np.asarray(time_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
            raise ValueError(
                "tabulated ephemeris time must be a finite vector with at least 2 rows"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("tabulated ephemeris time must be strictly increasing")
        if not states_by_body:
            raise ValueError("tabulated ephemeris requires at least one body")
        tables: dict[str, FloatArray] = {}
        for name, raw_states in states_by_body.items():
            values = np.asarray(raw_states, dtype=np.float64)
            if not name.strip() or values.shape != (times.size, 6):
                raise ValueError("each tabulated body must have a finite (sample_count, 6) table")
            if not np.all(np.isfinite(values)):
                raise ValueError("tabulated ephemeris states must be finite")
            tables[name] = values.copy()
        if not source.strip():
            raise ValueError("tabulated ephemeris source cannot be empty")
        self._time_s = times.copy()
        self._states_by_body = tables
        self._coverage = EphemerisCoverage(
            float(times[0]),
            float(times[-1]),
            tuple(sorted(tables)),
            frame,
            center,
            time_system,
        )
        self._source = source

    @property
    def coverage(self) -> EphemerisCoverage:
        """Return exact table coverage."""
        return self._coverage

    def state(self, body_name: str, time_s: float) -> EphemerisState:
        """Interpolate position and its derivative consistently with endpoint velocities."""
        self.coverage.require(body_name, time_s)
        states = self._states_by_body[body_name]
        if time_s == self._time_s[-1]:
            index = self._time_s.size - 2
            fraction = 1.0
        else:
            index = int(np.searchsorted(self._time_s, time_s, side="right") - 1)
            index = max(index, 0)
            fraction = float(
                (time_s - self._time_s[index]) / (self._time_s[index + 1] - self._time_s[index])
            )
        step_s = float(self._time_s[index + 1] - self._time_s[index])
        position_0 = states[index, :3]
        velocity_0 = states[index, 3:]
        position_1 = states[index + 1, :3]
        velocity_1 = states[index + 1, 3:]
        s = fraction
        position = (
            (2.0 * s**3 - 3.0 * s**2 + 1.0) * position_0
            + (s**3 - 2.0 * s**2 + s) * step_s * velocity_0
            + (-2.0 * s**3 + 3.0 * s**2) * position_1
            + (s**3 - s**2) * step_s * velocity_1
        )
        velocity = (
            (6.0 * s**2 - 6.0 * s) / step_s * position_0
            + (3.0 * s**2 - 4.0 * s + 1.0) * velocity_0
            + (-6.0 * s**2 + 6.0 * s) / step_s * position_1
            + (3.0 * s**2 - 2.0 * s) * velocity_1
        )
        return _state(self.coverage, body_name, time_s, position, velocity, self._source)

    def state_at_time(self, body_name: str, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Return the legacy tuple while retaining strict coverage."""
        value = self.state(body_name, time_s)
        return value.position_m.copy(), value.velocity_mps.copy()


class SpiceEphemerisProvider:
    """Optional declared-coverage SPICE provider with no analytical fallback."""

    def __init__(
        self,
        kernel_paths: Sequence[str | Path],
        *,
        observer: str,
        body_names: tuple[str, ...],
        start_time_s: float,
        end_time_s: float,
        frame: str = "J2000",
        time_system: str = "TDB",
        aberration_correction: str = "NONE",
        epoch_et_s: float = 0.0,
    ) -> None:
        self._coverage = EphemerisCoverage(
            start_time_s,
            end_time_s,
            body_names,
            frame,
            observer,
            time_system,
        )
        self._adapter = SpiceEphemeris(
            kernel_paths,
            observer=observer,
            frame=frame,
            aberration_correction=aberration_correction,
            epoch_et_s=epoch_et_s,
        )

    @property
    def coverage(self) -> EphemerisCoverage:
        """Return user-declared coverage, which must be backed by supplied kernels."""
        return self._coverage

    def state(self, body_name: str, time_s: float) -> EphemerisState:
        """Query SPICE only after declared coverage checks; propagate kernel failures."""
        self.coverage.require(body_name, time_s)
        try:
            position, velocity = self._adapter.state_at_time(body_name, time_s)
        except Exception as error:
            raise EphemerisUnavailableError(
                f"SPICE state unavailable for {body_name!r} at {time_s} s: {error}"
            ) from error
        return _state(
            self.coverage,
            body_name,
            time_s,
            position,
            velocity,
            "user-supplied SPICE kernels",
        )
