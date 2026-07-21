"""Analytical, tabular, and optional external planetary ephemeris interfaces."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.mathematics.vectors import FloatArray


class EphemerisUnavailableError(RuntimeError):
    """Raised when an explicitly selected optional ephemeris cannot be used."""


class EphemerisProvider(Protocol):
    """Common primary-centred SI ephemeris interface."""

    def state_at_time(self, body_name: str, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Return position in metres and velocity in metres per second."""
        ...


@dataclass(frozen=True, slots=True)
class AnalyticalEphemeris:
    """Configured Keplerian ephemeris used by the reproducible public examples."""

    bodies: Mapping[str, CircularOrbitBody]
    primary_mu_m3_s2: float

    def state_at_time(self, body_name: str, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Evaluate a configured analytical body state."""
        try:
            body = self.bodies[body_name]
        except KeyError as error:
            raise KeyError(f"analytical ephemeris has no body named {body_name!r}") from error
        return body.state_at_time(time_s, self.primary_mu_m3_s2)


@dataclass(frozen=True, slots=True)
class TabulatedEphemeris:
    """Linearly interpolated SI ephemeris with explicit error/hold extrapolation."""

    time_s: FloatArray
    states_by_body: Mapping[str, FloatArray]
    out_of_range: str = "error"

    def __post_init__(self) -> None:
        times = np.asarray(self.time_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
            raise ValueError("ephemeris time must be a finite one-dimensional array")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("ephemeris time must be strictly increasing")
        if self.out_of_range not in {"error", "hold"}:
            raise ValueError("ephemeris out_of_range must be error or hold")
        for name, states in self.states_by_body.items():
            values = np.asarray(states, dtype=np.float64)
            if (
                not name.strip()
                or values.shape != (times.size, 6)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError("each ephemeris body must have a finite (sample_count, 6) table")

    def state_at_time(self, body_name: str, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Interpolate one body state at the requested epoch."""
        if not np.isfinite(time_s):
            raise ValueError("ephemeris query time must be finite")
        try:
            states = np.asarray(self.states_by_body[body_name], dtype=np.float64)
        except KeyError as error:
            raise KeyError(f"tabulated ephemeris has no body named {body_name!r}") from error
        query = time_s
        if query < self.time_s[0] or query > self.time_s[-1]:
            if self.out_of_range == "error":
                raise ValueError("ephemeris query lies outside the configured table")
            query = float(np.clip(query, self.time_s[0], self.time_s[-1]))
        interpolated = np.array(
            [np.interp(query, self.time_s, states[:, component]) for component in range(6)]
        )
        return interpolated[:3], interpolated[3:]


class SpiceEphemeris:
    """Optional SPICE adapter; kernels and ``spiceypy`` are always user-supplied."""

    def __init__(
        self,
        kernel_paths: Sequence[str | Path],
        *,
        observer: str,
        frame: str = "J2000",
        aberration_correction: str = "NONE",
        epoch_et_s: float = 0.0,
    ) -> None:
        if not kernel_paths:
            raise ValueError("at least one SPICE kernel path is required")
        try:
            spice = importlib.import_module("spiceypy")
        except ImportError as error:
            raise EphemerisUnavailableError(
                "SPICE ephemerides require the optional 'spiceypy' package and user-supplied "
                "public kernels; no external data are bundled"
            ) from error
        paths = [Path(path).expanduser().resolve() for path in kernel_paths]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise EphemerisUnavailableError(f"SPICE kernel was not found: {missing[0]}")
        self._spice = spice
        self._observer = observer
        self._frame = frame
        self._aberration_correction = aberration_correction
        self._epoch_et_s = float(epoch_et_s)
        for path in paths:
            self._spice.furnsh(str(path))

    def state_at_time(self, body_name: str, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Query SPICE and convert its km/km-s state into SI units."""
        if not np.isfinite(time_s):
            raise ValueError("SPICE query time must be finite")
        state_km, _light_time_s = self._spice.spkezr(
            body_name,
            self._epoch_et_s + time_s,
            self._frame,
            self._aberration_correction,
            self._observer,
        )
        state = np.asarray(cast(list[float], state_km), dtype=np.float64) * 1_000.0
        return state[:3], state[3:]
