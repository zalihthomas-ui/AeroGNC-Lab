"""Deterministic vertical wind profiles and reproducible stochastic gusts."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class WindProfile:
    """Piecewise-linear NED wind velocity versus geometric altitude."""

    altitudes_m: FloatArray
    velocities_ned_mps: FloatArray

    def __init__(self, altitudes_m: npt.ArrayLike, velocities_ned_mps: npt.ArrayLike) -> None:
        altitudes = np.asarray(altitudes_m, dtype=np.float64)
        velocities = np.asarray(velocities_ned_mps, dtype=np.float64)
        if altitudes.ndim != 1 or altitudes.size < 2 or not np.all(np.isfinite(altitudes)):
            raise ValueError("altitudes_m must be a finite one-dimensional array with two points")
        if not np.all(np.diff(altitudes) > 0.0):
            raise ValueError("altitudes_m must be strictly increasing")
        if velocities.shape != (altitudes.size, 3) or not np.all(np.isfinite(velocities)):
            raise ValueError("velocities_ned_mps must have shape (len(altitudes_m), 3)")
        object.__setattr__(self, "altitudes_m", altitudes.copy())
        object.__setattr__(self, "velocities_ned_mps", velocities.copy())

    @classmethod
    def constant(cls, velocity_ned_mps: npt.ArrayLike) -> "WindProfile":
        """Construct an altitude-independent profile."""
        velocity = as_vector(velocity_ned_mps, 3, name="velocity_ned_mps")
        return cls([-1.0e6, 1.0e6], np.vstack([velocity, velocity]))

    def velocity_ned_mps(self, altitude_m: float) -> FloatArray:
        """Return clamped linearly interpolated wind velocity."""
        if not np.isfinite(altitude_m):
            raise ValueError("altitude_m must be finite")
        return np.array(
            [
                np.interp(
                    altitude_m,
                    self.altitudes_m,
                    self.velocities_ned_mps[:, component],
                )
                for component in range(3)
            ],
            dtype=np.float64,
        )


class WindModel:
    """Profile plus precomputed seeded first-order Gauss-Markov gust sequence.

    Gusts are generated at construction and linearly interpolated. A query therefore
    has no side effects, which is important because multistage ODE solvers may call a
    model repeatedly or out of order.
    """

    def __init__(
        self,
        profile: WindProfile,
        *,
        gust_std_ned_mps: npt.ArrayLike = (0.0, 0.0, 0.0),
        correlation_time_s: float = 2.0,
        sample_step_s: float = 0.1,
        horizon_s: float = 300.0,
        seed: int = 0,
    ) -> None:
        gust_std = as_vector(gust_std_ned_mps, 3, name="gust_std_ned_mps")
        if np.any(gust_std < 0.0):
            raise ValueError("gust standard deviations must be nonnegative")
        if correlation_time_s <= 0.0 or sample_step_s <= 0.0 or horizon_s <= 0.0:
            raise ValueError("gust time constants, sample step, and horizon must be positive")
        self.profile = profile
        self.seed = int(seed)
        self.gust_std_ned_mps = gust_std.copy()
        self.correlation_time_s = float(correlation_time_s)
        self.sample_step_s = float(sample_step_s)
        self.horizon_s = float(horizon_s)
        self._time_s = np.arange(0.0, horizon_s + sample_step_s, sample_step_s)
        self._gust_ned_mps = np.zeros((self._time_s.size, 3), dtype=np.float64)
        decay = float(np.exp(-sample_step_s / correlation_time_s))
        innovation_scale = np.sqrt(1.0 - decay**2) * gust_std
        generator = np.random.default_rng(self.seed)
        for index in range(1, self._time_s.size):
            self._gust_ned_mps[index] = decay * self._gust_ned_mps[
                index - 1
            ] + innovation_scale * generator.standard_normal(3)

    def velocity_ned_mps(self, time_s: float, altitude_m: float) -> FloatArray:
        """Return deterministic profile plus time-correlated gust [m/s]."""
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        if time_s > self._time_s[-1]:
            raise ValueError(f"time_s={time_s} exceeds stochastic wind horizon {self._time_s[-1]}")
        gust = np.array(
            [
                np.interp(time_s, self._time_s, self._gust_ned_mps[:, component])
                for component in range(3)
            ]
        )
        return np.asarray(self.profile.velocity_ned_mps(altitude_m) + gust, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class OneCosineGust:
    """Deterministic finite 1-cosine wind pulse in local NED components."""

    start_time_s: float
    duration_s: float
    amplitude_ned_mps: FloatArray

    def __init__(
        self,
        start_time_s: float,
        duration_s: float,
        amplitude_ned_mps: npt.ArrayLike,
    ) -> None:
        amplitude = as_vector(amplitude_ned_mps, 3, name="gust_amplitude_ned_mps")
        if (
            not np.all(np.isfinite([start_time_s, duration_s]))
            or start_time_s < 0.0
            or duration_s <= 0.0
        ):
            raise ValueError("1-cosine gust start/duration must be finite and ordered")
        object.__setattr__(self, "start_time_s", float(start_time_s))
        object.__setattr__(self, "duration_s", float(duration_s))
        object.__setattr__(self, "amplitude_ned_mps", amplitude.copy())

    def velocity_ned_mps(self, time_s: float) -> FloatArray:
        """Return a smooth zero-to-peak-to-zero pulse with no query side effects."""
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("1-cosine gust query time must be finite and nonnegative")
        phase = (time_s - self.start_time_s) / self.duration_s
        if phase < 0.0 or phase > 1.0:
            return np.zeros(3, dtype=np.float64)
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        return np.asarray(shape * self.amplitude_ned_mps, dtype=np.float64)
