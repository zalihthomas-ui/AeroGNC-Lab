"""Synthetic tabulated propulsion and propellant-depletion model."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.interpolation import LinearTable1D
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class ThrustCurve:
    """Piecewise-linear thrust curve with impulse-proportional propellant use.

    This transparent approximation is suitable for the fictional solid-motor
    research vehicle. Thrust is zero before the first and after the last time point.
    """

    time_s: FloatArray
    thrust_n: FloatArray
    propellant_mass_kg: float
    total_impulse_ns: float
    _cumulative_impulse_ns: FloatArray
    _table: LinearTable1D

    def __init__(
        self,
        time_s: npt.ArrayLike,
        thrust_n: npt.ArrayLike,
        propellant_mass_kg: float,
    ) -> None:
        times = np.asarray(time_s, dtype=np.float64)
        thrust = np.asarray(thrust_n, dtype=np.float64)
        table = LinearTable1D(times, thrust, out_of_range="clamp")
        if np.any(thrust < 0.0):
            raise ValueError("thrust_n cannot contain negative values")
        if not np.isfinite(propellant_mass_kg) or propellant_mass_kg <= 0.0:
            raise ValueError("propellant_mass_kg must be positive and finite")
        segment_impulse = 0.5 * (thrust[:-1] + thrust[1:]) * np.diff(times)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_impulse)))
        total_impulse = float(cumulative[-1])
        if total_impulse <= 0.0:
            raise ValueError("thrust curve total impulse must be positive")
        object.__setattr__(self, "time_s", times.copy())
        object.__setattr__(self, "thrust_n", thrust.copy())
        object.__setattr__(self, "propellant_mass_kg", float(propellant_mass_kg))
        object.__setattr__(self, "total_impulse_ns", total_impulse)
        object.__setattr__(self, "_cumulative_impulse_ns", cumulative)
        object.__setattr__(self, "_table", table)

    @property
    def ignition_time_s(self) -> float:
        """First tabulated motor time [s]."""
        return float(self.time_s[0])

    @property
    def burnout_time_s(self) -> float:
        """Last tabulated motor time [s]."""
        return float(self.time_s[-1])

    def thrust_at_time_n(self, time_s: float) -> float:
        """Return thrust [N], explicitly zero outside the motor interval."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if time_s < self.ignition_time_s or time_s > self.burnout_time_s:
            return 0.0
        return self._table(float(time_s))

    def delivered_impulse_ns(self, time_s: float) -> float:
        """Return exact integral of the piecewise-linear curve through ``time_s``."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if time_s <= self.ignition_time_s:
            return 0.0
        if time_s >= self.burnout_time_s:
            return self.total_impulse_ns
        index = int(np.searchsorted(self.time_s, time_s, side="right") - 1)
        elapsed_s = float(time_s - self.time_s[index])
        segment_duration_s = float(self.time_s[index + 1] - self.time_s[index])
        thrust_slope_nps = float(
            (self.thrust_n[index + 1] - self.thrust_n[index]) / segment_duration_s
        )
        partial_impulse_ns = (
            self.thrust_n[index] * elapsed_s + 0.5 * thrust_slope_nps * elapsed_s**2
        )
        return float(self._cumulative_impulse_ns[index] + partial_impulse_ns)

    def propellant_remaining_kg(self, time_s: float) -> float:
        """Return remaining propellant bounded to ``[0, initial]`` [kg]."""
        burned_fraction = self.delivered_impulse_ns(time_s) / self.total_impulse_ns
        remaining = self.propellant_mass_kg * (1.0 - burned_fraction)
        return float(np.clip(remaining, 0.0, self.propellant_mass_kg))

    def mass_flow_rate_kgps(self, time_s: float) -> float:
        """Return nonnegative propellant mass-flow magnitude [kg/s]."""
        return self.propellant_mass_kg * self.thrust_at_time_n(time_s) / self.total_impulse_ns
