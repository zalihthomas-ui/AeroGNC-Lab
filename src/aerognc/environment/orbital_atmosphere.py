"""Bounded reference atmosphere for educational orbit-decay studies.

The lower-atmosphere model elsewhere in AeroGNC-Lab remains the normative 1976 ISA
implementation. This module extends density into the thermosphere with a transparent
log-linear reference table so drag sensitivity and finite-horizon lifetime studies can
be reproduced without a network service. It is not a space-weather forecast model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerognc.environment.atmosphere import AtmosphereState, StandardAtmosphere1976

_ALTITUDE_KM = np.array(
    [
        47.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
        150.0,
        180.0,
        200.0,
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
        1_000.0,
    ],
    dtype=np.float64,
)

# Earth-like reference values are deliberately used only as a synthetic Orbis-A
# baseline. Users can scale the complete profile to expose lifetime sensitivity.
_DENSITY_KGPM3 = np.array(
    [
        1.4275e-3,
        1.057e-3,
        3.206e-4,
        8.770e-5,
        1.905e-5,
        3.396e-6,
        5.297e-7,
        9.661e-8,
        2.438e-8,
        8.484e-9,
        3.845e-9,
        2.076e-9,
        5.464e-10,
        2.789e-10,
        7.248e-11,
        2.418e-11,
        9.518e-12,
        3.725e-12,
        1.585e-12,
        6.967e-13,
        1.454e-13,
        3.614e-14,
        1.170e-14,
        5.245e-15,
        3.019e-15,
    ],
    dtype=np.float64,
)

_TEMPERATURE_ALTITUDE_KM = np.array(
    [47.0, 60.0, 80.0, 100.0, 120.0, 200.0, 500.0, 1_000.0], dtype=np.float64
)
_TEMPERATURE_K = np.array(
    [270.65, 247.0, 198.0, 195.0, 350.0, 800.0, 1_000.0, 1_000.0], dtype=np.float64
)
_AIR_GAMMA = 1.4
_AIR_GAS_CONSTANT_JPKGK = 287.05287


@dataclass(frozen=True, slots=True)
class OrbitalAtmosphereState:
    """Reference atmospheric values at one geometric altitude."""

    altitude_m: float
    density_kgpm3: float
    temperature_k: float
    speed_of_sound_mps: float


@dataclass(frozen=True, slots=True)
class ReferenceOrbitalAtmosphere:
    """ISA-to-thermosphere reference with an explicit global density scale."""

    density_scale: float = 1.0
    maximum_altitude_m: float = 1_500_000.0

    def __post_init__(self) -> None:
        values = np.array([self.density_scale, self.maximum_altitude_m])
        if not np.all(np.isfinite(values)):
            raise ValueError("orbital-atmosphere settings must be finite")
        if self.density_scale < 0.0 or self.density_scale > 1.0e6:
            raise ValueError("density_scale must lie in [0, 1e6]")
        if self.maximum_altitude_m < 1_000_000.0:
            raise ValueError("maximum_altitude_m must be at least 1000 km")

    @staticmethod
    def _lower_state(altitude_m: float) -> AtmosphereState:
        atmosphere = StandardAtmosphere1976(minimum_altitude_m=-500.0)
        return atmosphere.properties(float(np.clip(altitude_m, -500.0, 47_000.0)))

    def properties(self, altitude_m: float) -> OrbitalAtmosphereState:
        """Return bounded density, temperature, and acoustic speed in SI units."""
        if not np.isfinite(altitude_m):
            raise ValueError("altitude_m must be finite")
        if altitude_m < -500.0:
            raise ValueError("altitude_m lies below the supported surface domain")
        if altitude_m <= 47_000.0:
            lower = self._lower_state(altitude_m)
            return OrbitalAtmosphereState(
                float(altitude_m),
                self.density_scale * lower.density_kgpm3,
                lower.temperature_k,
                lower.speed_of_sound_mps,
            )

        altitude_km = altitude_m / 1_000.0
        if altitude_m > self.maximum_altitude_m:
            density = 0.0
        elif altitude_km <= _ALTITUDE_KM[-1]:
            density = float(np.exp(np.interp(altitude_km, _ALTITUDE_KM, np.log(_DENSITY_KGPM3))))
        else:
            # A bounded 200 km tail avoids an artificial discontinuity at 1000 km.
            density = float(_DENSITY_KGPM3[-1] * np.exp(-(altitude_km - 1_000.0) / 200.0))
        temperature = float(
            np.interp(
                min(altitude_km, _TEMPERATURE_ALTITUDE_KM[-1]),
                _TEMPERATURE_ALTITUDE_KM,
                _TEMPERATURE_K,
            )
        )
        speed_of_sound = float(np.sqrt(_AIR_GAMMA * _AIR_GAS_CONSTANT_JPKGK * temperature))
        return OrbitalAtmosphereState(
            float(altitude_m),
            self.density_scale * density,
            temperature,
            speed_of_sound,
        )

    def density_kgpm3(self, altitude_m: float) -> float:
        """Return density only for force-model hot paths."""
        return self.properties(altitude_m).density_kgpm3
