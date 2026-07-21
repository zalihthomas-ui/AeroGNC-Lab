"""1976 standard-atmosphere implementation for the lower 47 km."""

from dataclasses import dataclass

import numpy as np

_SEA_LEVEL_TEMPERATURE_K = 288.15
_SEA_LEVEL_PRESSURE_PA = 101_325.0
_STANDARD_GRAVITY_MPS2 = 9.80665
_AIR_GAS_CONSTANT_JPKGK = 287.05287
_AIR_GAMMA = 1.4
_EARTH_RADIUS_M = 6_356_766.0
_BASE_GEOPOTENTIAL_ALTITUDES_M = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0])
_LAPSE_RATES_KPM = np.array([-0.0065, 0.0, 0.0010, 0.0028])


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    """Atmospheric properties at one geometric altitude, all in SI units."""

    altitude_m: float
    geopotential_altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kgpm3: float
    speed_of_sound_mps: float


class StandardAtmosphere1976:
    """Hydrostatic, layer-wise ISA model from configurable lower/upper bounds.

    Input altitude is geometric above the launch/sea-level datum. It is converted to
    geopotential altitude before applying the standard lapse-rate layers.
    """

    def __init__(
        self, minimum_altitude_m: float = -500.0, maximum_altitude_m: float = 47_000.0
    ) -> None:
        if not np.isfinite([minimum_altitude_m, maximum_altitude_m]).all():
            raise ValueError("atmosphere altitude limits must be finite")
        if minimum_altitude_m >= maximum_altitude_m:
            raise ValueError("minimum_altitude_m must be less than maximum_altitude_m")
        if minimum_altitude_m <= -_EARTH_RADIUS_M:
            raise ValueError("minimum_altitude_m is below the geometric model domain")
        self.minimum_altitude_m = float(minimum_altitude_m)
        self.maximum_altitude_m = float(maximum_altitude_m)
        self._base_temperature_k, self._base_pressure_pa = self._compute_layer_bases()

    @staticmethod
    def geometric_to_geopotential(altitude_m: float) -> float:
        """Convert geometric altitude [m] to geopotential altitude [m]."""
        if not np.isfinite(altitude_m) or altitude_m <= -_EARTH_RADIUS_M:
            raise ValueError("altitude_m is outside the geometric model domain")
        return float(_EARTH_RADIUS_M * altitude_m / (_EARTH_RADIUS_M + altitude_m))

    @staticmethod
    def _compute_layer_bases() -> tuple[np.ndarray, np.ndarray]:
        temperature = np.empty(_BASE_GEOPOTENTIAL_ALTITUDES_M.size)
        pressure = np.empty_like(temperature)
        temperature[0] = _SEA_LEVEL_TEMPERATURE_K
        pressure[0] = _SEA_LEVEL_PRESSURE_PA
        for index, lapse_rate in enumerate(_LAPSE_RATES_KPM):
            altitude_change = (
                _BASE_GEOPOTENTIAL_ALTITUDES_M[index + 1] - _BASE_GEOPOTENTIAL_ALTITUDES_M[index]
            )
            temperature[index + 1] = temperature[index] + lapse_rate * altitude_change
            if lapse_rate == 0.0:
                pressure[index + 1] = pressure[index] * np.exp(
                    -_STANDARD_GRAVITY_MPS2
                    * altitude_change
                    / (_AIR_GAS_CONSTANT_JPKGK * temperature[index])
                )
            else:
                pressure[index + 1] = pressure[index] * (
                    temperature[index] / temperature[index + 1]
                ) ** (_STANDARD_GRAVITY_MPS2 / (_AIR_GAS_CONSTANT_JPKGK * lapse_rate))
        return temperature, pressure

    def properties(self, altitude_m: float) -> AtmosphereState:
        """Evaluate temperature, pressure, density, and acoustic speed."""
        if not np.isfinite(altitude_m):
            raise ValueError("altitude_m must be finite")
        if not self.minimum_altitude_m <= altitude_m <= self.maximum_altitude_m:
            raise ValueError(
                f"altitude_m={altitude_m} outside configured atmosphere range "
                f"[{self.minimum_altitude_m}, {self.maximum_altitude_m}]"
            )
        geopotential_m = self.geometric_to_geopotential(float(altitude_m))
        if geopotential_m < 0.0:
            layer = 0
        else:
            layer = int(
                np.clip(
                    np.searchsorted(_BASE_GEOPOTENTIAL_ALTITUDES_M, geopotential_m, side="right")
                    - 1,
                    0,
                    _LAPSE_RATES_KPM.size - 1,
                )
            )
        base_altitude_m = _BASE_GEOPOTENTIAL_ALTITUDES_M[layer]
        base_temperature_k = self._base_temperature_k[layer]
        base_pressure_pa = self._base_pressure_pa[layer]
        lapse_rate = _LAPSE_RATES_KPM[layer]
        altitude_change_m = geopotential_m - base_altitude_m
        temperature_k = base_temperature_k + lapse_rate * altitude_change_m
        if lapse_rate == 0.0:
            pressure_pa = base_pressure_pa * np.exp(
                -_STANDARD_GRAVITY_MPS2
                * altitude_change_m
                / (_AIR_GAS_CONSTANT_JPKGK * base_temperature_k)
            )
        else:
            pressure_pa = base_pressure_pa * (base_temperature_k / temperature_k) ** (
                _STANDARD_GRAVITY_MPS2 / (_AIR_GAS_CONSTANT_JPKGK * lapse_rate)
            )
        density_kgpm3 = pressure_pa / (_AIR_GAS_CONSTANT_JPKGK * temperature_k)
        speed_of_sound_mps = np.sqrt(_AIR_GAMMA * _AIR_GAS_CONSTANT_JPKGK * temperature_k)
        return AtmosphereState(
            altitude_m=float(altitude_m),
            geopotential_altitude_m=geopotential_m,
            temperature_k=float(temperature_k),
            pressure_pa=float(pressure_pa),
            density_kgpm3=float(density_kgpm3),
            speed_of_sound_mps=float(speed_of_sound_mps),
        )


def mach_number(airspeed_mps: float, speed_of_sound_mps: float) -> float:
    """Return Mach number from nonnegative airspeed and positive acoustic speed."""
    if not np.isfinite([airspeed_mps, speed_of_sound_mps]).all():
        raise ValueError("airspeed and speed of sound must be finite")
    if airspeed_mps < 0.0 or speed_of_sound_mps <= 0.0:
        raise ValueError("airspeed must be nonnegative and speed of sound must be positive")
    return float(airspeed_mps / speed_of_sound_mps)


def dynamic_pressure_pa(density_kgpm3: float, airspeed_mps: float) -> float:
    """Return dynamic pressure [Pa]."""
    if not np.isfinite([density_kgpm3, airspeed_mps]).all():
        raise ValueError("density and airspeed must be finite")
    if density_kgpm3 < 0.0 or airspeed_mps < 0.0:
        raise ValueError("density and airspeed must be nonnegative")
    return float(0.5 * density_kgpm3 * airspeed_mps**2)
