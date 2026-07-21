"""Configurable local gravity models in NED coordinates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aerognc.mathematics.vectors import FloatArray

GravityMode = Literal["constant", "inverse_square"]


@dataclass(frozen=True, slots=True)
class GravityModel:
    """Local vertical gravity with positive acceleration along NED down."""

    mode: GravityMode = "inverse_square"
    sea_level_mps2: float = 9.80665
    earth_radius_m: float = 6_371_000.0

    def __post_init__(self) -> None:
        if self.mode not in {"constant", "inverse_square"}:
            raise ValueError("gravity mode must be 'constant' or 'inverse_square'")
        if self.sea_level_mps2 <= 0.0 or not np.isfinite(self.sea_level_mps2):
            raise ValueError("sea_level_mps2 must be positive and finite")
        if self.earth_radius_m <= 0.0 or not np.isfinite(self.earth_radius_m):
            raise ValueError("earth_radius_m must be positive and finite")

    def magnitude_mps2(self, altitude_m: float) -> float:
        """Return positive local gravitational acceleration magnitude."""
        if not np.isfinite(altitude_m) or altitude_m <= -self.earth_radius_m:
            raise ValueError("altitude_m is outside the gravity model domain")
        if self.mode == "constant":
            return self.sea_level_mps2
        ratio = self.earth_radius_m / (self.earth_radius_m + altitude_m)
        return float(self.sea_level_mps2 * ratio**2)

    def acceleration_ned_mps2(self, altitude_m: float) -> FloatArray:
        """Return gravity acceleration in NED coordinates [m/s²]."""
        return np.array([0.0, 0.0, self.magnitude_mps2(altitude_m)], dtype=np.float64)
