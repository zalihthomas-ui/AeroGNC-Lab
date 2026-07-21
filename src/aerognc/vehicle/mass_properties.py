"""Time-varying fictional vehicle mass, centre of gravity, and inertia."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray, as_matrix3
from aerognc.vehicle.propulsion import ThrustCurve


def _positive_definite_inertia(value: npt.ArrayLike, name: str) -> FloatArray:
    inertia = as_matrix3(value, name=name)
    if not np.allclose(inertia, inertia.T, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
        raise ValueError(f"{name} must be positive definite")
    return inertia


@dataclass(frozen=True, slots=True)
class MassProperties:
    """Mass properties at an instant, all in SI units."""

    mass_kg: float
    propellant_mass_kg: float
    centre_of_gravity_from_nose_m: float
    inertia_body_kgm2: FloatArray
    inertia_rate_body_kgm2ps: FloatArray


@dataclass(frozen=True, slots=True)
class MassPropertiesModel:
    """Linear schedule versus remaining propellant fraction."""

    dry_mass_kg: float
    propulsion: ThrustCurve
    dry_cg_from_nose_m: float
    wet_cg_from_nose_m: float
    dry_inertia_body_kgm2: FloatArray
    wet_inertia_body_kgm2: FloatArray

    def __init__(
        self,
        dry_mass_kg: float,
        propulsion: ThrustCurve,
        dry_cg_from_nose_m: float,
        wet_cg_from_nose_m: float,
        dry_inertia_body_kgm2: npt.ArrayLike,
        wet_inertia_body_kgm2: npt.ArrayLike,
    ) -> None:
        scalar_values = [dry_mass_kg, dry_cg_from_nose_m, wet_cg_from_nose_m]
        if not np.isfinite(scalar_values).all():
            raise ValueError("mass and centre-of-gravity values must be finite")
        if dry_mass_kg <= 0.0:
            raise ValueError("dry_mass_kg must be positive")
        if dry_cg_from_nose_m < 0.0 or wet_cg_from_nose_m < 0.0:
            raise ValueError("centre-of-gravity locations must be nonnegative")
        object.__setattr__(self, "dry_mass_kg", float(dry_mass_kg))
        object.__setattr__(self, "propulsion", propulsion)
        object.__setattr__(self, "dry_cg_from_nose_m", float(dry_cg_from_nose_m))
        object.__setattr__(self, "wet_cg_from_nose_m", float(wet_cg_from_nose_m))
        object.__setattr__(
            self,
            "dry_inertia_body_kgm2",
            _positive_definite_inertia(dry_inertia_body_kgm2, "dry_inertia_body_kgm2"),
        )
        object.__setattr__(
            self,
            "wet_inertia_body_kgm2",
            _positive_definite_inertia(wet_inertia_body_kgm2, "wet_inertia_body_kgm2"),
        )

    @property
    def wet_mass_kg(self) -> float:
        """Initial mass [kg]."""
        return self.dry_mass_kg + self.propulsion.propellant_mass_kg

    def at_time(self, time_s: float) -> MassProperties:
        """Evaluate mass properties and prescribed inertia rate at time [s]."""
        propellant_kg = self.propulsion.propellant_remaining_kg(time_s)
        fraction = propellant_kg / self.propulsion.propellant_mass_kg
        cg_m = self.dry_cg_from_nose_m + fraction * (
            self.wet_cg_from_nose_m - self.dry_cg_from_nose_m
        )
        inertia = self.dry_inertia_body_kgm2 + fraction * (
            self.wet_inertia_body_kgm2 - self.dry_inertia_body_kgm2
        )
        fraction_rate = (
            -self.propulsion.mass_flow_rate_kgps(time_s) / self.propulsion.propellant_mass_kg
        )
        inertia_rate = fraction_rate * (self.wet_inertia_body_kgm2 - self.dry_inertia_body_kgm2)
        return MassProperties(
            mass_kg=float(self.dry_mass_kg + propellant_kg),
            propellant_mass_kg=propellant_kg,
            centre_of_gravity_from_nose_m=float(cg_m),
            inertia_body_kgm2=inertia,
            inertia_rate_body_kgm2ps=inertia_rate,
        )
