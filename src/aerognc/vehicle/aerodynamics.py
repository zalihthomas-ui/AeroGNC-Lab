"""Configurable synthetic aerodynamic coefficient and load model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.coordinates import aerodynamic_angles
from aerognc.mathematics.interpolation import LinearTable1D, OutOfRange
from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class AerodynamicCoefficients:
    """Body-axis aerodynamic force/moment coefficients."""

    drag: float
    side: float
    normal: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True, slots=True)
class AerodynamicLoads:
    """Aerodynamic loads and diagnostic quantities in SI units."""

    force_body_n: FloatArray
    moment_body_nm: FloatArray
    airspeed_mps: float
    alpha_rad: float
    beta_rad: float
    mach: float
    dynamic_pressure_pa: float
    coefficients: AerodynamicCoefficients


class AerodynamicCoefficientProvider(Protocol):
    """Replaceable source of body-axis aerodynamic coefficients."""

    def coefficients(
        self,
        mach: float,
        alpha_rad: float,
        beta_rad: float,
        nondimensional_rates: npt.ArrayLike = (0.0, 0.0, 0.0),
        control_coefficients: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> AerodynamicCoefficients:
        """Return coefficients at one aerodynamic condition."""
        ...


class AerodynamicModel:
    """Synthetic coefficient model designed for replaceable CFD-derived tables.

    Baseline axial drag is tabulated versus Mach. Angle and rate derivatives are
    explicit configuration values. A future coefficient provider can implement the
    same ``coefficients``/``loads`` interface with multi-dimensional tables.
    """

    def __init__(
        self,
        *,
        reference_area_m2: float,
        reference_length_m: float,
        mach_points: npt.ArrayLike | None = None,
        drag_coefficients: npt.ArrayLike | None = None,
        out_of_range: OutOfRange = "clamp",
        drag_alpha2_per_rad2: float = 0.3,
        side_beta_per_rad: float = -1.5,
        normal_alpha_per_rad: float = 2.4,
        roll_beta_per_rad: float = -0.05,
        pitch_alpha_per_rad: float = -1.8,
        yaw_beta_per_rad: float = 1.5,
        roll_rate: float = -0.12,
        pitch_rate: float = -6.0,
        yaw_rate: float = -5.0,
        coefficient_provider: AerodynamicCoefficientProvider | None = None,
    ) -> None:
        if reference_area_m2 <= 0.0 or reference_length_m <= 0.0:
            raise ValueError("aerodynamic reference dimensions must be positive")
        derivatives = np.asarray(
            [
                drag_alpha2_per_rad2,
                side_beta_per_rad,
                normal_alpha_per_rad,
                roll_beta_per_rad,
                pitch_alpha_per_rad,
                yaw_beta_per_rad,
                roll_rate,
                pitch_rate,
                yaw_rate,
            ]
        )
        if not np.all(np.isfinite(derivatives)):
            raise ValueError("aerodynamic derivatives must be finite")
        self.reference_area_m2 = float(reference_area_m2)
        self.reference_length_m = float(reference_length_m)
        if coefficient_provider is None:
            if mach_points is None or drag_coefficients is None:
                raise ValueError(
                    "mach_points and drag_coefficients are required without a coefficient provider"
                )
            legacy_drag_table = LinearTable1D(mach_points, drag_coefficients, out_of_range)
            if np.any(legacy_drag_table.y < 0.0):
                raise ValueError("drag coefficients must be nonnegative")
            drag_table: LinearTable1D | None = legacy_drag_table
        else:
            if mach_points is not None or drag_coefficients is not None:
                raise ValueError(
                    "coefficient_provider cannot be combined with legacy Mach/drag tables"
                )
            drag_table = None
        self.drag_table = drag_table
        self.coefficient_provider = coefficient_provider
        self.drag_alpha2_per_rad2 = float(drag_alpha2_per_rad2)
        self.side_beta_per_rad = float(side_beta_per_rad)
        self.normal_alpha_per_rad = float(normal_alpha_per_rad)
        self.roll_beta_per_rad = float(roll_beta_per_rad)
        self.pitch_alpha_per_rad = float(pitch_alpha_per_rad)
        self.yaw_beta_per_rad = float(yaw_beta_per_rad)
        self.roll_rate = float(roll_rate)
        self.pitch_rate = float(pitch_rate)
        self.yaw_rate = float(yaw_rate)

    def drag_coefficient(self, mach: float, alpha_rad: float = 0.0) -> float:
        """Return nonnegative drag coefficient."""
        if not np.isfinite(alpha_rad):
            raise ValueError("alpha_rad must be finite")
        if self.coefficient_provider is not None:
            return max(
                0.0,
                self.coefficient_provider.coefficients(mach, alpha_rad, 0.0).drag,
            )
        if self.drag_table is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("legacy drag table is unavailable")
        return max(0.0, self.drag_table(mach) + self.drag_alpha2_per_rad2 * alpha_rad**2)

    def coefficients(
        self,
        mach: float,
        alpha_rad: float,
        beta_rad: float,
        nondimensional_rates: npt.ArrayLike = (0.0, 0.0, 0.0),
        control_coefficients: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> AerodynamicCoefficients:
        """Evaluate body force/moment coefficients.

        Control coefficients are ordered roll, pitch, yaw. Positive normal force is
        defined upward, therefore it appears as negative body-Z coefficient.
        """
        if self.coefficient_provider is not None:
            return self.coefficient_provider.coefficients(
                mach,
                alpha_rad,
                beta_rad,
                nondimensional_rates,
                control_coefficients,
            )
        p_hat, q_hat, r_hat = as_vector(nondimensional_rates, 3, name="nondimensional_rates")
        control_roll, control_pitch, control_yaw = as_vector(
            control_coefficients, 3, name="control_coefficients"
        )
        return AerodynamicCoefficients(
            drag=self.drag_coefficient(mach, alpha_rad),
            side=self.side_beta_per_rad * beta_rad,
            normal=-self.normal_alpha_per_rad * alpha_rad,
            roll=self.roll_beta_per_rad * beta_rad + self.roll_rate * p_hat + control_roll,
            pitch=self.pitch_alpha_per_rad * alpha_rad + self.pitch_rate * q_hat + control_pitch,
            yaw=self.yaw_beta_per_rad * beta_rad + self.yaw_rate * r_hat + control_yaw,
        )

    def drag_force_n(
        self, dynamic_pressure_pa: float, mach: float, alpha_rad: float = 0.0
    ) -> float:
        """Return positive drag magnitude [N]."""
        if not np.isfinite(dynamic_pressure_pa) or dynamic_pressure_pa < 0.0:
            raise ValueError("dynamic_pressure_pa must be finite and nonnegative")
        return dynamic_pressure_pa * self.reference_area_m2 * self.drag_coefficient(mach, alpha_rad)

    def loads(
        self,
        air_velocity_body_mps: npt.ArrayLike,
        *,
        density_kgpm3: float,
        speed_of_sound_mps: float,
        angular_rate_body_radps: npt.ArrayLike = (0.0, 0.0, 0.0),
        control_coefficients: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> AerodynamicLoads:
        """Return aerodynamic force/moment about the centre of mass."""
        velocity = as_vector(air_velocity_body_mps, 3, name="air_velocity_body_mps")
        if density_kgpm3 < 0.0 or speed_of_sound_mps <= 0.0:
            raise ValueError("density must be nonnegative and speed of sound positive")
        airspeed, alpha, beta = aerodynamic_angles(velocity)
        mach = airspeed / speed_of_sound_mps
        dynamic_pressure = 0.5 * density_kgpm3 * airspeed**2
        if airspeed > 1.0e-9:
            rate_scale = self.reference_length_m / (2.0 * airspeed)
            nondimensional_rates = rate_scale * as_vector(
                angular_rate_body_radps, 3, name="angular_rate_body_radps"
            )
            drag_direction = -velocity / airspeed
        else:
            nondimensional_rates = np.zeros(3)
            drag_direction = np.zeros(3)
        coefficients = self.coefficients(
            mach, alpha, beta, nondimensional_rates, control_coefficients
        )
        scale_n = dynamic_pressure * self.reference_area_m2
        force_body_n = scale_n * (
            coefficients.drag * drag_direction
            + np.array([0.0, coefficients.side, coefficients.normal])
        )
        moment_body_nm = (
            scale_n
            * self.reference_length_m
            * np.array([coefficients.roll, coefficients.pitch, coefficients.yaw])
        )
        return AerodynamicLoads(
            force_body_n=force_body_n,
            moment_body_nm=moment_body_nm,
            airspeed_mps=airspeed,
            alpha_rad=alpha,
            beta_rad=beta,
            mach=mach,
            dynamic_pressure_pa=dynamic_pressure,
            coefficients=coefficients,
        )
