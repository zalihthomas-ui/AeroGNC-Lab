"""Cascaded scalar and quaternion attitude-control structures."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.gnc.pid import PIDController
from aerognc.mathematics.quaternion import (
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
)
from aerognc.mathematics.vectors import FloatArray, as_vector


def wrapped_angle_error(reference_rad: float, measured_rad: float) -> float:
    """Return shortest signed angular error in ``[-pi, pi)``."""
    if not np.isfinite([reference_rad, measured_rad]).all():
        raise ValueError("angles must be finite")
    return float((reference_rad - measured_rad + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(slots=True)
class CascadedAttitudeController:
    """Scalar outer attitude loop and inner angular-rate loop."""

    attitude_controller: PIDController
    rate_controller: PIDController
    rate_command_limit_radps: float
    moment_command_limit_nm: float

    def __post_init__(self) -> None:
        if self.rate_command_limit_radps <= 0.0 or self.moment_command_limit_nm <= 0.0:
            raise ValueError("cascaded-loop command limits must be positive")

    def reset(self) -> None:
        """Reset both loops."""
        self.attitude_controller.reset()
        self.rate_controller.reset()

    def update(
        self,
        reference_angle_rad: float,
        measured_angle_rad: float,
        measured_rate_radps: float,
        step_s: float,
    ) -> tuple[float, float]:
        """Return bounded moment command and intermediate rate command."""
        attitude_error = wrapped_angle_error(reference_angle_rad, measured_angle_rad)
        rate_command = float(
            np.clip(
                self.attitude_controller.update(attitude_error, step_s),
                -self.rate_command_limit_radps,
                self.rate_command_limit_radps,
            )
        )
        moment_command = float(
            np.clip(
                self.rate_controller.update(rate_command - measured_rate_radps, step_s),
                -self.moment_command_limit_nm,
                self.moment_command_limit_nm,
            )
        )
        return moment_command, rate_command


@dataclass(frozen=True, slots=True)
class QuaternionAttitudePD:
    """Vector quaternion attitude hold with angular-rate damping."""

    proportional_moment_nm_per_rad: FloatArray
    rate_damping_nms_per_rad: FloatArray
    moment_limit_nm: FloatArray

    def __init__(
        self,
        proportional_moment_nm_per_rad: npt.ArrayLike,
        rate_damping_nms_per_rad: npt.ArrayLike,
        moment_limit_nm: npt.ArrayLike,
    ) -> None:
        proportional = as_vector(
            proportional_moment_nm_per_rad, 3, name="proportional_moment_nm_per_rad"
        )
        damping = as_vector(rate_damping_nms_per_rad, 3, name="rate_damping_nms_per_rad")
        limit = as_vector(moment_limit_nm, 3, name="moment_limit_nm")
        if np.any(proportional < 0.0) or np.any(damping < 0.0) or np.any(limit <= 0.0):
            raise ValueError("attitude gains must be nonnegative and limits positive")
        object.__setattr__(self, "proportional_moment_nm_per_rad", proportional)
        object.__setattr__(self, "rate_damping_nms_per_rad", damping)
        object.__setattr__(self, "moment_limit_nm", limit)

    def command_moment_body_nm(
        self,
        reference_quaternion_nb: npt.ArrayLike,
        measured_quaternion_nb: npt.ArrayLike,
        measured_rate_body_radps: npt.ArrayLike,
    ) -> FloatArray:
        """Return bounded body moment from shortest-path quaternion error."""
        reference = normalize_quaternion(reference_quaternion_nb)
        measured = normalize_quaternion(measured_quaternion_nb)
        error = quaternion_multiply(quaternion_inverse(measured), reference)
        if error[0] < 0.0:
            error = -error
        small_angle_error_body_rad = 2.0 * error[1:]
        rate = as_vector(measured_rate_body_radps, 3, name="measured_rate_body_radps")
        unconstrained = (
            self.proportional_moment_nm_per_rad * small_angle_error_body_rad
            - self.rate_damping_nms_per_rad * rate
        )
        return np.clip(unconstrained, -self.moment_limit_nm, self.moment_limit_nm)
