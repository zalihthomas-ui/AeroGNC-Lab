"""Time-indexed, non-operational research-ascent reference guidance."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class AttitudeReferenceSchedule:
    """Clamped piecewise-linear roll/pitch/yaw schedule.

    This schedule is solely a time-indexed academic ascent reference. It accepts no
    target state and contains no interception, homing, or engagement logic.
    """

    time_s: FloatArray
    euler321_rad: FloatArray

    def __init__(self, time_s: npt.ArrayLike, euler321_deg: npt.ArrayLike) -> None:
        times = np.asarray(time_s, dtype=np.float64)
        euler_deg = np.asarray(euler321_deg, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
            raise ValueError("time_s must be a finite one-dimensional array with two points")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("time_s must be strictly increasing")
        if euler_deg.shape != (times.size, 3) or not np.all(np.isfinite(euler_deg)):
            raise ValueError("euler321_deg must have shape (len(time_s), 3)")
        object.__setattr__(self, "time_s", times.copy())
        object.__setattr__(self, "euler321_rad", np.deg2rad(euler_deg))

    def euler_at_time_rad(self, time_s: float) -> FloatArray:
        """Return clamped roll, pitch, yaw reference [rad]."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        return np.array(
            [
                np.interp(time_s, self.time_s, self.euler321_rad[:, component])
                for component in range(3)
            ],
            dtype=np.float64,
        )

    def quaternion_at_time_nb(self, time_s: float) -> FloatArray:
        """Return scalar-first body-to-navigation reference quaternion."""
        roll, pitch, yaw = self.euler_at_time_rad(time_s)
        return euler321_to_quaternion(float(roll), float(pitch), float(yaw))
