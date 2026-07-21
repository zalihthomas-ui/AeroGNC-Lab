"""Hamilton scalar-first quaternion operations.

Quaternions ``q_nb`` actively rotate body-resolved components into NED navigation
components. Composition uses Hamilton order: ``q_ac = q_ab ⊗ q_bc``.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray, as_matrix3, as_vector

_NORM_EPS = 1.0e-15


def normalize_quaternion(quaternion: Sequence[float] | npt.ArrayLike) -> FloatArray:
    """Return a unit quaternion; reject a numerically zero norm."""
    q = as_vector(quaternion, 4, name="quaternion")
    norm = float(np.linalg.norm(q))
    if norm < _NORM_EPS:
        raise ValueError("quaternion norm is too small to normalize")
    return q / norm


def quaternion_multiply(
    left: Sequence[float] | npt.ArrayLike,
    right: Sequence[float] | npt.ArrayLike,
) -> FloatArray:
    """Return Hamilton product ``left ⊗ right`` without implicit normalisation."""
    w1, x1, y1, z1 = as_vector(left, 4, name="left quaternion")
    w2, x2, y2, z2 = as_vector(right, 4, name="right quaternion")
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_conjugate(quaternion: Sequence[float] | npt.ArrayLike) -> FloatArray:
    """Return the quaternion conjugate."""
    q = as_vector(quaternion, 4, name="quaternion")
    q[1:] *= -1.0
    return q


def quaternion_inverse(quaternion: Sequence[float] | npt.ArrayLike) -> FloatArray:
    """Return the multiplicative inverse of a nonzero quaternion."""
    q = as_vector(quaternion, 4, name="quaternion")
    norm_squared = float(q @ q)
    if norm_squared < _NORM_EPS**2:
        raise ValueError("quaternion norm is too small to invert")
    return quaternion_conjugate(q) / norm_squared


def rotation_vector_to_quaternion(rotation_vector_rad: npt.ArrayLike) -> FloatArray:
    """Convert a right-hand active rotation vector [rad] to a unit quaternion.

    The small-angle branch avoids loss of precision while retaining the second-order
    scalar term needed by high-rate inertial mechanisation.
    """
    rotation_vector = as_vector(rotation_vector_rad, 3, name="rotation_vector_rad")
    angle_rad = float(np.linalg.norm(rotation_vector))
    if angle_rad <= 1.0e-8:
        angle_squared = angle_rad * angle_rad
        return normalize_quaternion(
            np.concatenate(
                (
                    [1.0 - angle_squared / 8.0],
                    (0.5 - angle_squared / 48.0) * rotation_vector,
                )
            )
        )
    half_angle_rad = 0.5 * angle_rad
    return np.concatenate(
        (
            [np.cos(half_angle_rad)],
            np.sin(half_angle_rad) * rotation_vector / angle_rad,
        )
    )


def quaternion_to_dcm(quaternion_nb: Sequence[float] | npt.ArrayLike) -> FloatArray:
    """Return ``C_nb``, mapping body components to navigation components."""
    w, x, y, z = normalize_quaternion(quaternion_nb)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def dcm_to_quaternion(dcm_nb: npt.ArrayLike) -> FloatArray:
    """Convert a proper body-to-navigation DCM to a unit quaternion.

    The input is checked for orthogonality and a positive unit determinant to catch
    accidental frame/reflection errors early.
    """
    matrix = as_matrix3(dcm_nb, name="dcm_nb")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-10):
        raise ValueError("dcm_nb must be orthogonal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-10):
        raise ValueError("dcm_nb must have determinant +1")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            q = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            q = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            q = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    q = normalize_quaternion(q)
    return -q if q[0] < 0.0 else q


def euler321_to_quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float) -> FloatArray:
    """Convert intrinsic aerospace yaw-pitch-roll angles to ``q_nb``."""
    angles = np.asarray([roll_rad, pitch_rad, yaw_rad], dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Euler angles must be finite")
    half_roll, half_pitch, half_yaw = 0.5 * angles
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    return normalize_quaternion(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quaternion_to_euler321(
    quaternion_nb: Sequence[float] | npt.ArrayLike,
) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in radians using the documented 3-2-1 sequence."""
    matrix = quaternion_to_dcm(quaternion_nb)
    sin_pitch = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sin_pitch))
    roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
    yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return roll, pitch, yaw


def rotate_vector(
    quaternion_nb: Sequence[float] | npt.ArrayLike,
    vector_b: Sequence[float] | npt.ArrayLike,
) -> FloatArray:
    """Rotate a body-resolved vector into navigation components."""
    return quaternion_to_dcm(quaternion_nb) @ as_vector(vector_b, 3, name="vector_b")
