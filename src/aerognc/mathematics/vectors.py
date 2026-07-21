"""Validated vector and matrix helpers."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def as_vector(
    value: Sequence[float] | npt.ArrayLike,
    length: int,
    *,
    name: str = "vector",
) -> FloatArray:
    """Return a finite one-dimensional float vector of an exact length.

    Parameters
    ----------
    value:
        Values to validate.
    length:
        Required element count. Must be positive.
    name:
        Context included in exceptions.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def as_matrix3(value: npt.ArrayLike, *, name: str = "matrix") -> FloatArray:
    """Return a finite 3-by-3 float matrix."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def skew_symmetric(vector: Sequence[float] | npt.ArrayLike) -> FloatArray:
    """Construct ``S(v)`` such that ``S(v) @ w == cross(v, w)``."""
    x, y, z = as_vector(vector, 3, name="vector")
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
