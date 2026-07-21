import numpy as np
import pytest

from aerognc.mathematics.quaternion import (
    dcm_to_quaternion,
    euler321_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler321,
    rotate_vector,
)


def _assert_same_rotation(left: np.ndarray, right: np.ndarray) -> None:
    if np.dot(left, right) < 0.0:
        right = -right
    np.testing.assert_allclose(left, right, atol=1.0e-12)


def test_identity_rotation_and_inverse() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quaternion_to_dcm(identity), np.eye(3), atol=1.0e-15)
    np.testing.assert_allclose(quaternion_multiply(identity, identity), identity)
    np.testing.assert_allclose(quaternion_inverse(identity), identity)


@pytest.mark.parametrize(
    ("angles", "source", "expected"),
    [
        ((np.pi / 2.0, 0.0, 0.0), [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ((0.0, np.pi / 2.0, 0.0), [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ((0.0, 0.0, np.pi / 2.0), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ],
)
def test_known_ninety_degree_rotations(
    angles: tuple[float, float, float], source: list[float], expected: list[float]
) -> None:
    quaternion = euler321_to_quaternion(*angles)
    np.testing.assert_allclose(rotate_vector(quaternion, source), expected, atol=1.0e-12)


def test_composition_and_conjugate() -> None:
    q_roll = euler321_to_quaternion(np.deg2rad(20.0), 0.0, 0.0)
    q_yaw = euler321_to_quaternion(0.0, 0.0, np.deg2rad(35.0))
    composed = quaternion_multiply(q_yaw, q_roll)
    np.testing.assert_allclose(
        quaternion_to_dcm(composed),
        quaternion_to_dcm(q_yaw) @ quaternion_to_dcm(q_roll),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        quaternion_multiply(composed, quaternion_conjugate(composed)),
        [1.0, 0.0, 0.0, 0.0],
        atol=1.0e-12,
    )


def test_euler_and_dcm_round_trips_away_from_singularity() -> None:
    angles = (np.deg2rad(-32.0), np.deg2rad(41.0), np.deg2rad(117.0))
    quaternion = euler321_to_quaternion(*angles)
    recovered_angles = quaternion_to_euler321(quaternion)
    np.testing.assert_allclose(recovered_angles, angles, atol=1.0e-12)
    _assert_same_rotation(dcm_to_quaternion(quaternion_to_dcm(quaternion)), quaternion)


def test_normalisation_and_rejections() -> None:
    quaternion = normalize_quaternion([2.0, -3.0, 4.0, 1.0])
    assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1.0e-15)
    with pytest.raises(ValueError, match="too small"):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="orthogonal"):
        dcm_to_quaternion(np.ones((3, 3)))
