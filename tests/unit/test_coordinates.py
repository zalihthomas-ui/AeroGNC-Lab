import numpy as np

from aerognc.mathematics.coordinates import (
    aerodynamic_angles,
    body_to_navigation,
    launch_direction_ned,
    navigation_to_body,
)
from aerognc.mathematics.quaternion import euler321_to_quaternion


def test_transform_inverse_consistency() -> None:
    quaternion = euler321_to_quaternion(0.3, -0.4, 1.2)
    vector_b = np.array([31.0, -2.0, 7.0])
    np.testing.assert_allclose(
        navigation_to_body(body_to_navigation(vector_b, quaternion), quaternion),
        vector_b,
        atol=1.0e-12,
    )


def test_aerodynamic_angles_and_zero_speed() -> None:
    speed, alpha, beta = aerodynamic_angles([10.0, 2.0, 1.0])
    assert speed == np.sqrt(105.0)
    assert alpha == np.arctan2(1.0, 10.0)
    assert beta == np.arcsin(2.0 / np.sqrt(105.0))
    assert aerodynamic_angles([0.0, 0.0, 0.0]) == (0.0, 0.0, 0.0)


def test_vertical_launch_points_negative_down() -> None:
    np.testing.assert_allclose(launch_direction_ned(np.pi / 2.0, 0.7), [0.0, 0.0, -1.0], atol=1e-15)
