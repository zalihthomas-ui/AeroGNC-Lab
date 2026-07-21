import numpy as np
import pytest

from aerognc.gnc.control_loops import QuaternionAttitudePD, wrapped_angle_error
from aerognc.mathematics.quaternion import euler321_to_quaternion


def test_wrapped_angle_error_uses_short_path() -> None:
    assert np.rad2deg(wrapped_angle_error(np.deg2rad(-179.0), np.deg2rad(179.0))) == pytest.approx(
        2.0
    )


def test_quaternion_attitude_pd_moment_sign_and_rate_damping() -> None:
    controller = QuaternionAttitudePD([10.0, 10.0, 10.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0])
    reference = euler321_to_quaternion(np.deg2rad(10.0), 0.0, 0.0)
    command = controller.command_moment_body_nm(reference, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert command[0] > 0.0
    damped = controller.command_moment_body_nm(
        [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    )
    assert damped[0] < 0.0
