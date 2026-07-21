import numpy as np
import pytest

from aerognc.dynamics.six_dof import RigidBodyInputs, six_dof_derivative
from aerognc.dynamics.state import project_six_dof_quaternion


def test_force_and_moment_sign_conventions_at_identity() -> None:
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    inputs = RigidBodyInputs(
        mass_kg=2.0,
        inertia_body_kgm2=np.diag([1.0, 2.0, 4.0]),
        force_body_n=[2.0, -4.0, 6.0],
        moment_body_nm=[1.0, -2.0, 4.0],
        gravity_ned_mps2=[0.0, 0.0, 9.0],
    )
    derivative = six_dof_derivative(state, inputs)
    np.testing.assert_allclose(derivative[3:6], [1.0, -2.0, 12.0])
    np.testing.assert_allclose(derivative[10:13], [1.0, -1.0, 1.0])


def test_quaternion_projection_normalises_and_rejects_zero() -> None:
    state = np.zeros(13)
    state[6] = 2.0
    projected = project_six_dof_quaternion(state)
    assert np.linalg.norm(projected[6:10]) == pytest.approx(1.0, abs=1.0e-15)
    with pytest.raises(ValueError, match="too small"):
        project_six_dof_quaternion(np.zeros(13))


def test_invalid_inertia_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        RigidBodyInputs(
            mass_kg=1.0,
            inertia_body_kgm2=np.diag([1.0, 0.0, 1.0]),
            force_body_n=np.zeros(3),
            moment_body_nm=np.zeros(3),
            gravity_ned_mps2=np.zeros(3),
        )
