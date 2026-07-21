import numpy as np
import pytest

from aerognc.dynamics.rotating_six_dof import (
    RotatingRigidBodyInputs,
    RotatingRigidBodyState,
    rotating_six_dof_derivative,
)


def test_rotating_six_dof_core_has_explicit_inertial_force_and_moment_signs() -> None:
    state = RotatingRigidBodyState(
        position_inertial_m=np.array([10.0, 20.0, 30.0]),
        velocity_inertial_mps=np.array([1.0, 2.0, 3.0]),
        quaternion_ib=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_inertial_body_radps=np.zeros(3),
    ).as_array()
    inputs = RotatingRigidBodyInputs(
        mass_kg=2.0,
        inertia_body_kgm2=np.diag([1.0, 2.0, 4.0]),
        force_body_n=[4.0, -2.0, 6.0],
        moment_body_nm=[1.0, -2.0, 4.0],
        gravity_inertial_mps2=[0.0, 0.0, -1.0],
    )
    derivative = rotating_six_dof_derivative(state, inputs)

    np.testing.assert_allclose(derivative[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(derivative[3:6], [2.0, -1.0, 2.0])
    np.testing.assert_allclose(derivative[6:10], 0.0)
    np.testing.assert_allclose(derivative[10:13], [1.0, -1.0, 1.0])


def test_rotating_six_dof_inputs_reject_nonphysical_inertia() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        RotatingRigidBodyInputs(
            mass_kg=1.0,
            inertia_body_kgm2=np.diag([1.0, 0.0, 1.0]),
            force_body_n=np.zeros(3),
            moment_body_nm=np.zeros(3),
            gravity_inertial_mps2=np.zeros(3),
        )
