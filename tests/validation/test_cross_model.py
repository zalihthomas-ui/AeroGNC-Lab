import numpy as np

from aerognc.dynamics.six_dof import RigidBodyInputs, six_dof_derivative
from aerognc.dynamics.state import project_six_dof_quaternion
from aerognc.dynamics.three_dof import point_mass_derivative
from aerognc.mathematics.integrators import integrate_fixed_step


def test_three_and_six_dof_translation_match_with_identical_inputs() -> None:
    force = np.array([3.0, -1.0, 2.0])
    gravity = np.array([0.0, 0.0, 9.80665])
    mass = 5.0
    point_initial = np.array([1.0, 2.0, 3.0, 4.0, -2.0, 1.0])
    rigid_initial = np.zeros(13)
    rigid_initial[:6] = point_initial
    rigid_initial[6] = 1.0
    inputs = RigidBodyInputs(
        mass_kg=mass,
        inertia_body_kgm2=np.eye(3),
        force_body_n=force,
        moment_body_nm=np.zeros(3),
        gravity_ned_mps2=gravity,
    )
    point = integrate_fixed_step(
        lambda _time_s, state: point_mass_derivative(
            state,
            mass_kg=mass,
            applied_force_ned_n=force,
            gravity_ned_mps2=gravity,
        ),
        point_initial,
        (0.0, 3.0),
        0.02,
    )
    rigid = integrate_fixed_step(
        lambda _time_s, state: six_dof_derivative(state, inputs),
        rigid_initial,
        (0.0, 3.0),
        0.02,
        state_projection=project_six_dof_quaternion,
    )
    np.testing.assert_allclose(rigid.state[:, :6], point.state, atol=1.0e-12)
