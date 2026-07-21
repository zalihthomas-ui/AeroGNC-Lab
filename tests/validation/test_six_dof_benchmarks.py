import numpy as np

from aerognc.dynamics.six_dof import RigidBodyInputs, six_dof_derivative
from aerognc.dynamics.state import project_six_dof_quaternion
from aerognc.mathematics.integrators import integrate_fixed_step
from aerognc.mathematics.quaternion import quaternion_to_dcm


def _state(velocity: np.ndarray | None = None, rate: np.ndarray | None = None) -> np.ndarray:
    state = np.zeros(13)
    state[3:6] = np.zeros(3) if velocity is None else velocity
    state[6] = 1.0
    state[10:13] = np.zeros(3) if rate is None else rate
    return state


def _integrate(inputs: RigidBodyInputs, initial: np.ndarray, duration_s: float = 5.0) -> np.ndarray:
    result = integrate_fixed_step(
        lambda _time_s, state: six_dof_derivative(state, inputs),
        initial,
        (0.0, duration_s),
        0.01,
        state_projection=project_six_dof_quaternion,
    )
    assert np.max(np.abs(np.linalg.norm(result.state[:, 6:10], axis=1) - 1.0)) < 1.0e-12
    return result.state


def test_force_free_translation_is_uniform() -> None:
    velocity = np.array([12.0, -3.0, 7.0])
    inputs = RigidBodyInputs(
        mass_kg=4.0,
        inertia_body_kgm2=np.eye(3),
        force_body_n=np.zeros(3),
        moment_body_nm=np.zeros(3),
        gravity_ned_mps2=np.zeros(3),
    )
    trajectory = _integrate(inputs, _state(velocity))
    np.testing.assert_allclose(trajectory[-1, :3], 5.0 * velocity, atol=1.0e-11)
    np.testing.assert_allclose(trajectory[-1, 3:6], velocity, atol=1.0e-13)


def test_constant_body_force_matches_analytical_translation() -> None:
    force = np.array([8.0, -4.0, 2.0])
    mass = 2.0
    inputs = RigidBodyInputs(
        mass_kg=mass,
        inertia_body_kgm2=np.diag([1.0, 2.0, 3.0]),
        force_body_n=force,
        moment_body_nm=np.zeros(3),
        gravity_ned_mps2=[0.0, 0.0, 9.80665],
    )
    trajectory = _integrate(inputs, _state(), duration_s=3.0)
    acceleration = force / mass + np.array([0.0, 0.0, 9.80665])
    np.testing.assert_allclose(trajectory[-1, :3], 0.5 * acceleration * 3.0**2, rtol=1e-12)
    np.testing.assert_allclose(trajectory[-1, 3:6], acceleration * 3.0, rtol=1e-12)


def test_constant_principal_axis_torque_matches_angular_acceleration() -> None:
    inputs = RigidBodyInputs(
        mass_kg=1.0,
        inertia_body_kgm2=np.diag([2.0, 3.0, 4.0]),
        force_body_n=np.zeros(3),
        moment_body_nm=[1.0, 0.0, 0.0],
        gravity_ned_mps2=np.zeros(3),
    )
    trajectory = _integrate(inputs, _state(), duration_s=2.0)
    np.testing.assert_allclose(trajectory[-1, 10:13], [1.0, 0.0, 0.0], atol=1.0e-11)


def test_torque_free_spherical_body_has_constant_rate_and_orthogonal_dcm() -> None:
    initial_rate = np.array([0.3, -0.2, 0.5])
    inputs = RigidBodyInputs(
        mass_kg=1.0,
        inertia_body_kgm2=2.0 * np.eye(3),
        force_body_n=np.zeros(3),
        moment_body_nm=np.zeros(3),
        gravity_ned_mps2=np.zeros(3),
    )
    trajectory = _integrate(inputs, _state(rate=initial_rate), duration_s=8.0)
    np.testing.assert_allclose(
        trajectory[:, 10:13],
        np.broadcast_to(initial_rate, trajectory[:, 10:13].shape),
        atol=1.0e-12,
    )
    final_dcm = quaternion_to_dcm(trajectory[-1, 6:10])
    np.testing.assert_allclose(final_dcm.T @ final_dcm, np.eye(3), atol=1.0e-12)


def test_torque_free_axisymmetric_body_conserves_energy_and_angular_momentum_norm() -> None:
    inertia = np.diag([1.0, 2.0, 2.0])
    inputs = RigidBodyInputs(
        mass_kg=1.0,
        inertia_body_kgm2=inertia,
        force_body_n=np.zeros(3),
        moment_body_nm=np.zeros(3),
        gravity_ned_mps2=np.zeros(3),
    )
    trajectory = _integrate(inputs, _state(rate=np.array([0.7, 0.2, -0.3])), duration_s=10.0)
    rates = trajectory[:, 10:13]
    energy = 0.5 * np.einsum("ij,ij->i", rates @ inertia, rates)
    momentum_norm = np.linalg.norm(rates @ inertia, axis=1)
    assert np.ptp(energy) / energy[0] < 1.0e-9
    assert np.ptp(momentum_norm) / momentum_norm[0] < 1.0e-9
