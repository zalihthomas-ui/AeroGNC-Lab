import numpy as np
import pytest
from scipy.linalg import solve_continuous_are

from aerognc.gnc.flight_analysis import (
    analyze_modes,
    benchmark_controller_sil,
    build_lqr_gain_schedule,
    continuous_lqr,
    controllability_matrix,
    frequency_response,
    identify_linear_state_space,
    linearize_dynamics,
    observability_matrix,
    solve_trim,
    stability_margins_siso,
)


def test_trim_and_central_linearisation_match_known_nonlinear_model() -> None:
    trim = solve_trim(lambda decision: np.array([decision[0] ** 2 - 4.0]), [1.0])

    def dynamics(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return np.array([state[1], -9.81 * np.sin(state[0]) - 0.4 * state[1] + control[0]])

    model = linearize_dynamics(dynamics, [0.0, 0.0], [0.0])

    assert trim.converged
    assert trim.decision[0] == pytest.approx(2.0, abs=1.0e-9)
    np.testing.assert_allclose(model.system_matrix, [[0.0, 1.0], [-9.81, -0.4]], atol=1.0e-9)
    np.testing.assert_allclose(model.input_matrix, [[0.0], [1.0]], atol=1.0e-10)
    assert (
        np.linalg.matrix_rank(controllability_matrix(model.system_matrix, model.input_matrix)) == 2
    )
    assert np.linalg.matrix_rank(observability_matrix(model.system_matrix, [1.0, 0.0])) == 2


def test_manual_hamiltonian_lqr_matches_independent_scipy_reference() -> None:
    system = np.array([[0.0, 1.0], [0.0, 0.0]])
    inputs = np.array([[0.0], [1.0]])
    state_weight = np.diag([10.0, 1.0])
    input_weight = np.array([[2.0]])

    design = continuous_lqr(system, inputs, state_weight, input_weight)
    reference_riccati = solve_continuous_are(system, inputs, state_weight, input_weight)
    reference_gain = np.linalg.solve(input_weight, inputs.T @ reference_riccati)

    np.testing.assert_allclose(design.riccati_solution, reference_riccati, rtol=1.0e-10)
    np.testing.assert_allclose(design.gain, reference_gain, rtol=1.0e-10)
    assert design.riccati_residual_norm < 1.0e-10
    assert np.all(design.closed_loop_eigenvalues.real < 0.0)
    modes = analyze_modes(system - inputs @ design.gain)
    assert all(mode.stable and mode.damping_ratio > 0.0 for mode in modes)

    models = (
        linearize_dynamics(lambda x, u: system @ x + inputs @ u, [0.0, 0.0], [0.0]),
        linearize_dynamics(lambda x, u: 1.2 * system @ x + inputs @ u, [0.0, 0.0], [0.0]),
    )
    gains, scheduled_designs = build_lqr_gain_schedule(
        [0.0, 1.0], models, state_weight, input_weight
    )
    assert gains.shape == (2, 2)
    assert len(scheduled_designs) == 2


def test_frequency_margins_identification_and_sil_timing_are_quantitative() -> None:
    system = np.array([[0.0, 1.0], [0.0, -1.0]])
    inputs = np.array([[0.0], [1.0]])
    outputs = np.array([[1.0, 0.0]])
    frequency = np.logspace(-3, 3, 10_000)
    response = frequency_response(system, inputs, outputs, [[0.0]], frequency)[:, 0, 0]
    margins = stability_margins_siso(frequency, response)

    assert margins.phase_margin_deg == pytest.approx(51.83, abs=0.15)
    assert margins.gain_crossover_radps == pytest.approx(0.786, abs=0.01)
    assert np.isinf(margins.gain_margin)

    time_s = np.linspace(0.0, 20.0, 4_001)
    states = np.column_stack((np.sin(time_s), np.cos(time_s)))
    controls = np.sin(2.0 * time_s)
    identified = identify_linear_state_space(time_s, states, controls)
    np.testing.assert_allclose(identified.system_matrix, [[0.0, 1.0], [-1.0, 0.0]], atol=2.0e-5)
    np.testing.assert_allclose(identified.input_matrix, np.zeros((2, 1)), atol=2.0e-5)
    assert np.max(identified.derivative_rms_error) < 2.0e-5

    timing = benchmark_controller_sil(
        lambda sample: np.array([-2.0 * sample[0] - 0.5 * sample[1]]),
        states[:100],
        deadline_s=0.1,
        repeat_count=2,
    )
    assert timing.sample_count == 200
    assert 0.0 <= timing.mean_execution_s <= timing.maximum_execution_s
    assert timing.missed_deadline_count == 0
    assert np.isfinite(timing.output_checksum)
