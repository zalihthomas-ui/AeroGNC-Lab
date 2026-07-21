import numpy as np
import pytest

from aerognc.verification.robust_identification import (
    huber_linear_regression,
    pitch_parameters_from_coefficients,
    residual_diagnostics,
)


def test_huber_fit_rejects_outliers_and_recovers_coefficients() -> None:
    generator = np.random.default_rng(391)
    design = np.column_stack((generator.normal(size=(500, 3)), np.ones(500)))
    truth = np.array([-0.32, -0.09, 0.08, 0.014])
    observations = design @ truth + generator.normal(0.0, 0.004, 500)
    observations[::55] += 1.2

    fit = huber_linear_regression(design, observations)

    assert fit.coefficients == pytest.approx(truth, abs=1.5e-3)
    assert np.count_nonzero(fit.weights < 0.2) >= 8
    assert fit.condition_number < 2.0


def test_pitch_parameter_mapping_recovers_physical_plant() -> None:
    inertia = 12.5
    coefficients = np.array([-4.0 / inertia, -1.2 / inertia, 1.0 / inertia, 0.2 / inertia])
    estimates = pitch_parameters_from_coefficients(coefficients, np.eye(4) * 1.0e-10)

    values = {estimate.name: estimate.estimate for estimate in estimates}
    assert values == pytest.approx(
        {"inertia": 12.5, "damping": 1.2, "stiffness": 4.0, "disturbance_moment": 0.2}
    )
    assert all(estimate.lower_95 < estimate.estimate < estimate.upper_95 for estimate in estimates)


def test_residual_diagnostics_are_finite_and_bounded() -> None:
    generator = np.random.default_rng(114)
    residual = generator.normal(size=500)
    input_signal = generator.normal(size=500)

    diagnostics = residual_diagnostics(residual, input_signal, maximum_lag=12)

    assert diagnostics.autocorrelation.shape == (13,)
    assert diagnostics.autocorrelation[0] == pytest.approx(1.0)
    assert 0.0 <= diagnostics.ljung_box_p_value <= 1.0
    assert diagnostics.maximum_absolute_autocorrelation < 0.2
