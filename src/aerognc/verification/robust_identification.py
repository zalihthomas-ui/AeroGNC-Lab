"""Robust linear regression, physical parameter mapping, and residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2  # type: ignore[import-untyped]

from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class RobustLinearFit:
    """Huber iteratively reweighted least-squares result."""

    coefficients: FloatArray
    covariance: FloatArray
    residual: FloatArray
    weights: FloatArray
    iterations: int
    condition_number: float
    residual_standard_deviation: float


def huber_linear_regression(
    regressors: npt.ArrayLike,
    observations: npt.ArrayLike,
    *,
    threshold_sigma: float = 1.5,
    maximum_iterations: int = 30,
    tolerance: float = 1.0e-10,
) -> RobustLinearFit:
    """Fit a linear model with manually implemented Huber reweighting."""
    design = np.asarray(regressors, dtype=np.float64)
    values = np.asarray(observations, dtype=np.float64)
    if design.ndim != 2 or values.shape != (design.shape[0],):
        raise ValueError("regression design and observations have incompatible shapes")
    if design.shape[0] <= design.shape[1] or not np.all(np.isfinite(design)):
        raise ValueError("regression design must be finite and overdetermined")
    if not np.all(np.isfinite(values)):
        raise ValueError("regression observations must be finite")
    if threshold_sigma <= 0.0 or maximum_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("Huber threshold, iteration limit, and tolerance must be positive")
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design,
        values,
        rcond=None,
    )
    if rank != design.shape[1]:
        raise ValueError("regression design is rank deficient")
    weights = np.ones(values.size)
    iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        residual = values - design @ coefficients
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        scale = max(scale, np.finfo(np.float64).eps)
        normalised = np.abs(residual - median) / (threshold_sigma * scale)
        new_weights = np.ones(values.size)
        large = normalised > 1.0
        new_weights[large] = 1.0 / normalised[large]
        weighted_design = design * np.sqrt(new_weights)[:, None]
        weighted_values = values * np.sqrt(new_weights)
        updated, _residuals, rank, _singular_values = np.linalg.lstsq(
            weighted_design,
            weighted_values,
            rcond=None,
        )
        if rank != design.shape[1]:
            raise ValueError("weighted regression design became rank deficient")
        relative_change = np.linalg.norm(updated - coefficients) / max(
            1.0,
            np.linalg.norm(coefficients),
        )
        coefficients = updated
        weights = new_weights
        if relative_change <= tolerance:
            break
    residual = values - design @ coefficients
    degrees_of_freedom = design.shape[0] - design.shape[1]
    weighted_sum_squares = float(np.sum(weights * residual**2))
    residual_variance = weighted_sum_squares / degrees_of_freedom
    information = design.T @ (weights[:, None] * design)
    covariance = residual_variance * np.linalg.inv(information)
    condition = (
        np.inf if singular_values[-1] == 0.0 else float(singular_values[0] / singular_values[-1])
    )
    return RobustLinearFit(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        iterations=iterations,
        condition_number=condition,
        residual_standard_deviation=float(np.sqrt(residual_variance)),
    )


@dataclass(frozen=True, slots=True)
class PhysicalParameterEstimate:
    """One mapped pitch-plant parameter with an approximate 95% interval."""

    name: str
    unit: str
    estimate: float
    standard_deviation: float
    lower_95: float
    upper_95: float


def pitch_parameters_from_coefficients(
    coefficients: npt.ArrayLike,
    covariance: npt.ArrayLike,
) -> tuple[PhysicalParameterEstimate, ...]:
    """Map ``q_dot=[a_theta,a_q,b,d][theta,q,u,1]`` to physical parameters."""
    values = np.asarray(coefficients, dtype=np.float64)
    coefficient_covariance = np.asarray(covariance, dtype=np.float64)
    if values.shape != (4,) or coefficient_covariance.shape != (4, 4):
        raise ValueError("pitch identification requires four coefficients and 4-by-4 covariance")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coefficient_covariance)):
        raise ValueError("pitch coefficients and covariance must be finite")
    a_theta, a_rate, input_gain, disturbance_gain = values
    if input_gain <= 0.0:
        raise ValueError("identified input gain must be positive for a physical inertia")
    physical = np.array(
        [
            1.0 / input_gain,
            -a_rate / input_gain,
            -a_theta / input_gain,
            disturbance_gain / input_gain,
        ]
    )
    jacobian = np.zeros((4, 4), dtype=np.float64)
    jacobian[0, 2] = -1.0 / input_gain**2
    jacobian[1, 1] = -1.0 / input_gain
    jacobian[1, 2] = a_rate / input_gain**2
    jacobian[2, 0] = -1.0 / input_gain
    jacobian[2, 2] = a_theta / input_gain**2
    jacobian[3, 2] = -disturbance_gain / input_gain**2
    jacobian[3, 3] = 1.0 / input_gain
    physical_covariance = jacobian @ coefficient_covariance @ jacobian.T
    standard_deviation = np.sqrt(np.maximum(np.diag(physical_covariance), 0.0))
    metadata = (
        ("inertia", "kg m^2"),
        ("damping", "N m s/rad"),
        ("stiffness", "N m/rad"),
        ("disturbance_moment", "N m"),
    )
    return tuple(
        PhysicalParameterEstimate(
            name=name,
            unit=unit,
            estimate=float(physical[index]),
            standard_deviation=float(standard_deviation[index]),
            lower_95=float(physical[index] - 1.96 * standard_deviation[index]),
            upper_95=float(physical[index] + 1.96 * standard_deviation[index]),
        )
        for index, (name, unit) in enumerate(metadata)
    )


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    """Residual magnitude, autocorrelation, and whiteness statistics."""

    mean: float
    rms: float
    standard_deviation: float
    durbin_watson: float
    autocorrelation: FloatArray
    maximum_absolute_autocorrelation: float
    ljung_box_q: float
    ljung_box_p_value: float
    maximum_input_residual_correlation: float


def _normalised_correlation(first: FloatArray, second: FloatArray, lag: int) -> float:
    if lag >= 0:
        left = first[lag:]
        right = second[: second.size - lag] if lag else second
    else:
        left = first[: first.size + lag]
        right = second[-lag:]
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(left @ right / denominator)


def residual_diagnostics(
    residual: npt.ArrayLike,
    input_signal: npt.ArrayLike,
    *,
    maximum_lag: int = 20,
) -> ResidualDiagnostics:
    """Calculate ACF, Ljung-Box, and residual/input correlation diagnostics."""
    values = np.asarray(residual, dtype=np.float64)
    inputs = np.asarray(input_signal, dtype=np.float64)
    if values.ndim != 1 or inputs.shape != values.shape or values.size <= maximum_lag + 2:
        raise ValueError("residual/input histories must be matching vectors longer than max lag")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(inputs)):
        raise ValueError("residual/input histories must be finite")
    if maximum_lag <= 0:
        raise ValueError("maximum_lag must be positive")
    centred = values - np.mean(values)
    variance_sum = float(centred @ centred)
    if variance_sum <= 0.0:
        raise ValueError("residual variance must be positive")
    autocorrelation = np.array(
        [1.0]
        + [
            float(centred[lag:] @ centred[:-lag] / variance_sum)
            for lag in range(1, maximum_lag + 1)
        ],
        dtype=np.float64,
    )
    sample_count = values.size
    lags = np.arange(1, maximum_lag + 1)
    ljung_box_q = float(
        sample_count * (sample_count + 2) * np.sum(autocorrelation[1:] ** 2 / (sample_count - lags))
    )
    input_correlations = np.array(
        [
            _normalised_correlation(values, inputs, lag)
            for lag in range(-maximum_lag, maximum_lag + 1)
        ]
    )
    difference = np.diff(values)
    return ResidualDiagnostics(
        mean=float(np.mean(values)),
        rms=float(np.sqrt(np.mean(values**2))),
        standard_deviation=float(np.std(values, ddof=1)),
        durbin_watson=float(difference @ difference / (values @ values)),
        autocorrelation=autocorrelation,
        maximum_absolute_autocorrelation=float(np.max(np.abs(autocorrelation[1:]))),
        ljung_box_q=ljung_box_q,
        ljung_box_p_value=float(chi2.sf(ljung_box_q, maximum_lag)),
        maximum_input_residual_correlation=float(np.max(np.abs(input_correlations))),
    )
