"""Synthetic asynchronous flight-data alignment and parameter-identification workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.flight_data_loader import (
    FlightDataIdentificationConfiguration,
)
from aerognc.mathematics.integrators import rk4_step
from aerognc.mathematics.signal_processing import (
    AffineClockAlignment,
    estimate_affine_clock_alignment,
    hampel_filter,
    local_polynomial_smooth_derivative,
    resample_with_gap_policy,
)
from aerognc.mathematics.vectors import FloatArray
from aerognc.verification.robust_identification import (
    PhysicalParameterEstimate,
    ResidualDiagnostics,
    RobustLinearFit,
    huber_linear_regression,
    pitch_parameters_from_coefficients,
    residual_diagnostics,
)


@dataclass(frozen=True, slots=True)
class SyntheticFlightLogPaths:
    """Measurement-only command/reference and asynchronous response logs."""

    command_log: Path
    sensor_log: Path


@dataclass(frozen=True, slots=True)
class FlightDataIdentificationResult:
    """Aligned signals, robust fit, validation, and diagnostic evidence."""

    configuration: FlightDataIdentificationConfiguration
    clock_alignment: AffineClockAlignment
    time_s: FloatArray
    commanded_moment_nm: FloatArray
    resampled_pitch_rad: FloatArray
    resampled_rate_radps: FloatArray
    cleaned_pitch_rad: FloatArray
    cleaned_rate_radps: FloatArray
    smoothed_pitch_rad: FloatArray
    smoothed_rate_radps: FloatArray
    pitch_outlier_mask: np.ndarray[tuple[int], np.dtype[np.bool_]]
    rate_outlier_mask: np.ndarray[tuple[int], np.dtype[np.bool_]]
    fit: RobustLinearFit
    parameter_estimates: tuple[PhysicalParameterEstimate, ...]
    identification_time_s: FloatArray
    identification_residual_radps2: FloatArray
    residual_diagnostics: ResidualDiagnostics
    identification_r_squared: float
    validation_start_time_s: float
    validation_pitch_prediction_rad: FloatArray
    validation_rate_prediction_radps: FloatArray
    validation_pitch_rms_deg: float
    validation_rate_rms_degps: float
    measurement_missing_fraction: float

    @property
    def detected_outlier_count(self) -> int:
        """Unique time samples flagged in either pitch or rate."""
        return int(np.count_nonzero(self.pitch_outlier_mask | self.rate_outlier_mask))


@dataclass(frozen=True, slots=True)
class FlightDataIdentificationAssessment:
    """Measurable alignment, identification, residual, and validation outcomes."""

    clock_offset_pass: bool
    clock_drift_pass: bool
    marker_fit_pass: bool
    outlier_detection_pass: bool
    parameter_accuracy_pass: bool
    parameter_interval_pass: bool
    identification_fit_pass: bool
    residual_autocorrelation_pass: bool
    validation_pitch_pass: bool
    validation_rate_pass: bool
    missing_data_policy_pass: bool

    @property
    def all_pass(self) -> bool:
        """Return whether every declared flight-data requirement passes."""
        return all(
            (
                self.clock_offset_pass,
                self.clock_drift_pass,
                self.marker_fit_pass,
                self.outlier_detection_pass,
                self.parameter_accuracy_pass,
                self.parameter_interval_pass,
                self.identification_fit_pass,
                self.residual_autocorrelation_pass,
                self.validation_pitch_pass,
                self.validation_rate_pass,
                self.missing_data_policy_pass,
            )
        )


@dataclass(frozen=True, slots=True)
class FlightDataIdentificationWorkflow:
    """Executed workflow and generated evidence paths."""

    logs: SyntheticFlightLogPaths
    result: FlightDataIdentificationResult
    aligned_csv: Path
    report_json: Path


def _command_moment(
    time_s: FloatArray | float,
    configuration: FlightDataIdentificationConfiguration,
) -> FloatArray:
    time = np.asarray(time_s, dtype=np.float64)
    value = np.zeros(time.shape, dtype=np.float64)
    for frequency_hz, amplitude_nm, phase_rad in zip(
        configuration.excitation.frequencies_hz,
        configuration.excitation.amplitudes_nm,
        configuration.excitation.phases_rad,
        strict=True,
    ):
        value += amplitude_nm * np.sin(2.0 * np.pi * frequency_hz * time + phase_rad)
    fade_in = np.clip(time / 1.0, 0.0, 1.0)
    fade_out = np.clip((configuration.duration_s - time) / 1.0, 0.0, 1.0)
    return np.asarray(
        np.clip(
            value * fade_in * fade_out,
            -configuration.excitation.limit_nm,
            configuration.excitation.limit_nm,
        ),
        dtype=np.float64,
    )


def _sync_marker(
    time_s: FloatArray,
    configuration: FlightDataIdentificationConfiguration,
) -> FloatArray:
    marker = np.zeros(time_s.shape, dtype=np.float64)
    width_s = configuration.clock.marker_width_s
    for marker_time_s in configuration.clock.marker_times_s:
        marker += np.exp(-0.5 * ((time_s - marker_time_s) / width_s) ** 2)
    return np.minimum(marker, 1.0)


def _quantise(values: FloatArray, quantum: float) -> FloatArray:
    if quantum <= 0.0:
        return values.copy()
    return np.round(values / quantum) * quantum


def generate_synthetic_flight_data_logs(
    configuration: FlightDataIdentificationConfiguration,
    output_directory: str | Path,
) -> SyntheticFlightLogPaths:
    """Generate two asynchronous measurement-only CSV logs from synthetic truth."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    step_s = configuration.integration_step_s
    integration_steps = round(configuration.duration_s / step_s)
    truth_time_s = np.arange(integration_steps + 1, dtype=np.float64) * step_s
    truth_state = np.empty((truth_time_s.size, 2), dtype=np.float64)
    truth_state[0] = 0.0
    plant = configuration.plant

    def derivative(time_s: float, state: FloatArray) -> FloatArray:
        command_nm = float(_command_moment(time_s, configuration))
        return np.array(
            [
                state[1],
                (
                    command_nm
                    + plant.disturbance_moment_nm
                    - plant.damping_nms_per_rad * state[1]
                    - plant.stiffness_nm_per_rad * state[0]
                )
                / plant.inertia_kgm2,
            ]
        )

    for index in range(integration_steps):
        truth_state[index + 1] = rk4_step(
            derivative,
            truth_time_s[index],
            truth_state[index],
            step_s,
        )

    command_count = round(configuration.duration_s * configuration.command_sample_rate_hz)
    command_time_s = (
        np.arange(command_count + 1, dtype=np.float64) / configuration.command_sample_rate_hz
    )
    command_moment_nm = _command_moment(command_time_s, configuration)
    reference_marker = _sync_marker(command_time_s, configuration)
    command_path = output / "flight_data_command_log.csv"
    with command_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("reference_time_s", "commanded_moment_nm", "sync_marker"))
        for row in zip(command_time_s, command_moment_nm, reference_marker, strict=True):
            writer.writerow(f"{float(value):.12g}" for value in row)

    sensor_count = round(configuration.duration_s * configuration.sensor_sample_rate_hz)
    sensor_true_time_s = (
        np.arange(sensor_count + 1, dtype=np.float64) / configuration.sensor_sample_rate_hz
    )
    pitch_rad = np.interp(sensor_true_time_s, truth_time_s, truth_state[:, 0])
    rate_radps = np.interp(sensor_true_time_s, truth_time_s, truth_state[:, 1])
    generator = np.random.default_rng(configuration.random_seed)
    pitch_rad += configuration.sensor.pitch_noise_std_rad * generator.standard_normal(
        pitch_rad.size
    )
    rate_radps += configuration.sensor.rate_noise_std_radps * generator.standard_normal(
        rate_radps.size
    )
    pitch_rad = _quantise(pitch_rad, configuration.sensor.pitch_quantisation_rad)
    rate_radps = _quantise(rate_radps, configuration.sensor.rate_quantisation_radps)
    for start_s, end_s in configuration.sensor.dropout_intervals_s:
        dropout = (sensor_true_time_s >= start_s) & (sensor_true_time_s < end_s)
        pitch_rad[dropout] = np.nan
        rate_radps[dropout] = np.nan
    for outlier in configuration.outliers:
        index = int(np.argmin(np.abs(sensor_true_time_s - outlier.time_s)))
        if np.isfinite(pitch_rad[index]):
            pitch_rad[index] += outlier.pitch_offset_rad
        if np.isfinite(rate_radps[index]):
            rate_radps[index] += outlier.rate_offset_radps
    sensor_marker = _sync_marker(sensor_true_time_s, configuration)
    sensor_marker += configuration.sensor.sync_noise_std * generator.standard_normal(
        sensor_marker.size
    )
    sensor_clock_s = (
        1.0 + configuration.clock.sensor_drift_ppm * 1.0e-6
    ) * sensor_true_time_s + configuration.clock.sensor_offset_s
    sensor_path = output / "flight_data_sensor_log.csv"
    with sensor_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sensor_clock_s", "pitch_angle_rad", "pitch_rate_radps", "sync_marker"))
        for timestamp_s, pitch, rate, marker in zip(
            sensor_clock_s,
            pitch_rad,
            rate_radps,
            sensor_marker,
            strict=True,
        ):
            writer.writerow(
                (
                    f"{timestamp_s:.12g}",
                    "" if not np.isfinite(pitch) else f"{pitch:.12g}",
                    "" if not np.isfinite(rate) else f"{rate:.12g}",
                    f"{marker:.12g}",
                )
            )
    return SyntheticFlightLogPaths(command_path, sensor_path)


def _read_csv(path: Path, required: tuple[str, ...]) -> dict[str, FloatArray]:
    values: dict[str, list[float]] = {name: [] for name in required}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(required):
            raise ValueError(f"{path.name} columns must be exactly {required}")
        for row_number, row in enumerate(reader, start=2):
            for name in required:
                text = row[name]
                try:
                    values[name].append(np.nan if text == "" else float(text))
                except ValueError as error:
                    raise ValueError(f"invalid {name} at {path.name} row {row_number}") from error
    result = {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}
    time = result[required[0]]
    if time.size < 5 or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError(f"{path.name} timestamps must be finite and strictly increasing")
    return result


def _simulate_identified_validation(
    time_s: FloatArray,
    command_nm: FloatArray,
    initial_index: int,
    initial_state: FloatArray,
    coefficients: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    prediction = np.full((time_s.size, 2), np.nan)
    prediction[initial_index] = initial_state

    def derivative(current_time_s: float, state: FloatArray) -> FloatArray:
        control = float(np.interp(current_time_s, time_s, command_nm))
        return np.array([state[1], coefficients @ np.array([state[0], state[1], control, 1.0])])

    for index in range(initial_index, time_s.size - 1):
        prediction[index + 1] = rk4_step(
            derivative,
            time_s[index],
            prediction[index],
            float(time_s[index + 1] - time_s[index]),
        )
    return prediction[:, 0], prediction[:, 1]


def analyse_flight_data_logs(
    logs: SyntheticFlightLogPaths,
    configuration: FlightDataIdentificationConfiguration,
) -> FlightDataIdentificationResult:
    """Reload, align, clean, identify, validate, and diagnose the two logs."""
    command = _read_csv(
        logs.command_log,
        ("reference_time_s", "commanded_moment_nm", "sync_marker"),
    )
    sensor = _read_csv(
        logs.sensor_log,
        ("sensor_clock_s", "pitch_angle_rad", "pitch_rate_radps", "sync_marker"),
    )
    clock_alignment = estimate_affine_clock_alignment(
        command["reference_time_s"],
        command["sync_marker"],
        sensor["sensor_clock_s"],
        sensor["sync_marker"],
        threshold=configuration.analysis.marker_threshold,
    )
    corrected_sensor_time_s = clock_alignment.sensor_to_reference(sensor["sensor_clock_s"])
    time_s = command["reference_time_s"]
    pitch = resample_with_gap_policy(
        corrected_sensor_time_s,
        sensor["pitch_angle_rad"],
        time_s,
        maximum_gap_s=configuration.analysis.maximum_interpolation_gap_s,
    )
    rate = resample_with_gap_policy(
        corrected_sensor_time_s,
        sensor["pitch_rate_radps"],
        time_s,
        maximum_gap_s=configuration.analysis.maximum_interpolation_gap_s,
    )
    cleaned_pitch, pitch_outliers = hampel_filter(
        pitch,
        half_window=configuration.analysis.hampel_half_window,
        threshold_sigma=configuration.analysis.hampel_threshold_sigma,
    )
    cleaned_rate, rate_outliers = hampel_filter(
        rate,
        half_window=configuration.analysis.hampel_half_window,
        threshold_sigma=configuration.analysis.hampel_threshold_sigma,
    )
    smoothed_pitch, _pitch_derivative = local_polynomial_smooth_derivative(
        time_s,
        cleaned_pitch,
        window=configuration.analysis.derivative_window,
        polynomial_order=configuration.analysis.derivative_polynomial_order,
    )
    smoothed_rate, rate_derivative = local_polynomial_smooth_derivative(
        time_s,
        cleaned_rate,
        window=configuration.analysis.derivative_window,
        polynomial_order=configuration.analysis.derivative_polynomial_order,
    )
    split_time_s = configuration.duration_s * (1.0 - configuration.analysis.validation_fraction)
    finite = np.isfinite(smoothed_pitch) & np.isfinite(smoothed_rate) & np.isfinite(rate_derivative)
    identification_mask = finite & (time_s >= 0.5) & (time_s < split_time_s)
    regressors = np.column_stack(
        (
            smoothed_pitch[identification_mask],
            smoothed_rate[identification_mask],
            command["commanded_moment_nm"][identification_mask],
            np.ones(np.count_nonzero(identification_mask)),
        )
    )
    observations = rate_derivative[identification_mask]
    fit = huber_linear_regression(
        regressors,
        observations,
        threshold_sigma=configuration.analysis.huber_threshold_sigma,
        maximum_iterations=configuration.analysis.huber_maximum_iterations,
    )
    parameters = pitch_parameters_from_coefficients(fit.coefficients, fit.covariance)
    predicted_derivative = regressors @ fit.coefficients
    residual = observations - predicted_derivative
    # Adjacent local-polynomial derivatives share most of their window and therefore
    # cannot provide independent residual samples. Diagnose whiteness at one sample
    # per non-overlapping derivative window while retaining every sample in the fit.
    diagnostic_stride = configuration.analysis.derivative_window
    diagnostic_residual = residual[::diagnostic_stride]
    diagnostic_input = regressors[::diagnostic_stride, 2]
    diagnostic_maximum_lag = min(10, (diagnostic_residual.size - 3) // 2)
    diagnostics = residual_diagnostics(
        diagnostic_residual,
        diagnostic_input,
        maximum_lag=diagnostic_maximum_lag,
    )
    total_variation = float(np.sum((observations - np.mean(observations)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total_variation

    validation_candidates = np.flatnonzero(finite & (time_s >= split_time_s))
    if validation_candidates.size == 0:
        raise ValueError("flight data contains no finite validation samples")
    validation_start = int(validation_candidates[0])
    predicted_pitch, predicted_rate = _simulate_identified_validation(
        time_s,
        command["commanded_moment_nm"],
        validation_start,
        np.array([smoothed_pitch[validation_start], smoothed_rate[validation_start]]),
        fit.coefficients,
    )
    validation_mask = finite & (np.arange(time_s.size) >= validation_start)
    pitch_rms_deg = float(
        np.rad2deg(
            np.sqrt(
                np.mean((predicted_pitch[validation_mask] - smoothed_pitch[validation_mask]) ** 2)
            )
        )
    )
    rate_rms_degps = float(
        np.rad2deg(
            np.sqrt(
                np.mean((predicted_rate[validation_mask] - smoothed_rate[validation_mask]) ** 2)
            )
        )
    )
    missing_fraction = float(np.mean(~np.isfinite(pitch) | ~np.isfinite(rate)))
    return FlightDataIdentificationResult(
        configuration=configuration,
        clock_alignment=clock_alignment,
        time_s=time_s,
        commanded_moment_nm=command["commanded_moment_nm"],
        resampled_pitch_rad=pitch,
        resampled_rate_radps=rate,
        cleaned_pitch_rad=cleaned_pitch,
        cleaned_rate_radps=cleaned_rate,
        smoothed_pitch_rad=smoothed_pitch,
        smoothed_rate_radps=smoothed_rate,
        pitch_outlier_mask=pitch_outliers,
        rate_outlier_mask=rate_outliers,
        fit=fit,
        parameter_estimates=parameters,
        identification_time_s=time_s[identification_mask],
        identification_residual_radps2=residual,
        residual_diagnostics=diagnostics,
        identification_r_squared=r_squared,
        validation_start_time_s=float(time_s[validation_start]),
        validation_pitch_prediction_rad=predicted_pitch,
        validation_rate_prediction_radps=predicted_rate,
        validation_pitch_rms_deg=pitch_rms_deg,
        validation_rate_rms_degps=rate_rms_degps,
        measurement_missing_fraction=missing_fraction,
    )


def _truth_parameters(configuration: FlightDataIdentificationConfiguration) -> dict[str, float]:
    return {
        "inertia": configuration.plant.inertia_kgm2,
        "damping": configuration.plant.damping_nms_per_rad,
        "stiffness": configuration.plant.stiffness_nm_per_rad,
        "disturbance_moment": configuration.plant.disturbance_moment_nm,
    }


def assess_flight_data_identification(
    result: FlightDataIdentificationResult,
) -> FlightDataIdentificationAssessment:
    """Assess alignment, physical estimates, residuals, and held-out validation."""
    configuration = result.configuration
    truth = _truth_parameters(configuration)
    relative_errors = {
        estimate.name: abs(estimate.estimate - truth[estimate.name])
        / max(abs(truth[estimate.name]), np.finfo(np.float64).eps)
        for estimate in result.parameter_estimates
    }
    interval_coverage = {
        estimate.name: estimate.lower_95 <= truth[estimate.name] <= estimate.upper_95
        for estimate in result.parameter_estimates
    }
    expected_outliers = len(configuration.outliers)
    return FlightDataIdentificationAssessment(
        clock_offset_pass=(
            abs(result.clock_alignment.offset_s - configuration.clock.sensor_offset_s)
            <= configuration.analysis.clock_offset_tolerance_s
        ),
        clock_drift_pass=(
            abs(result.clock_alignment.drift_ppm - configuration.clock.sensor_drift_ppm)
            <= configuration.analysis.clock_drift_tolerance_ppm
        ),
        marker_fit_pass=result.clock_alignment.marker_fit_rms_s <= 0.010,
        outlier_detection_pass=(
            expected_outliers <= result.detected_outlier_count <= 3 * expected_outliers
        ),
        parameter_accuracy_pass=all(
            error <= configuration.analysis.parameter_relative_tolerance
            for error in relative_errors.values()
        ),
        parameter_interval_pass=sum(interval_coverage.values()) >= 3,
        identification_fit_pass=result.identification_r_squared >= 0.95,
        residual_autocorrelation_pass=(
            result.residual_diagnostics.maximum_absolute_autocorrelation
            <= configuration.analysis.maximum_residual_autocorrelation
        ),
        validation_pitch_pass=(
            result.validation_pitch_rms_deg <= configuration.analysis.validation_pitch_rms_limit_deg
        ),
        validation_rate_pass=(
            result.validation_rate_rms_degps
            <= configuration.analysis.validation_rate_rms_limit_degps
        ),
        missing_data_policy_pass=result.measurement_missing_fraction > 0.0,
    )


def flight_data_identification_payload(
    result: FlightDataIdentificationResult,
) -> dict[str, object]:
    """Return a stable JSON-safe flight-data analysis report."""
    configuration = result.configuration
    assessment = assess_flight_data_identification(result)
    truth = _truth_parameters(configuration)
    parameter_payload = {
        estimate.name: {
            "unit": estimate.unit,
            "truth": truth[estimate.name],
            "estimate": estimate.estimate,
            "standard_deviation": estimate.standard_deviation,
            "confidence_interval_95": [estimate.lower_95, estimate.upper_95],
            "relative_error": abs(estimate.estimate - truth[estimate.name])
            / max(abs(truth[estimate.name]), np.finfo(np.float64).eps),
            "truth_inside_interval": estimate.lower_95 <= truth[estimate.name] <= estimate.upper_95,
        }
        for estimate in result.parameter_estimates
    }
    return {
        "scenario": configuration.name,
        "safety_scope": configuration.safety_scope,
        "clock_alignment": {
            "marker_count": result.clock_alignment.marker_count,
            "estimated_offset_s": result.clock_alignment.offset_s,
            "true_offset_s": configuration.clock.sensor_offset_s,
            "estimated_drift_ppm": result.clock_alignment.drift_ppm,
            "true_drift_ppm": configuration.clock.sensor_drift_ppm,
            "marker_fit_rms_s": result.clock_alignment.marker_fit_rms_s,
        },
        "data_quality": {
            "measurement_missing_fraction": result.measurement_missing_fraction,
            "configured_outlier_count": len(configuration.outliers),
            "detected_outlier_count": result.detected_outlier_count,
        },
        "identification": {
            "model": "theta_dot=q; q_dot=a_theta*theta+a_q*q+b*u+d",
            "huber_iterations": result.fit.iterations,
            "design_condition_number": result.fit.condition_number,
            "r_squared": result.identification_r_squared,
            "parameters": parameter_payload,
        },
        "residuals": {
            "sampling_basis": "one sample per non-overlapping derivative window",
            "mean_radps2": result.residual_diagnostics.mean,
            "rms_radps2": result.residual_diagnostics.rms,
            "durbin_watson": result.residual_diagnostics.durbin_watson,
            "maximum_absolute_autocorrelation": (
                result.residual_diagnostics.maximum_absolute_autocorrelation
            ),
            "ljung_box_q": result.residual_diagnostics.ljung_box_q,
            "ljung_box_p_value": result.residual_diagnostics.ljung_box_p_value,
            "maximum_input_residual_correlation": (
                result.residual_diagnostics.maximum_input_residual_correlation
            ),
        },
        "held_out_validation": {
            "start_time_s": result.validation_start_time_s,
            "pitch_rms_deg": result.validation_pitch_rms_deg,
            "rate_rms_degps": result.validation_rate_rms_degps,
        },
        "requirements": {
            "clock_offset_pass": assessment.clock_offset_pass,
            "clock_drift_pass": assessment.clock_drift_pass,
            "marker_fit_pass": assessment.marker_fit_pass,
            "outlier_detection_pass": assessment.outlier_detection_pass,
            "parameter_accuracy_pass": assessment.parameter_accuracy_pass,
            "parameter_interval_pass": assessment.parameter_interval_pass,
            "identification_fit_pass": assessment.identification_fit_pass,
            "residual_autocorrelation_pass": assessment.residual_autocorrelation_pass,
            "validation_pitch_pass": assessment.validation_pitch_pass,
            "validation_rate_pass": assessment.validation_rate_pass,
            "missing_data_policy_pass": assessment.missing_data_policy_pass,
            "all_pass": assessment.all_pass,
        },
        "limitations": [
            "All truth, clock errors, dropouts, outliers, and measurements are synthetic.",
            "The identified plant is a rigid, linear, single-axis educational model.",
            "Approximate confidence intervals assume the local weighted regression model.",
            "Residual tests diagnose this record and do not certify a physical vehicle.",
        ],
    }


def write_flight_data_identification_results(
    result: FlightDataIdentificationResult,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write the aligned/cleaned record and automatic JSON report."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    aligned_path = output / "flight_data_aligned_cleaned.csv"
    with aligned_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "reference_time_s",
                "commanded_moment_nm",
                "resampled_pitch_rad",
                "resampled_rate_radps",
                "cleaned_pitch_rad",
                "cleaned_rate_radps",
                "pitch_outlier_flag",
                "rate_outlier_flag",
                "validation_pitch_prediction_rad",
                "validation_rate_prediction_radps",
            )
        )
        for index, time_s in enumerate(result.time_s):
            values: tuple[float | bool, ...] = (
                time_s,
                result.commanded_moment_nm[index],
                result.resampled_pitch_rad[index],
                result.resampled_rate_radps[index],
                result.cleaned_pitch_rad[index],
                result.cleaned_rate_radps[index],
                bool(result.pitch_outlier_mask[index]),
                bool(result.rate_outlier_mask[index]),
                result.validation_pitch_prediction_rad[index],
                result.validation_rate_prediction_radps[index],
            )
            writer.writerow(
                str(value).lower()
                if isinstance(value, bool)
                else ("" if not np.isfinite(value) else f"{value:.10g}")
                for value in values
            )
    report_path = output / "flight_data_identification_report.json"
    report_path.write_text(
        json.dumps(flight_data_identification_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aligned_path, report_path


def run_flight_data_identification_workflow(
    configuration: FlightDataIdentificationConfiguration,
    output_directory: str | Path | None = None,
) -> FlightDataIdentificationWorkflow:
    """Generate, reload, analyse, validate, and report one synthetic record."""
    output = Path(output_directory or configuration.output_directory)
    logs = generate_synthetic_flight_data_logs(configuration, output)
    result = analyse_flight_data_logs(logs, configuration)
    aligned_path, report_path = write_flight_data_identification_results(result, output)
    return FlightDataIdentificationWorkflow(logs, result, aligned_path, report_path)
