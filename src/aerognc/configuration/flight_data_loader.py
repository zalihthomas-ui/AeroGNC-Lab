"""Strict configuration for synthetic flight-data alignment and identification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.loader import (
    ConfigurationError,
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _sequence,
    _string,
)


@dataclass(frozen=True, slots=True)
class IdentificationPlantConfiguration:
    """Synthetic pitch-channel truth parameters in SI units."""

    inertia_kgm2: float
    damping_nms_per_rad: float
    stiffness_nm_per_rad: float
    disturbance_moment_nm: float


@dataclass(frozen=True, slots=True)
class ExcitationConfiguration:
    """Bounded multisine moment command."""

    frequencies_hz: tuple[float, ...]
    amplitudes_nm: tuple[float, ...]
    phases_rad: tuple[float, ...]
    limit_nm: float


@dataclass(frozen=True, slots=True)
class ClockConfiguration:
    """Synthetic sensor clock and common synchronization markers."""

    sensor_offset_s: float
    sensor_drift_ppm: float
    marker_times_s: tuple[float, ...]
    marker_width_s: float


@dataclass(frozen=True, slots=True)
class FlightDataSensorConfiguration:
    """Asynchronous pitch/rate sensor errors and missing-data intervals."""

    pitch_noise_std_rad: float
    rate_noise_std_radps: float
    pitch_quantisation_rad: float
    rate_quantisation_radps: float
    sync_noise_std: float
    dropout_intervals_s: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class SyntheticFlightDataOutlier:
    """One isolated synthetic response-channel outlier."""

    time_s: float
    pitch_offset_rad: float
    rate_offset_radps: float


@dataclass(frozen=True, slots=True)
class FlightDataAnalysisConfiguration:
    """Alignment, cleaning, identification, and acceptance settings."""

    marker_threshold: float
    maximum_interpolation_gap_s: float
    hampel_half_window: int
    hampel_threshold_sigma: float
    derivative_window: int
    derivative_polynomial_order: int
    huber_threshold_sigma: float
    huber_maximum_iterations: int
    validation_fraction: float
    clock_offset_tolerance_s: float
    clock_drift_tolerance_ppm: float
    parameter_relative_tolerance: float
    validation_pitch_rms_limit_deg: float
    validation_rate_rms_limit_degps: float
    maximum_residual_autocorrelation: float


@dataclass(frozen=True, slots=True)
class FlightDataIdentificationConfiguration:
    """Complete public-safe synthetic flight-data workflow."""

    source_path: Path
    name: str
    safety_scope: str
    output_directory: Path
    duration_s: float
    integration_step_s: float
    command_sample_rate_hz: float
    sensor_sample_rate_hz: float
    random_seed: int
    plant: IdentificationPlantConfiguration
    excitation: ExcitationConfiguration
    clock: ClockConfiguration
    sensor: FlightDataSensorConfiguration
    outliers: tuple[SyntheticFlightDataOutlier, ...]
    analysis: FlightDataAnalysisConfiguration


def _dropouts(value: object, context: str) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for index, item in enumerate(_sequence(value, context)):
        pair = _number_tuple(item, f"{context}[{index}]", length=2)
        if pair[0] < 0.0 or pair[1] <= pair[0]:
            raise ConfigurationError(f"{context}[{index}]: expected 0 <= start < end")
        intervals.append((pair[0], pair[1]))
    return tuple(intervals)


def load_flight_data_identification_configuration(
    path: str | Path,
) -> FlightDataIdentificationConfiguration:
    """Load a synthetic asynchronous flight-data identification case."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "flight_data_identification",
        required={
            "metadata",
            "simulation",
            "plant",
            "excitation",
            "clock",
            "sensors",
            "outliers",
            "analysis",
        },
    )
    metadata = _mapping(root["metadata"], "flight_data_identification.metadata")
    _keys(
        metadata,
        "flight_data_identification.metadata",
        required={"name", "safety_scope", "fictional"},
    )
    if metadata["fictional"] is not True:
        raise ConfigurationError("flight_data_identification.metadata.fictional: must be true")
    safety_scope = _string(
        metadata["safety_scope"], "flight_data_identification.metadata.safety_scope"
    )
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        raise ConfigurationError(
            "flight_data_identification safety_scope must state fictional and civilian"
        )

    simulation = _mapping(root["simulation"], "flight_data_identification.simulation")
    _keys(
        simulation,
        "flight_data_identification.simulation",
        required={
            "output_directory",
            "duration_s",
            "integration_step_s",
            "command_sample_rate_hz",
            "sensor_sample_rate_hz",
            "random_seed",
        },
    )
    duration_s = _number(
        simulation["duration_s"],
        "flight_data_identification.simulation.duration_s",
        positive=True,
    )
    integration_step_s = _number(
        simulation["integration_step_s"],
        "flight_data_identification.simulation.integration_step_s",
        positive=True,
    )
    command_rate_hz = _number(
        simulation["command_sample_rate_hz"],
        "flight_data_identification.simulation.command_sample_rate_hz",
        positive=True,
    )
    sensor_rate_hz = _number(
        simulation["sensor_sample_rate_hz"],
        "flight_data_identification.simulation.sensor_sample_rate_hz",
        positive=True,
    )
    if integration_step_s > 0.25 / max(command_rate_hz, sensor_rate_hz):
        raise ConfigurationError(
            "flight_data_identification.simulation.integration_step_s: "
            "must resolve each log interval with at least four steps"
        )

    plant_data = _mapping(root["plant"], "flight_data_identification.plant")
    _keys(
        plant_data,
        "flight_data_identification.plant",
        required={
            "inertia_kgm2",
            "damping_nms_per_rad",
            "stiffness_nm_per_rad",
            "disturbance_moment_nm",
        },
    )
    plant = IdentificationPlantConfiguration(
        inertia_kgm2=_number(
            plant_data["inertia_kgm2"],
            "flight_data_identification.plant.inertia_kgm2",
            positive=True,
        ),
        damping_nms_per_rad=_number(
            plant_data["damping_nms_per_rad"],
            "flight_data_identification.plant.damping_nms_per_rad",
            nonnegative=True,
        ),
        stiffness_nm_per_rad=_number(
            plant_data["stiffness_nm_per_rad"],
            "flight_data_identification.plant.stiffness_nm_per_rad",
            nonnegative=True,
        ),
        disturbance_moment_nm=_number(
            plant_data["disturbance_moment_nm"],
            "flight_data_identification.plant.disturbance_moment_nm",
        ),
    )

    excitation_data = _mapping(root["excitation"], "flight_data_identification.excitation")
    _keys(
        excitation_data,
        "flight_data_identification.excitation",
        required={"frequencies_hz", "amplitudes_nm", "phases_deg", "limit_nm"},
    )
    frequencies = _number_tuple(
        excitation_data["frequencies_hz"],
        "flight_data_identification.excitation.frequencies_hz",
    )
    amplitudes = _number_tuple(
        excitation_data["amplitudes_nm"],
        "flight_data_identification.excitation.amplitudes_nm",
    )
    phases_deg = _number_tuple(
        excitation_data["phases_deg"],
        "flight_data_identification.excitation.phases_deg",
    )
    lengths_match = len(frequencies) == len(amplitudes) == len(phases_deg)
    if not frequencies or not lengths_match:
        raise ConfigurationError(
            "flight_data_identification.excitation: frequency/amplitude/phase lengths must match"
        )
    if any(value <= 0.0 for value in frequencies) or any(value <= 0.0 for value in amplitudes):
        raise ConfigurationError(
            "flight_data_identification.excitation: frequencies and amplitudes must be positive"
        )
    if max(frequencies) >= 0.4 * min(command_rate_hz, sensor_rate_hz):
        raise ConfigurationError(
            "flight_data_identification.excitation: frequencies exceed the sampled-data domain"
        )
    excitation = ExcitationConfiguration(
        frequencies_hz=frequencies,
        amplitudes_nm=amplitudes,
        phases_rad=tuple(float(value) for value in np.deg2rad(phases_deg)),
        limit_nm=_number(
            excitation_data["limit_nm"],
            "flight_data_identification.excitation.limit_nm",
            positive=True,
        ),
    )

    clock_data = _mapping(root["clock"], "flight_data_identification.clock")
    _keys(
        clock_data,
        "flight_data_identification.clock",
        required={"sensor_offset_s", "sensor_drift_ppm", "marker_times_s", "marker_width_s"},
    )
    marker_times = _number_tuple(
        clock_data["marker_times_s"],
        "flight_data_identification.clock.marker_times_s",
    )
    if len(marker_times) < 3 or any(np.diff(marker_times) <= 0.0):
        raise ConfigurationError(
            "flight_data_identification.clock.marker_times_s: require 3 increasing markers"
        )
    if marker_times[0] <= 0.0 or marker_times[-1] >= duration_s:
        raise ConfigurationError(
            "flight_data_identification.clock.marker_times_s: markers must be inside the record"
        )
    clock = ClockConfiguration(
        sensor_offset_s=_number(
            clock_data["sensor_offset_s"],
            "flight_data_identification.clock.sensor_offset_s",
        ),
        sensor_drift_ppm=_number(
            clock_data["sensor_drift_ppm"],
            "flight_data_identification.clock.sensor_drift_ppm",
        ),
        marker_times_s=marker_times,
        marker_width_s=_number(
            clock_data["marker_width_s"],
            "flight_data_identification.clock.marker_width_s",
            positive=True,
        ),
    )

    sensors = _mapping(root["sensors"], "flight_data_identification.sensors")
    _keys(
        sensors,
        "flight_data_identification.sensors",
        required={
            "pitch_noise_std_deg",
            "rate_noise_std_degps",
            "pitch_quantisation_deg",
            "rate_quantisation_degps",
            "sync_noise_std",
            "dropout_intervals_s",
        },
    )
    sensor = FlightDataSensorConfiguration(
        pitch_noise_std_rad=float(
            np.deg2rad(
                _number(
                    sensors["pitch_noise_std_deg"],
                    "flight_data_identification.sensors.pitch_noise_std_deg",
                    nonnegative=True,
                )
            )
        ),
        rate_noise_std_radps=float(
            np.deg2rad(
                _number(
                    sensors["rate_noise_std_degps"],
                    "flight_data_identification.sensors.rate_noise_std_degps",
                    nonnegative=True,
                )
            )
        ),
        pitch_quantisation_rad=float(
            np.deg2rad(
                _number(
                    sensors["pitch_quantisation_deg"],
                    "flight_data_identification.sensors.pitch_quantisation_deg",
                    nonnegative=True,
                )
            )
        ),
        rate_quantisation_radps=float(
            np.deg2rad(
                _number(
                    sensors["rate_quantisation_degps"],
                    "flight_data_identification.sensors.rate_quantisation_degps",
                    nonnegative=True,
                )
            )
        ),
        sync_noise_std=_number(
            sensors["sync_noise_std"],
            "flight_data_identification.sensors.sync_noise_std",
            nonnegative=True,
        ),
        dropout_intervals_s=_dropouts(
            sensors["dropout_intervals_s"],
            "flight_data_identification.sensors.dropout_intervals_s",
        ),
    )

    outliers: list[SyntheticFlightDataOutlier] = []
    outlier_rows = _sequence(root["outliers"], "flight_data_identification.outliers")
    for index, item in enumerate(outlier_rows):
        context = f"flight_data_identification.outliers[{index}]"
        data = _mapping(item, context)
        _keys(data, context, required={"time_s", "pitch_offset_deg", "rate_offset_degps"})
        time_s = _number(data["time_s"], f"{context}.time_s", nonnegative=True)
        if time_s >= duration_s:
            raise ConfigurationError(f"{context}.time_s: must be inside the record")
        outliers.append(
            SyntheticFlightDataOutlier(
                time_s=time_s,
                pitch_offset_rad=float(
                    np.deg2rad(_number(data["pitch_offset_deg"], f"{context}.pitch_offset_deg"))
                ),
                rate_offset_radps=float(
                    np.deg2rad(_number(data["rate_offset_degps"], f"{context}.rate_offset_degps"))
                ),
            )
        )

    analysis_data = _mapping(root["analysis"], "flight_data_identification.analysis")
    _keys(
        analysis_data,
        "flight_data_identification.analysis",
        required={
            "marker_threshold",
            "maximum_interpolation_gap_s",
            "hampel_half_window",
            "hampel_threshold_sigma",
            "derivative_window",
            "derivative_polynomial_order",
            "huber_threshold_sigma",
            "huber_maximum_iterations",
            "validation_fraction",
            "clock_offset_tolerance_s",
            "clock_drift_tolerance_ppm",
            "parameter_relative_tolerance",
            "validation_pitch_rms_limit_deg",
            "validation_rate_rms_limit_degps",
            "maximum_residual_autocorrelation",
        },
    )
    derivative_window = _integer(
        analysis_data["derivative_window"],
        "flight_data_identification.analysis.derivative_window",
        nonnegative=True,
    )
    derivative_order = _integer(
        analysis_data["derivative_polynomial_order"],
        "flight_data_identification.analysis.derivative_polynomial_order",
        nonnegative=True,
    )
    if derivative_window < 5 or derivative_window % 2 == 0 or derivative_order >= derivative_window:
        raise ConfigurationError(
            "flight_data_identification.analysis: derivative window must be odd and exceed order"
        )
    validation_fraction = _number(
        analysis_data["validation_fraction"],
        "flight_data_identification.analysis.validation_fraction",
        positive=True,
    )
    if validation_fraction >= 0.5:
        raise ConfigurationError(
            "flight_data_identification.analysis.validation_fraction: must be below 0.5"
        )
    analysis = FlightDataAnalysisConfiguration(
        marker_threshold=_number(
            analysis_data["marker_threshold"],
            "flight_data_identification.analysis.marker_threshold",
            positive=True,
        ),
        maximum_interpolation_gap_s=_number(
            analysis_data["maximum_interpolation_gap_s"],
            "flight_data_identification.analysis.maximum_interpolation_gap_s",
            positive=True,
        ),
        hampel_half_window=_integer(
            analysis_data["hampel_half_window"],
            "flight_data_identification.analysis.hampel_half_window",
            nonnegative=True,
        ),
        hampel_threshold_sigma=_number(
            analysis_data["hampel_threshold_sigma"],
            "flight_data_identification.analysis.hampel_threshold_sigma",
            positive=True,
        ),
        derivative_window=derivative_window,
        derivative_polynomial_order=derivative_order,
        huber_threshold_sigma=_number(
            analysis_data["huber_threshold_sigma"],
            "flight_data_identification.analysis.huber_threshold_sigma",
            positive=True,
        ),
        huber_maximum_iterations=_integer(
            analysis_data["huber_maximum_iterations"],
            "flight_data_identification.analysis.huber_maximum_iterations",
            nonnegative=True,
        ),
        validation_fraction=validation_fraction,
        clock_offset_tolerance_s=_number(
            analysis_data["clock_offset_tolerance_s"],
            "flight_data_identification.analysis.clock_offset_tolerance_s",
            positive=True,
        ),
        clock_drift_tolerance_ppm=_number(
            analysis_data["clock_drift_tolerance_ppm"],
            "flight_data_identification.analysis.clock_drift_tolerance_ppm",
            positive=True,
        ),
        parameter_relative_tolerance=_number(
            analysis_data["parameter_relative_tolerance"],
            "flight_data_identification.analysis.parameter_relative_tolerance",
            positive=True,
        ),
        validation_pitch_rms_limit_deg=_number(
            analysis_data["validation_pitch_rms_limit_deg"],
            "flight_data_identification.analysis.validation_pitch_rms_limit_deg",
            positive=True,
        ),
        validation_rate_rms_limit_degps=_number(
            analysis_data["validation_rate_rms_limit_degps"],
            "flight_data_identification.analysis.validation_rate_rms_limit_degps",
            positive=True,
        ),
        maximum_residual_autocorrelation=_number(
            analysis_data["maximum_residual_autocorrelation"],
            "flight_data_identification.analysis.maximum_residual_autocorrelation",
            positive=True,
        ),
    )
    return FlightDataIdentificationConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "flight_data_identification.metadata.name"),
        safety_scope=safety_scope,
        output_directory=Path(
            _string(
                simulation["output_directory"],
                "flight_data_identification.simulation.output_directory",
            )
        ),
        duration_s=duration_s,
        integration_step_s=integration_step_s,
        command_sample_rate_hz=command_rate_hz,
        sensor_sample_rate_hz=sensor_rate_hz,
        random_seed=_integer(
            simulation["random_seed"],
            "flight_data_identification.simulation.random_seed",
            nonnegative=True,
        ),
        plant=plant,
        excitation=excitation,
        clock=clock,
        sensor=sensor,
        outliers=tuple(outliers),
        analysis=analysis,
    )
