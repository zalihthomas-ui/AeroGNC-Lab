"""Synthetic truth/measurement/estimate vertical-navigation workflow."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.navigation_loader import NavigationDemoConfiguration
from aerognc.gnc.ekf import VerticalNavigationEKF
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.logging import SimulationResult
from aerognc.simulation.simulator import simulate_three_dof
from aerognc.vehicle.sensors import AccelerometerSensor, BarometricAltimeter, CivilianGnssSensor


@dataclass(frozen=True, slots=True)
class NavigationDemoResult:
    """Truth, sparse measurements, estimate, covariance, and RMS evidence."""

    time_s: FloatArray
    true_altitude_m: FloatArray
    true_vertical_velocity_up_mps: FloatArray
    measured_acceleration_up_mps2: FloatArray
    measured_barometric_altitude_m: FloatArray
    measured_gnss_altitude_m: FloatArray
    measured_gnss_vertical_velocity_up_mps: FloatArray
    estimated_altitude_m: FloatArray
    estimated_vertical_velocity_up_mps: FloatArray
    estimated_accelerometer_bias_mps2: FloatArray
    altitude_sigma_m: FloatArray
    velocity_sigma_mps: FloatArray
    raw_barometer_rms_m: float
    estimated_altitude_rms_m: float


def run_navigation_demo(
    configuration: NavigationDemoConfiguration,
    *,
    truth_override: SimulationResult | None = None,
) -> NavigationDemoResult:
    """Simulate truth, delayed/dropout sensors, and the scoped vertical filter."""
    truth = simulate_three_dof(configuration.base) if truth_override is None else truth_override
    time_s = truth.time_s
    count = time_s.size
    true_altitude = truth.columns["altitude_m"]
    true_velocity_up = truth.columns["vertical_velocity_up_mps"]
    true_acceleration_up = -truth.columns["acceleration_down_mps2"]
    accelerometer = AccelerometerSensor(
        configuration.accelerometer, seed=configuration.random_seed + 1
    )
    barometer = BarometricAltimeter(configuration.barometer, seed=configuration.random_seed + 2)
    gnss = CivilianGnssSensor(configuration.gnss, seed=configuration.random_seed + 3)
    filter_instance = VerticalNavigationEKF(
        configuration.initial_state,
        configuration.initial_covariance,
        configuration.filter_tuning,
    )
    measured_acceleration = np.full(count, np.nan)
    measured_baro = np.full(count, np.nan)
    measured_gnss_altitude = np.full(count, np.nan)
    measured_gnss_velocity = np.full(count, np.nan)
    estimated_altitude = np.empty(count)
    estimated_velocity = np.empty(count)
    estimated_bias = np.empty(count)
    altitude_sigma = np.empty(count)
    velocity_sigma = np.empty(count)
    held_acceleration = float(true_acceleration_up[0])

    def sample_index(sample_time_s: float) -> int:
        candidate = int(np.searchsorted(time_s, sample_time_s, side="left"))
        if candidate == 0:
            return 0
        if candidate >= count:
            return count - 1
        return (
            candidate - 1
            if abs(time_s[candidate - 1] - sample_time_s) <= abs(time_s[candidate] - sample_time_s)
            else candidate
        )

    for index, current_time_s in enumerate(time_s):
        if index > 0:
            filter_instance.predict(held_acceleration, float(time_s[index] - time_s[index - 1]))
        acceleration_measurement = accelerometer.measure(
            float(current_time_s), [0.0, 0.0, true_acceleration_up[index]]
        )
        if acceleration_measurement is not None:
            held_acceleration = float(acceleration_measurement.value[2])
            measured_acceleration[sample_index(acceleration_measurement.sample_time_s)] = (
                held_acceleration
            )
        barometer_measurement = barometer.measure(float(current_time_s), [true_altitude[index]])
        if barometer_measurement is not None:
            acquisition_index = sample_index(barometer_measurement.sample_time_s)
            measured_baro[acquisition_index] = barometer_measurement.value[0]
            measurement_age_s = float(current_time_s - barometer_measurement.sample_time_s)
            compensated_altitude = (
                barometer_measurement.value[0]
                + filter_instance.state[1] * measurement_age_s
                + 0.5 * (held_acceleration - filter_instance.state[2]) * measurement_age_s**2
            )
            filter_instance.update_barometer(float(compensated_altitude))
        gnss_truth = np.array(
            [
                truth.columns["north_m"][index],
                truth.columns["east_m"][index],
                truth.columns["down_m"][index],
                truth.columns["velocity_north_mps"][index],
                truth.columns["velocity_east_mps"][index],
                truth.columns["velocity_down_mps"][index],
            ]
        )
        gnss_measurement = gnss.measure(float(current_time_s), gnss_truth)
        if gnss_measurement is not None:
            acquisition_index = sample_index(gnss_measurement.sample_time_s)
            measured_gnss_altitude[acquisition_index] = -gnss_measurement.value[2]
            measured_gnss_velocity[acquisition_index] = -gnss_measurement.value[5]
            measurement_age_s = float(current_time_s - gnss_measurement.sample_time_s)
            compensated_velocity = (
                measured_gnss_velocity[acquisition_index]
                + (held_acceleration - filter_instance.state[2]) * measurement_age_s
            )
            compensated_altitude = (
                measured_gnss_altitude[acquisition_index]
                + measured_gnss_velocity[acquisition_index] * measurement_age_s
                + 0.5 * (held_acceleration - filter_instance.state[2]) * measurement_age_s**2
            )
            filter_instance.update_gnss(
                float(compensated_altitude),
                float(compensated_velocity),
            )
        estimated_altitude[index] = filter_instance.state[0]
        estimated_velocity[index] = filter_instance.state[1]
        estimated_bias[index] = filter_instance.state[2]
        altitude_sigma[index] = np.sqrt(filter_instance.covariance[0, 0])
        velocity_sigma[index] = np.sqrt(filter_instance.covariance[1, 1])

    barometer_mask = np.isfinite(measured_baro)
    raw_rms = float(
        np.sqrt(np.mean((measured_baro[barometer_mask] - true_altitude[barometer_mask]) ** 2))
    )
    estimated_rms = float(np.sqrt(np.mean((estimated_altitude - true_altitude) ** 2)))
    return NavigationDemoResult(
        time_s=time_s.copy(),
        true_altitude_m=true_altitude.copy(),
        true_vertical_velocity_up_mps=true_velocity_up.copy(),
        measured_acceleration_up_mps2=measured_acceleration,
        measured_barometric_altitude_m=measured_baro,
        measured_gnss_altitude_m=measured_gnss_altitude,
        measured_gnss_vertical_velocity_up_mps=measured_gnss_velocity,
        estimated_altitude_m=estimated_altitude,
        estimated_vertical_velocity_up_mps=estimated_velocity,
        estimated_accelerometer_bias_mps2=estimated_bias,
        altitude_sigma_m=altitude_sigma,
        velocity_sigma_mps=velocity_sigma,
        raw_barometer_rms_m=raw_rms,
        estimated_altitude_rms_m=estimated_rms,
    )


def write_navigation_demo(
    result: NavigationDemoResult, output_directory: str | Path
) -> tuple[Path, Path]:
    """Write a documented CSV and compact navigation summary."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "navigation_demo.csv"
    columns = {
        "time_s": result.time_s,
        "true_altitude_m": result.true_altitude_m,
        "true_vertical_velocity_up_mps": result.true_vertical_velocity_up_mps,
        "measured_acceleration_up_mps2": result.measured_acceleration_up_mps2,
        "measured_barometric_altitude_m": result.measured_barometric_altitude_m,
        "measured_gnss_altitude_m": result.measured_gnss_altitude_m,
        "measured_gnss_vertical_velocity_up_mps": result.measured_gnss_vertical_velocity_up_mps,
        "estimated_altitude_m": result.estimated_altitude_m,
        "estimated_vertical_velocity_up_mps": result.estimated_vertical_velocity_up_mps,
        "estimated_accelerometer_bias_mps2": result.estimated_accelerometer_bias_mps2,
        "altitude_1sigma_m": result.altitude_sigma_m,
        "velocity_1sigma_mps": result.velocity_sigma_mps,
    }
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns.keys())
        for row in zip(*columns.values(), strict=True):
            writer.writerow(["" if not np.isfinite(value) else f"{value:.10g}" for value in row])
    summary_path = output / "navigation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "estimated_altitude_rms_m": result.estimated_altitude_rms_m,
                "raw_barometer_rms_m": result.raw_barometer_rms_m,
                "improvement_percent": 100.0
                * (1.0 - result.estimated_altitude_rms_m / result.raw_barometer_rms_m),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path
