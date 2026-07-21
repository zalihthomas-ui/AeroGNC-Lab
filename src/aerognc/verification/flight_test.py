"""Synthetic flight-test data generation, reload, event detection, and evaluation."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.navigation_loader import NavigationDemoConfiguration
from aerognc.gnc.ekf import VerticalNavigationEKF
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.logging import SimulationResult
from aerognc.simulation.navigation_demo import NavigationDemoResult, run_navigation_demo
from aerognc.simulation.simulator import simulate_three_dof


@dataclass(frozen=True, slots=True)
class FlightTestAnalysis:
    """Reloaded signals, reconstructed states, and detected event times."""

    time_s: FloatArray
    measured_acceleration_up_mps2: FloatArray
    measured_barometric_altitude_m: FloatArray
    measured_gnss_altitude_m: FloatArray
    measured_gnss_vertical_velocity_up_mps: FloatArray
    reconstructed_altitude_m: FloatArray
    reconstructed_vertical_velocity_up_mps: FloatArray
    burnout_time_s: float
    apogee_time_s: float
    ground_impact_time_s: float
    reconstructed_apogee_m: float
    maximum_vertical_velocity_up_mps: float


@dataclass(frozen=True, slots=True)
class FlightTestWorkflowResult:
    """Expected truth, reloaded analysis, and comparison errors."""

    truth: SimulationResult
    analysis: FlightTestAnalysis
    measurement_csv: Path
    summary_json: Path
    event_time_errors_s: dict[str, float]
    apogee_error_m: float


def write_synthetic_measurement_csv(
    navigation: NavigationDemoResult,
    path: str | Path,
) -> Path:
    """Write measurements only; blanks explicitly represent unavailable samples."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "time_s": navigation.time_s,
        "accelerometer_up_mps2": navigation.measured_acceleration_up_mps2,
        "barometric_altitude_m": navigation.measured_barometric_altitude_m,
        "gnss_altitude_m": navigation.measured_gnss_altitude_m,
        "gnss_vertical_velocity_up_mps": navigation.measured_gnss_vertical_velocity_up_mps,
    }
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns.keys())
        for row in zip(*columns.values(), strict=True):
            writer.writerow(["" if not np.isfinite(value) else f"{value:.10g}" for value in row])
    return output


def _load_measurement_csv(path: str | Path) -> dict[str, FloatArray]:
    input_path = Path(path)
    required = (
        "time_s",
        "accelerometer_up_mps2",
        "barometric_altitude_m",
        "gnss_altitude_m",
        "gnss_vertical_velocity_up_mps",
    )
    values: dict[str, list[float]] = {name: [] for name in required}
    with input_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(required):
            raise ValueError(f"flight CSV columns must be exactly {required}")
        for row_number, row in enumerate(reader, start=2):
            for name in required:
                text = row[name]
                try:
                    values[name].append(np.nan if text == "" else float(text))
                except ValueError as error:
                    raise ValueError(
                        f"invalid numeric value in {name} at CSV row {row_number}"
                    ) from error
    result = {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}
    time_s = result["time_s"]
    if time_s.size < 2 or not np.all(np.isfinite(time_s)) or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("flight CSV time_s must be finite and strictly increasing")
    return result


def analyse_synthetic_flight_csv(
    path: str | Path,
    configuration: NavigationDemoConfiguration,
) -> FlightTestAnalysis:
    """Reload a measurement-only CSV and reconstruct vertical performance/events."""
    data = _load_measurement_csv(path)
    time_s = data["time_s"]
    filter_instance = VerticalNavigationEKF(
        configuration.initial_state,
        configuration.initial_covariance,
        configuration.filter_tuning,
    )
    reconstructed_altitude = np.empty_like(time_s)
    reconstructed_velocity = np.empty_like(time_s)
    held_acceleration = 0.0
    for index, current_time_s in enumerate(time_s):
        if index > 0:
            filter_instance.predict(held_acceleration, float(current_time_s - time_s[index - 1]))
        if np.isfinite(data["accelerometer_up_mps2"][index]):
            held_acceleration = float(data["accelerometer_up_mps2"][index])
        if np.isfinite(data["barometric_altitude_m"][index]):
            filter_instance.update_barometer(float(data["barometric_altitude_m"][index]))
        if np.isfinite(data["gnss_altitude_m"][index]) and np.isfinite(
            data["gnss_vertical_velocity_up_mps"][index]
        ):
            filter_instance.update_gnss(
                float(data["gnss_altitude_m"][index]),
                float(data["gnss_vertical_velocity_up_mps"][index]),
            )
        reconstructed_altitude[index] = filter_instance.state[0]
        reconstructed_velocity[index] = filter_instance.state[1]

    acceleration_indices = np.flatnonzero(
        np.isfinite(data["accelerometer_up_mps2"]) & (time_s >= 0.5) & (time_s <= 8.0)
    )
    if acceleration_indices.size < 2:
        raise ValueError("insufficient accelerometer samples for burnout detection")
    acceleration_values = data["accelerometer_up_mps2"][acceleration_indices]
    acceleration_drop = np.diff(acceleration_values)
    burnout_pair = int(np.argmin(acceleration_drop))
    burnout_time_s = float(
        0.5
        * (
            time_s[acceleration_indices[burnout_pair]]
            + time_s[acceleration_indices[burnout_pair + 1]]
        )
    )

    crossing_indices = np.flatnonzero(
        (time_s[:-1] > burnout_time_s)
        & (reconstructed_velocity[:-1] > 0.0)
        & (reconstructed_velocity[1:] <= 0.0)
    )
    if crossing_indices.size == 0:
        raise ValueError("apogee crossing not found in reconstructed vertical velocity")
    crossing = int(crossing_indices[0])
    denominator = reconstructed_velocity[crossing] - reconstructed_velocity[crossing + 1]
    fraction = reconstructed_velocity[crossing] / denominator
    apogee_time_s = float(time_s[crossing] + fraction * (time_s[crossing + 1] - time_s[crossing]))
    return FlightTestAnalysis(
        time_s=time_s,
        measured_acceleration_up_mps2=data["accelerometer_up_mps2"],
        measured_barometric_altitude_m=data["barometric_altitude_m"],
        measured_gnss_altitude_m=data["gnss_altitude_m"],
        measured_gnss_vertical_velocity_up_mps=data["gnss_vertical_velocity_up_mps"],
        reconstructed_altitude_m=reconstructed_altitude,
        reconstructed_vertical_velocity_up_mps=reconstructed_velocity,
        burnout_time_s=burnout_time_s,
        apogee_time_s=apogee_time_s,
        ground_impact_time_s=float(time_s[-1]),
        reconstructed_apogee_m=float(np.max(reconstructed_altitude)),
        maximum_vertical_velocity_up_mps=float(np.max(reconstructed_velocity)),
    )


def run_synthetic_flight_test_workflow(
    configuration: NavigationDemoConfiguration,
    output_directory: str | Path,
) -> FlightTestWorkflowResult:
    """Generate truth/measurements, reload measurements, analyse, compare, and summarise."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    truth = simulate_three_dof(configuration.base)
    navigation = run_navigation_demo(configuration, truth_override=truth)
    measurement_csv = write_synthetic_measurement_csv(
        navigation, output / "synthetic_flight_measurements.csv"
    )
    analysis = analyse_synthetic_flight_csv(measurement_csv, configuration)
    expected_events = {event.name: event.time_s for event in truth.events}
    detected_events = {
        "burnout": analysis.burnout_time_s,
        "apogee": analysis.apogee_time_s,
        "ground_impact": analysis.ground_impact_time_s,
    }
    event_errors = {name: detected_events[name] - expected_events[name] for name in detected_events}
    expected_apogee_m = float(np.max(truth.columns["altitude_m"]))
    apogee_error_m = analysis.reconstructed_apogee_m - expected_apogee_m
    summary_path = output / "flight_test_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "data_scope": "synthetic fictional civilian research rocket",
                "expected": {
                    "events_s": expected_events,
                    "apogee_m": expected_apogee_m,
                    "maximum_vertical_velocity_up_mps": float(
                        np.max(truth.columns["vertical_velocity_up_mps"])
                    ),
                },
                "reconstructed": {
                    "events_s": detected_events,
                    "apogee_m": analysis.reconstructed_apogee_m,
                    "maximum_vertical_velocity_up_mps": analysis.maximum_vertical_velocity_up_mps,
                },
                "errors": {
                    "event_time_s": event_errors,
                    "apogee_m": apogee_error_m,
                },
                "measurement_availability_fraction": {
                    "accelerometer": float(
                        np.mean(np.isfinite(analysis.measured_acceleration_up_mps2))
                    ),
                    "barometer": float(
                        np.mean(np.isfinite(analysis.measured_barometric_altitude_m))
                    ),
                    "gnss": float(np.mean(np.isfinite(analysis.measured_gnss_altitude_m))),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return FlightTestWorkflowResult(
        truth=truth,
        analysis=analysis,
        measurement_csv=measurement_csv,
        summary_json=summary_path,
        event_time_errors_s=event_errors,
        apogee_error_m=apogee_error_m,
    )
