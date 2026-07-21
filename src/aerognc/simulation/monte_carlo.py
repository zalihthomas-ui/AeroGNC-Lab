"""Reproducible coupled flight/navigation/control Monte Carlo verification."""

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerognc.configuration import (
    load_attitude_control_configuration,
    load_navigation_demo_configuration,
    load_three_dof_configuration,
)
from aerognc.configuration.models import EnvironmentDefinition, VehicleDefinition
from aerognc.configuration.monte_carlo_loader import (
    MonteCarloConfiguration,
    MonteCarloDispersions,
    MonteCarloRequirements,
)
from aerognc.environment.wind import WindModel, WindProfile
from aerognc.gnc.pid import PIDGains
from aerognc.simulation.attitude_control import simulate_attitude_control
from aerognc.simulation.navigation_demo import run_navigation_demo
from aerognc.simulation.simulator import simulate_three_dof
from aerognc.vehicle.aerodynamics import (
    AerodynamicCoefficientProvider,
    AerodynamicCoefficients,
    AerodynamicModel,
)
from aerognc.vehicle.mass_properties import MassPropertiesModel
from aerognc.vehicle.propulsion import ThrustCurve
from aerognc.vehicle.sensors import SensorErrorParameters


@dataclass(frozen=True, slots=True)
class DispersionSample:
    """All independent sampled variables for one run."""

    run_index: int
    random_seed: int
    initial_speed_offset_mps: float
    initial_elevation_offset_deg: float
    vehicle_mass_scale: float
    thrust_scale: float
    thrust_misalignment_pitch_deg: float
    thrust_misalignment_yaw_deg: float
    aerodynamic_scale: float
    wind_scale: float
    sensor_noise_scale: float
    sensor_bias_scale: float
    actuator_delay_scale: float
    controller_gain_scale: float

    def parameter_dict(self) -> dict[str, float]:
        """Return sampled numeric inputs excluding bookkeeping identifiers."""
        values = asdict(self)
        values.pop("run_index")
        values.pop("random_seed")
        return {name: float(value) for name, value in values.items()}


@dataclass(frozen=True, slots=True)
class _ScaledDragCoefficientProvider:
    """Apply one Monte Carlo drag multiplier to any coefficient provider."""

    base: AerodynamicCoefficientProvider
    drag_scale: float

    def coefficients(
        self,
        mach: float,
        alpha_rad: float,
        beta_rad: float,
        nondimensional_rates: npt.ArrayLike = (0.0, 0.0, 0.0),
        control_coefficients: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> AerodynamicCoefficients:
        """Return base coefficients with the sampled drag scale applied."""
        values = self.base.coefficients(
            mach,
            alpha_rad,
            beta_rad,
            nondimensional_rates,
            control_coefficients,
        )
        return AerodynamicCoefficients(
            drag=values.drag * self.drag_scale,
            side=values.side,
            normal=values.normal,
            roll=values.roll,
            pitch=values.pitch,
            yaw=values.yaw,
        )


@dataclass(frozen=True, slots=True)
class MonteCarloRunResult:
    """One successful or gracefully failed coupled run."""

    run_index: int
    random_seed: int
    sample: dict[str, float]
    success: bool
    error: str | None
    metrics: dict[str, float]
    requirement_margins: dict[str, float]
    requirement_pass: dict[str, bool]


@dataclass(frozen=True, slots=True)
class MonteCarloSummary:
    """Ordered run records and aggregate evidence."""

    name: str
    master_seed: int
    runs: tuple[MonteCarloRunResult, ...]
    statistics: dict[str, dict[str, float]]
    correlations: dict[str, float]
    worst_case_runs: dict[str, int]
    requirement_pass_rates: dict[str, float]

    @property
    def successful_count(self) -> int:
        """Number of successful members."""
        return sum(run.success for run in self.runs)

    @property
    def failed_count(self) -> int:
        """Number of failed members."""
        return len(self.runs) - self.successful_count


def _positive_scale(generator: np.random.Generator, standard_deviation: float) -> float:
    return float(np.clip(generator.normal(1.0, standard_deviation), 0.25, 3.0))


def generate_dispersion_samples(
    sample_count: int,
    master_seed: int,
    dispersions: MonteCarloDispersions,
) -> tuple[DispersionSample, ...]:
    """Generate an ordered seed-tree ensemble independent of worker scheduling."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    children = np.random.SeedSequence(master_seed).spawn(sample_count)
    samples: list[DispersionSample] = []
    for run_index, child in enumerate(children):
        generator = np.random.default_rng(child)
        random_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        samples.append(
            DispersionSample(
                run_index=run_index,
                random_seed=random_seed,
                initial_speed_offset_mps=float(
                    generator.normal(0.0, dispersions.initial_speed_std_mps)
                ),
                initial_elevation_offset_deg=float(
                    generator.normal(0.0, dispersions.initial_elevation_std_deg)
                ),
                vehicle_mass_scale=_positive_scale(generator, dispersions.vehicle_mass_scale_std),
                thrust_scale=_positive_scale(generator, dispersions.thrust_scale_std),
                thrust_misalignment_pitch_deg=float(
                    generator.normal(0.0, dispersions.thrust_misalignment_std_deg)
                ),
                thrust_misalignment_yaw_deg=float(
                    generator.normal(0.0, dispersions.thrust_misalignment_std_deg)
                ),
                aerodynamic_scale=_positive_scale(generator, dispersions.aerodynamic_scale_std),
                wind_scale=_positive_scale(generator, dispersions.wind_scale_std),
                sensor_noise_scale=_positive_scale(generator, dispersions.sensor_noise_scale_std),
                sensor_bias_scale=_positive_scale(generator, dispersions.sensor_bias_scale_std),
                actuator_delay_scale=_positive_scale(
                    generator, dispersions.actuator_delay_scale_std
                ),
                controller_gain_scale=_positive_scale(
                    generator, dispersions.controller_gain_scale_std
                ),
            )
        )
    return tuple(samples)


def _perturbed_vehicle(base: VehicleDefinition, sample: DispersionSample) -> VehicleDefinition:
    propulsion = ThrustCurve(
        base.propulsion.time_s,
        base.propulsion.thrust_n * sample.thrust_scale,
        base.propulsion.propellant_mass_kg * sample.vehicle_mass_scale,
    )
    mass_properties = MassPropertiesModel(
        base.mass_properties.dry_mass_kg * sample.vehicle_mass_scale,
        propulsion,
        base.mass_properties.dry_cg_from_nose_m,
        base.mass_properties.wet_cg_from_nose_m,
        base.mass_properties.dry_inertia_body_kgm2 * sample.vehicle_mass_scale,
        base.mass_properties.wet_inertia_body_kgm2 * sample.vehicle_mass_scale,
    )
    source_aero = base.aerodynamics
    if source_aero.coefficient_provider is not None:
        aerodynamics = AerodynamicModel(
            reference_area_m2=source_aero.reference_area_m2,
            reference_length_m=source_aero.reference_length_m,
            coefficient_provider=_ScaledDragCoefficientProvider(
                source_aero.coefficient_provider,
                sample.aerodynamic_scale,
            ),
        )
    else:
        if source_aero.drag_table is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("legacy aerodynamic model has no drag table")
        aerodynamics = AerodynamicModel(
            reference_area_m2=source_aero.reference_area_m2,
            reference_length_m=source_aero.reference_length_m,
            mach_points=source_aero.drag_table.x,
            drag_coefficients=source_aero.drag_table.y * sample.aerodynamic_scale,
            out_of_range=source_aero.drag_table.out_of_range,
            drag_alpha2_per_rad2=(source_aero.drag_alpha2_per_rad2 * sample.aerodynamic_scale),
            side_beta_per_rad=source_aero.side_beta_per_rad,
            normal_alpha_per_rad=source_aero.normal_alpha_per_rad,
            roll_beta_per_rad=source_aero.roll_beta_per_rad,
            pitch_alpha_per_rad=source_aero.pitch_alpha_per_rad,
            yaw_beta_per_rad=source_aero.yaw_beta_per_rad,
            roll_rate=source_aero.roll_rate,
            pitch_rate=source_aero.pitch_rate,
            yaw_rate=source_aero.yaw_rate,
        )
    return replace(
        base,
        propulsion=propulsion,
        mass_properties=mass_properties,
        aerodynamics=aerodynamics,
    )


def _perturbed_wind(base: WindModel, sample: DispersionSample) -> WindModel:
    profile = WindProfile(
        base.profile.altitudes_m,
        base.profile.velocities_ned_mps * sample.wind_scale,
    )
    return WindModel(
        profile,
        gust_std_ned_mps=base.gust_std_ned_mps * sample.wind_scale,
        correlation_time_s=base.correlation_time_s,
        sample_step_s=base.sample_step_s,
        horizon_s=base.horizon_s,
        seed=sample.random_seed,
    )


def _scaled_sensor(
    parameters: SensorErrorParameters,
    noise_scale: float,
    bias_scale: float,
) -> SensorErrorParameters:
    return SensorErrorParameters(
        sample_rate_hz=parameters.sample_rate_hz,
        noise_std=parameters.noise_std * noise_scale,
        constant_bias=parameters.constant_bias * bias_scale,
        bias_drift_std_per_sqrt_s=parameters.bias_drift_std_per_sqrt_s * noise_scale,
        quantisation=parameters.quantisation,
        delay_s=parameters.delay_s,
        dropout_probability=parameters.dropout_probability,
        dropout_intervals_s=parameters.dropout_intervals_s,
    )


def _scaled_pid(gains: PIDGains, scale: float) -> PIDGains:
    return replace(
        gains,
        proportional=gains.proportional * scale,
        integral=gains.integral * scale,
        derivative=gains.derivative * scale,
    )


def _requirement_evidence(
    metrics: dict[str, float], requirements: MonteCarloRequirements
) -> tuple[dict[str, float], dict[str, bool]]:
    margins = {
        "minimum_apogee": metrics["apogee_m"] - requirements.minimum_apogee_m,
        "maximum_dynamic_pressure": requirements.maximum_dynamic_pressure_pa
        - metrics["maximum_dynamic_pressure_pa"],
        "maximum_landing_range": requirements.maximum_landing_range_m - metrics["landing_range_m"],
        "maximum_navigation_rms": requirements.maximum_navigation_rms_m
        - metrics["navigation_altitude_rms_m"],
        "maximum_control_settling_time": requirements.maximum_control_settling_time_s
        - metrics["control_settling_time_s"],
    }
    passes = {name: bool(np.isfinite(value) and value >= 0.0) for name, value in margins.items()}
    passes["overall"] = all(passes.values())
    return margins, passes


def _execute_sample(
    configuration: MonteCarloConfiguration,
    sample: DispersionSample,
) -> MonteCarloRunResult:
    try:
        base = load_three_dof_configuration(configuration.base_scenario_path)
        vehicle = _perturbed_vehicle(base.vehicle, sample)
        environment = EnvironmentDefinition(
            base.environment.atmosphere,
            base.environment.gravity,
            _perturbed_wind(base.environment.wind, sample),
        )
        launch = replace(
            base.launch,
            initial_speed_mps=max(
                0.1, base.launch.initial_speed_mps + sample.initial_speed_offset_mps
            ),
            elevation_deg=float(
                np.clip(
                    base.launch.elevation_deg + sample.initial_elevation_offset_deg,
                    75.0,
                    90.0,
                )
            ),
        )
        flight_configuration = replace(
            base,
            simulation=replace(base.simulation, random_seed=sample.random_seed),
            launch=launch,
            environment=environment,
            vehicle=vehicle,
        )
        flight = simulate_three_dof(
            flight_configuration,
            thrust_misalignment_pitch_rad=float(np.deg2rad(sample.thrust_misalignment_pitch_deg)),
            thrust_misalignment_yaw_rad=float(np.deg2rad(sample.thrust_misalignment_yaw_deg)),
        )

        navigation_base = load_navigation_demo_configuration(configuration.navigation_config_path)
        navigation_configuration = replace(
            navigation_base,
            random_seed=sample.random_seed,
            base=flight_configuration,
            accelerometer=_scaled_sensor(
                navigation_base.accelerometer,
                sample.sensor_noise_scale,
                sample.sensor_bias_scale,
            ),
            barometer=_scaled_sensor(
                navigation_base.barometer,
                sample.sensor_noise_scale,
                sample.sensor_bias_scale,
            ),
            gnss=_scaled_sensor(
                navigation_base.gnss,
                sample.sensor_noise_scale,
                sample.sensor_bias_scale,
            ),
        )
        navigation = run_navigation_demo(navigation_configuration, truth_override=flight)

        attitude_base = load_attitude_control_configuration(configuration.attitude_config_path)
        attitude_configuration = replace(
            attitude_base,
            actuator_limits=replace(
                attitude_base.actuator_limits,
                command_delay_s=attitude_base.actuator_limits.command_delay_s
                * sample.actuator_delay_scale,
            ),
            attitude_pid=_scaled_pid(attitude_base.attitude_pid, sample.controller_gain_scale),
            rate_pid=_scaled_pid(attitude_base.rate_pid, sample.controller_gain_scale),
        )
        attitude = simulate_attitude_control(attitude_configuration, "cascaded_pid")
        metrics = {
            "apogee_m": float(np.max(flight.columns["altitude_m"])),
            "landing_range_m": float(flight.columns["ground_range_m"][-1]),
            "maximum_dynamic_pressure_pa": float(np.max(flight.columns["dynamic_pressure_pa"])),
            "maximum_mach": float(np.max(flight.columns["mach"])),
            "maximum_speed_mps": float(np.max(flight.columns["total_velocity_mps"])),
            "flight_time_s": float(flight.time_s[-1]),
            "navigation_altitude_rms_m": navigation.estimated_altitude_rms_m,
            "raw_barometer_rms_m": navigation.raw_barometer_rms_m,
            "control_settling_time_s": attitude.metrics.settling_time_s,
            "control_overshoot_percent": attitude.metrics.overshoot_percent,
            "control_effort_nm2s": attitude.metrics.control_effort_nm2s,
            "actuator_saturation_duration_s": attitude.metrics.actuator_saturation_duration_s,
        }
        margins, passes = _requirement_evidence(metrics, configuration.requirements)
        return MonteCarloRunResult(
            run_index=sample.run_index,
            random_seed=sample.random_seed,
            sample=sample.parameter_dict(),
            success=True,
            error=None,
            metrics=metrics,
            requirement_margins=margins,
            requirement_pass=passes,
        )
    # Ensemble isolation deliberately converts any member-level model/configuration
    # exception into a recorded failed run while other members continue.
    except Exception as error:
        return MonteCarloRunResult(
            run_index=sample.run_index,
            random_seed=sample.random_seed,
            sample=sample.parameter_dict(),
            success=False,
            error=f"{type(error).__name__}: {error}",
            metrics={},
            requirement_margins={},
            requirement_pass={"overall": False},
        )


def _statistics(values: np.ndarray) -> dict[str, float]:
    count = values.size
    standard_deviation = float(np.std(values, ddof=1)) if count > 1 else 0.0
    half_width = float(1.96 * standard_deviation / np.sqrt(count))
    mean = float(np.mean(values))
    return {
        "count": float(count),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
        "mean_ci95_low": mean - half_width,
        "mean_ci95_high": mean + half_width,
    }


def summarise_monte_carlo(
    name: str,
    master_seed: int,
    runs: tuple[MonteCarloRunResult, ...],
) -> MonteCarloSummary:
    """Calculate statistics, Pearson sensitivity, margins, and worst cases."""
    successful = [run for run in runs if run.success]
    statistics: dict[str, dict[str, float]] = {}
    correlations: dict[str, float] = {}
    worst_cases: dict[str, int] = {}
    pass_rates: dict[str, float] = {}
    if successful:
        metric_names = successful[0].metrics.keys()
        for metric_name in metric_names:
            finite_runs = [run for run in successful if np.isfinite(run.metrics[metric_name])]
            if not finite_runs:
                continue
            values = np.array([run.metrics[metric_name] for run in finite_runs])
            statistics[metric_name] = _statistics(values)
            if metric_name in {"apogee_m"}:
                worst_cases[metric_name] = finite_runs[int(np.argmin(values))].run_index
            else:
                worst_cases[metric_name] = finite_runs[int(np.argmax(values))].run_index

        parameter_names = successful[0].sample.keys()
        for parameter_name in parameter_names:
            parameter_values = np.array([run.sample[parameter_name] for run in successful])
            if np.std(parameter_values) <= 1.0e-15:
                continue
            for metric_name in statistics:
                metric_values = np.array([run.metrics[metric_name] for run in successful])
                finite = np.isfinite(metric_values)
                if np.count_nonzero(finite) < 2 or np.std(metric_values[finite]) <= 1.0e-15:
                    continue
                correlations[f"{parameter_name}__{metric_name}"] = float(
                    np.corrcoef(parameter_values[finite], metric_values[finite])[0, 1]
                )
        requirement_names = successful[0].requirement_pass.keys()
        for requirement_name in requirement_names:
            pass_rates[requirement_name] = float(
                np.mean([run.requirement_pass.get(requirement_name, False) for run in runs])
            )
        margin_names = successful[0].requirement_margins.keys()
        for margin_name in margin_names:
            margin_values = np.array([run.requirement_margins[margin_name] for run in successful])
            worst_cases[f"margin_{margin_name}"] = successful[
                int(np.argmin(margin_values))
            ].run_index
    return MonteCarloSummary(
        name=name,
        master_seed=master_seed,
        runs=runs,
        statistics=statistics,
        correlations=correlations,
        worst_case_runs=worst_cases,
        requirement_pass_rates=pass_rates,
    )


def run_monte_carlo(
    configuration: MonteCarloConfiguration,
    *,
    sample_count: int | None = None,
    workers: int | None = None,
) -> MonteCarloSummary:
    """Run ordered coupled samples sequentially or with process-level parallelism."""
    actual_count = configuration.sample_count if sample_count is None else sample_count
    actual_workers = configuration.workers if workers is None else workers
    if actual_count <= 0 or actual_workers <= 0:
        raise ValueError("sample_count and workers must be positive")
    samples = generate_dispersion_samples(
        actual_count, configuration.master_seed, configuration.dispersions
    )
    if actual_workers == 1:
        runs = tuple(_execute_sample(configuration, sample) for sample in samples)
    else:
        indexed_results: dict[int, MonteCarloRunResult] = {}
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_execute_sample, configuration, sample): sample
                for sample in samples
            }
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    indexed_results[sample.run_index] = future.result()
                except Exception as error:  # pragma: no cover - worker process failure
                    indexed_results[sample.run_index] = MonteCarloRunResult(
                        sample.run_index,
                        sample.random_seed,
                        sample.parameter_dict(),
                        False,
                        f"worker failure: {type(error).__name__}: {error}",
                        {},
                        {},
                        {"overall": False},
                    )
        runs = tuple(indexed_results[index] for index in range(actual_count))
    return summarise_monte_carlo(configuration.name, configuration.master_seed, runs)


def write_monte_carlo_outputs(
    summary: MonteCarloSummary, output_directory: str | Path
) -> tuple[Path, Path]:
    """Write compact per-run CSV and aggregate JSON; no trajectory ensembles."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "monte_carlo_runs.csv"
    parameter_names = list(summary.runs[0].sample) if summary.runs else []
    metric_names = sorted({name for run in summary.runs for name in run.metrics})
    margin_names = sorted({name for run in summary.runs for name in run.requirement_margins})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "run_index",
                "random_seed",
                "success",
                "error",
                *parameter_names,
                *metric_names,
                *(f"margin_{name}" for name in margin_names),
                "overall_pass",
            ]
        )
        for run in summary.runs:
            writer.writerow(
                [
                    run.run_index,
                    run.random_seed,
                    int(run.success),
                    run.error or "",
                    *(f"{run.sample[name]:.10g}" for name in parameter_names),
                    *(
                        f"{run.metrics[name]:.10g}" if name in run.metrics else ""
                        for name in metric_names
                    ),
                    *(
                        f"{run.requirement_margins[name]:.10g}"
                        if name in run.requirement_margins
                        else ""
                        for name in margin_names
                    ),
                    int(run.requirement_pass.get("overall", False)),
                ]
            )
    json_path = output / "monte_carlo_summary.json"
    payload = {
        "name": summary.name,
        "master_seed": summary.master_seed,
        "sample_count": len(summary.runs),
        "successful_count": summary.successful_count,
        "failed_count": summary.failed_count,
        "statistics": summary.statistics,
        "correlations": summary.correlations,
        "worst_case_runs": summary.worst_case_runs,
        "requirement_pass_rates": summary.requirement_pass_rates,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path
