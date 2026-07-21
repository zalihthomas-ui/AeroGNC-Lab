"""Seeded interplanetary uncertainty analysis with graceful failure handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from aerognc.astrodynamics.maneuvers import FiniteBurn, ImpulsiveManeuver, SpacecraftManeuver
from aerognc.configuration.interplanetary_loader import InterplanetaryConfiguration
from aerognc.simulation.interplanetary import simulate_interplanetary


@dataclass(frozen=True, slots=True)
class UncertaintyRun:
    """Inputs, outputs, and status for one deterministic uncertainty sample."""

    index: int
    parameters: Mapping[str, float]
    metrics: Mapping[str, float]
    error: str | None = None

    @property
    def successful(self) -> bool:
        """Return whether the evaluator completed."""
        return self.error is None


@dataclass(frozen=True, slots=True)
class UncertaintySummary:
    """Ordered runs plus statistics, confidence intervals, and sensitivities."""

    seed: int
    runs: tuple[UncertaintyRun, ...]
    metric_statistics: Mapping[str, Mapping[str, float]]
    correlations: Mapping[str, Mapping[str, float]]
    worst_case_runs: Mapping[str, int]

    @property
    def successful_count(self) -> int:
        """Return completed run count."""
        return sum(run.successful for run in self.runs)

    @property
    def failed_count(self) -> int:
        """Return failed run count."""
        return len(self.runs) - self.successful_count


@dataclass(frozen=True, slots=True)
class InterplanetaryDispersion:
    """One-sigma independent dispersions for the synthetic mission model."""

    injection_velocity_sigma_mps: float = 5.0
    initial_mass_sigma_kg: float = 10.0
    body_phase_sigma_rad: float = 1.0e-5
    gravitational_parameter_fraction_sigma: float = 1.0e-6
    maneuver_magnitude_fraction_sigma: float = 0.005

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.injection_velocity_sigma_mps,
                self.initial_mass_sigma_kg,
                self.body_phase_sigma_rad,
                self.gravitational_parameter_fraction_sigma,
                self.maneuver_magnitude_fraction_sigma,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("mission dispersions must be finite and nonnegative")


def run_seeded_uncertainty(
    parameter_sigmas: Mapping[str, float],
    evaluator: Callable[[Mapping[str, float]], Mapping[str, float]],
    *,
    sample_count: int,
    seed: int,
    workers: int = 1,
) -> UncertaintySummary:
    """Evaluate independent Gaussian inputs in a reproducible, ordered ensemble."""
    if sample_count <= 0:
        raise ValueError("uncertainty sample_count must be positive")
    if workers <= 0:
        raise ValueError("uncertainty workers must be positive")
    names = tuple(parameter_sigmas)
    sigmas = np.array([parameter_sigmas[name] for name in names], dtype=np.float64)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0.0):
        raise ValueError("uncertainty standard deviations must be finite and nonnegative")
    generator = np.random.default_rng(seed)
    draws = generator.normal(size=(sample_count, len(names))) * sigmas

    def evaluate(index: int) -> UncertaintyRun:
        parameters = {name: float(draws[index, column]) for column, name in enumerate(names)}
        try:
            metrics = dict(evaluator(parameters))
            if not metrics or not np.all(np.isfinite(list(metrics.values()))):
                raise FloatingPointError("uncertainty evaluator returned non-finite metrics")
            return UncertaintyRun(index, parameters, metrics)
        except (ValueError, RuntimeError, FloatingPointError, OSError) as error:
            return UncertaintyRun(index, parameters, {}, f"{type(error).__name__}: {error}")

    if workers == 1:
        runs = tuple(evaluate(index) for index in range(sample_count))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            runs = tuple(executor.map(evaluate, range(sample_count)))
    successful = [run for run in runs if run.successful]
    if not successful:
        return UncertaintySummary(seed, runs, {}, {}, {})
    metric_names = tuple(successful[0].metrics)
    statistics: dict[str, dict[str, float]] = {}
    worst_cases: dict[str, int] = {}
    for metric_name in metric_names:
        values = np.array([run.metrics[metric_name] for run in successful])
        statistics[metric_name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "p02_5": float(np.percentile(values, 2.5)),
            "median": float(np.median(values)),
            "p97_5": float(np.percentile(values, 97.5)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        worst_cases[metric_name] = successful[
            int(np.argmax(np.abs(values - np.median(values))))
        ].index
    correlations: dict[str, dict[str, float]] = {}
    for parameter_name in names:
        parameter_values = np.array([run.parameters[parameter_name] for run in successful])
        correlations[parameter_name] = {}
        for metric_name in metric_names:
            metric_values = np.array([run.metrics[metric_name] for run in successful])
            if (
                parameter_values.size < 2
                or np.std(parameter_values) == 0.0
                or np.std(metric_values) == 0.0
            ):
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(parameter_values, metric_values)[0, 1])
            correlations[parameter_name][metric_name] = correlation
    return UncertaintySummary(seed, runs, statistics, correlations, worst_cases)


def _scaled_maneuvers(
    maneuvers: Sequence[SpacecraftManeuver], fractional_change: float
) -> tuple[SpacecraftManeuver, ...]:
    scale = max(0.0, 1.0 + fractional_change)
    output: list[SpacecraftManeuver] = []
    for maneuver in maneuvers:
        if isinstance(maneuver, ImpulsiveManeuver):
            output.append(
                replace(
                    maneuver,
                    delta_velocity_mps=(
                        scale * maneuver.delta_velocity_mps[0],
                        scale * maneuver.delta_velocity_mps[1],
                        scale * maneuver.delta_velocity_mps[2],
                    ),
                )
            )
        elif isinstance(maneuver, FiniteBurn):
            output.append(replace(maneuver, thrust_n=scale * maneuver.thrust_n))
    return tuple(output)


def _metric_value(summary: Mapping[str, Mapping[str, float | str]], key: str) -> float:
    value = summary[key]["value"]
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"mission metric {key!r} is not numeric")
    return float(value)


def run_interplanetary_uncertainty(
    configuration: InterplanetaryConfiguration,
    dispersion: InterplanetaryDispersion,
    *,
    sample_count: int,
    seed: int,
    workers: int = 1,
) -> UncertaintySummary:
    """Disperse injection, mass, ephemerides, gravity, and configured maneuvers."""
    parameter_sigmas = {
        "injection_radial_mps": dispersion.injection_velocity_sigma_mps,
        "injection_transverse_mps": dispersion.injection_velocity_sigma_mps,
        "injection_normal_mps": dispersion.injection_velocity_sigma_mps,
        "initial_mass_kg": dispersion.initial_mass_sigma_kg,
        "body_phase_rad": dispersion.body_phase_sigma_rad,
        "gravity_fraction": dispersion.gravitational_parameter_fraction_sigma,
        "maneuver_fraction": dispersion.maneuver_magnitude_fraction_sigma,
    }

    def evaluator(parameters: Mapping[str, float]) -> Mapping[str, float]:
        original_velocity = np.asarray(configuration.spacecraft.velocity_offset_rtn_mps)
        velocity_change = np.array(
            [
                parameters["injection_radial_mps"],
                parameters["injection_transverse_mps"],
                parameters["injection_normal_mps"],
            ]
        )
        mass_kg = max(
            configuration.spacecraft.dry_mass_kg,
            configuration.spacecraft.mass_kg + parameters["initial_mass_kg"],
        )
        spacecraft = replace(
            configuration.spacecraft,
            mass_kg=mass_kg,
            velocity_offset_rtn_mps=tuple(original_velocity + velocity_change),
        )
        bodies = tuple(
            replace(
                body,
                phase_at_epoch_rad=body.phase_at_epoch_rad
                + parameters["body_phase_rad"] * (index + 1) / len(configuration.bodies),
                gravitational_parameter_m3_s2=body.gravitational_parameter_m3_s2
                * max(1.0e-6, 1.0 + parameters["gravity_fraction"]),
            )
            for index, body in enumerate(configuration.bodies)
        )
        dispersed = replace(
            configuration,
            spacecraft=spacecraft,
            bodies=bodies,
            maneuvers=_scaled_maneuvers(configuration.maneuvers, parameters["maneuver_fraction"]),
        )
        summary = simulate_interplanetary(dispersed).result.maximum_summary
        return {
            "assist_closest_approach_m": _metric_value(summary, "assist_closest_approach"),
            "destination_closest_approach_m": _metric_value(
                summary, "destination_closest_approach"
            ),
            "maximum_heliocentric_speed_mps": _metric_value(summary, "maximum_heliocentric_speed"),
        }

    return run_seeded_uncertainty(
        parameter_sigmas,
        evaluator,
        sample_count=sample_count,
        seed=seed,
        workers=workers,
    )
