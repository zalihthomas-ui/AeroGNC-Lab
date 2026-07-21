"""Configured linear flight-control analysis and deterministic result writing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerognc.configuration.analysis_loader import FlightControlAnalysisConfiguration
from aerognc.gnc.flight_analysis import (
    DynamicMode,
    LinearModel,
    LQRDesign,
    SILTimingResult,
    StabilityMargins,
    TrimResult,
    analyze_modes,
    benchmark_controller_sil,
    build_lqr_gain_schedule,
    continuous_lqr,
    frequency_response,
    linearize_dynamics,
    solve_trim,
    stability_margins_siso,
)
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class FlightControlAnalysisResult:
    """Complete configured pitch-channel engineering analysis."""

    configuration: FlightControlAnalysisConfiguration
    trim: TrimResult
    nominal_model: LinearModel
    lqr: LQRDesign
    scheduling_points: FloatArray
    scheduled_inertia_kgm2: FloatArray
    scheduled_gains: FloatArray
    scheduled_designs: tuple[LQRDesign, ...]
    modes: tuple[DynamicMode, ...]
    angular_frequency_radps: FloatArray
    open_loop_response: npt.NDArray[np.complex128]
    margins: StabilityMargins
    sil_timing: SILTimingResult

    def as_dict(self, *, include_timing: bool = True) -> dict[str, object]:
        """Return a JSON-safe report; timing can be omitted from deterministic references."""
        payload: dict[str, object] = {
            "scenario": self.configuration.name,
            "trim": {
                "command_moment_nm": float(self.trim.decision[0]),
                "residual_nm": float(self.trim.residual[0]),
                "iterations": self.trim.iterations,
                "converged": self.trim.converged,
            },
            "nominal_system_matrix": self.nominal_model.system_matrix.tolist(),
            "nominal_input_matrix": self.nominal_model.input_matrix.tolist(),
            "lqr_gain": self.lqr.gain.tolist(),
            "riccati_residual_norm": self.lqr.riccati_residual_norm,
            "closed_loop_modes": [
                {
                    "real_radps": mode.eigenvalue.real,
                    "imaginary_radps": mode.eigenvalue.imag,
                    "natural_frequency_radps": mode.natural_frequency_radps,
                    "damping_ratio": mode.damping_ratio,
                    "time_constant_s": mode.time_constant_s,
                    "stable": mode.stable,
                }
                for mode in self.modes
            ],
            "gain_schedule": {
                "scheduling_points": self.scheduling_points.tolist(),
                "inertia_kgm2": self.scheduled_inertia_kgm2.tolist(),
                "gains": self.scheduled_gains.tolist(),
            },
            "stability_margins": {
                "gain_margin": (
                    self.margins.gain_margin if np.isfinite(self.margins.gain_margin) else None
                ),
                "phase_margin_deg": self.margins.phase_margin_deg,
                "gain_crossover_radps": self.margins.gain_crossover_radps,
                "phase_crossover_radps": (
                    self.margins.phase_crossover_radps
                    if np.isfinite(self.margins.phase_crossover_radps)
                    else None
                ),
            },
            "sil_timing": {
                "sample_count": self.sil_timing.sample_count,
                "deadline_s": self.configuration.sil_deadline_s,
                "missed_deadline_count": self.sil_timing.missed_deadline_count,
                "output_checksum": self.sil_timing.output_checksum,
            },
        }
        if include_timing:
            timing = payload["sil_timing"]
            if not isinstance(timing, dict):
                raise RuntimeError("internal SIL timing payload is not a mapping")
            timing.update(
                {
                    "mean_execution_s": self.sil_timing.mean_execution_s,
                    "p95_execution_s": self.sil_timing.p95_execution_s,
                    "maximum_execution_s": self.sil_timing.maximum_execution_s,
                }
            )
        return payload


def _pitch_model(inertia_kgm2: float, passive_damping_nms: float) -> tuple[FloatArray, FloatArray]:
    system = np.array([[0.0, 1.0], [0.0, -passive_damping_nms / inertia_kgm2]], dtype=np.float64)
    inputs = np.array([[0.0], [1.0 / inertia_kgm2]], dtype=np.float64)
    return system, inputs


def run_flight_control_analysis(
    configuration: FlightControlAnalysisConfiguration,
) -> FlightControlAnalysisResult:
    """Execute trim, finite-difference models, LQR, schedule, margins, and SIL timing."""
    trim = solve_trim(
        lambda command: np.array(
            [command[0] + configuration.trim_disturbance_moment_nm], dtype=np.float64
        ),
        [0.0],
        tolerance=1.0e-12,
    )

    def model_for_inertia(inertia_kgm2: float) -> LinearModel:
        system, inputs = _pitch_model(inertia_kgm2, configuration.passive_damping_nms)
        return linearize_dynamics(
            lambda state, control: (
                system @ state + inputs @ (control + configuration.trim_disturbance_moment_nm)
            ),
            [0.0, 0.0],
            [float(trim.decision[0])],
        )

    nominal_model = model_for_inertia(configuration.nominal_inertia_kgm2)
    q_weight = np.diag(configuration.state_weight_diagonal)
    r_weight = np.array([[configuration.input_weight]])
    lqr = continuous_lqr(
        nominal_model.system_matrix,
        nominal_model.input_matrix,
        q_weight,
        r_weight,
    )
    modes = analyze_modes(nominal_model.system_matrix - nominal_model.input_matrix @ lqr.gain)
    scheduled_models = tuple(
        model_for_inertia(inertia) for inertia in configuration.scheduled_inertia_kgm2
    )
    scheduled_gains, scheduled_designs = build_lqr_gain_schedule(
        configuration.scheduling_points,
        scheduled_models,
        q_weight,
        r_weight,
    )
    angular_frequency = np.logspace(
        np.log10(configuration.minimum_frequency_radps),
        np.log10(configuration.maximum_frequency_radps),
        configuration.frequency_sample_count,
    )
    open_loop_response = frequency_response(
        nominal_model.system_matrix,
        nominal_model.input_matrix,
        lqr.gain,
        np.zeros((1, 1)),
        angular_frequency,
    )[:, 0, 0]
    margins = stability_margins_siso(angular_frequency, open_loop_response)
    generator = np.random.default_rng(configuration.random_seed)
    sil_inputs = generator.normal(size=(configuration.sil_sample_count, 2))
    sil_timing = benchmark_controller_sil(
        lambda state: -lqr.gain @ state,
        sil_inputs,
        deadline_s=configuration.sil_deadline_s,
        repeat_count=configuration.sil_repeat_count,
    )
    return FlightControlAnalysisResult(
        configuration,
        trim,
        nominal_model,
        lqr,
        np.asarray(configuration.scheduling_points),
        np.asarray(configuration.scheduled_inertia_kgm2),
        scheduled_gains,
        scheduled_designs,
        modes,
        angular_frequency,
        open_loop_response,
        margins,
        sil_timing,
    )


def write_flight_control_analysis(
    result: FlightControlAnalysisResult,
    output_directory: str | Path | None = None,
    *,
    include_timing: bool = True,
) -> Path:
    """Write the engineering analysis as stable JSON."""
    output = Path(output_directory or result.configuration.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "flight_control_analysis.json"
    path.write_text(
        json.dumps(result.as_dict(include_timing=include_timing), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
