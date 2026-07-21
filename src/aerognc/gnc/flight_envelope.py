"""Envelope-wide trim, linearisation, modal, and scheduled-control analysis."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise, product

import numpy as np
import numpy.typing as npt

from aerognc.configuration.envelope_loader import FlightEnvelopeConfiguration
from aerognc.gnc.flight_analysis import (
    DynamicMode,
    LinearModel,
    LQRDesign,
    TrimResult,
    analyze_modes,
    continuous_lqr,
    controllability_matrix,
    linearize_dynamics,
    observability_matrix,
    solve_trim,
)
from aerognc.mathematics.interpolation import RegularGridTableND
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class EnvelopeOperatingPoint:
    """One dimensional flight condition and scheduled mass-property state."""

    mach: float
    altitude_m: float
    mass_kg: float
    airspeed_mps: float
    dynamic_pressure_pa: float
    centre_of_gravity_from_nose_m: float
    pitch_inertia_kgm2: float


@dataclass(frozen=True, slots=True)
class EnvelopePointAnalysis:
    """Trim and control evidence at one flight-envelope grid point."""

    operating_point: EnvelopeOperatingPoint
    trim: TrimResult
    linear_model: LinearModel
    open_loop_modes: tuple[DynamicMode, ...]
    lqr: LQRDesign
    closed_loop_modes: tuple[DynamicMode, ...]
    controllability_rank: int
    observability_rank: int
    control_authority_fraction: float
    remaining_control_moment_nm: float


class ScheduledStateFeedback:
    """Trilinear Mach/altitude/mass interpolation of state-feedback gains."""

    def __init__(
        self,
        mach_points: npt.ArrayLike,
        altitude_points_m: npt.ArrayLike,
        mass_points_kg: npt.ArrayLike,
        gain_values: npt.ArrayLike,
    ) -> None:
        mach = np.asarray(mach_points, dtype=np.float64)
        altitude = np.asarray(altitude_points_m, dtype=np.float64)
        mass = np.asarray(mass_points_kg, dtype=np.float64)
        gains = np.asarray(gain_values, dtype=np.float64)
        expected_shape = (mach.size, altitude.size, mass.size, 2)
        if gains.shape != expected_shape or not np.all(np.isfinite(gains)):
            raise ValueError(f"gain_values must be finite with shape {expected_shape}")
        axes = (mach, altitude, mass)
        self._tables = tuple(
            RegularGridTableND(axes, gains[..., state_index], "clamp") for state_index in range(2)
        )
        self.mach_points = mach.copy()
        self.altitude_points_m = altitude.copy()
        self.mass_points_kg = mass.copy()
        self.gain_values = gains.copy()

    def gain(self, mach: float, altitude_m: float, mass_kg: float) -> FloatArray:
        """Return the interpolated one-input, two-state gain row."""
        return np.array(
            [table(mach, altitude_m, mass_kg) for table in self._tables],
            dtype=np.float64,
        )[None, :]

    def command(
        self,
        state_error: npt.ArrayLike,
        *,
        mach: float,
        altitude_m: float,
        mass_kg: float,
        trim_command_rad: float = 0.0,
        command_limit_rad: float | None = None,
    ) -> float:
        """Evaluate bounded scheduled feedback around an optional trim command."""
        error = np.asarray(state_error, dtype=np.float64)
        if error.shape != (2,) or not np.all(np.isfinite(error)):
            raise ValueError("state_error must be a finite two-element vector")
        command = float(trim_command_rad - (self.gain(mach, altitude_m, mass_kg) @ error)[0])
        if command_limit_rad is not None:
            if not np.isfinite(command_limit_rad) or command_limit_rad <= 0.0:
                raise ValueError("command_limit_rad must be positive and finite")
            command = float(np.clip(command, -command_limit_rad, command_limit_rad))
        return command


@dataclass(frozen=True, slots=True)
class ScheduleVerification:
    """Interpolated-gain stability evidence between design grid points."""

    evaluated_point_count: int
    stable_point_count: int
    minimum_damping_ratio: float
    worst_real_eigenvalue_radps: float


@dataclass(frozen=True, slots=True)
class RobustnessVerification:
    """Seeded uncertain-model closed-loop stability evidence."""

    sample_count: int
    stable_sample_count: int
    stable_fraction: float
    minimum_damping_ratio: float
    worst_real_eigenvalue_radps: float


@dataclass(frozen=True, slots=True)
class FlightEnvelopeResult:
    """Complete deterministic envelope and scheduled-controller evidence."""

    configuration: FlightEnvelopeConfiguration
    analyses: tuple[EnvelopePointAnalysis, ...]
    gain_schedule: ScheduledStateFeedback
    schedule_verification: ScheduleVerification
    robustness_verification: RobustnessVerification

    @property
    def all_trim_converged(self) -> bool:
        """Return whether every grid-point trim met its numerical tolerance."""
        return all(item.trim.converged for item in self.analyses)

    @property
    def minimum_control_authority_fraction(self) -> float:
        """Return the smallest unused actuator-position fraction on the grid."""
        return min(item.control_authority_fraction for item in self.analyses)

    @property
    def minimum_closed_loop_damping_ratio(self) -> float:
        """Return the smallest finite damping ratio on the design grid."""
        return _minimum_finite_damping(
            mode for item in self.analyses for mode in item.closed_loop_modes
        )


@dataclass(frozen=True, slots=True)
class _UncertaintyScales:
    normal: float = 1.0
    pitch: float = 1.0
    control: float = 1.0
    inertia: float = 1.0


_NOMINAL_SCALES = _UncertaintyScales()


def _minimum_finite_damping(modes: Iterable[DynamicMode]) -> float:
    damping = [mode.damping_ratio for mode in modes if np.isfinite(mode.damping_ratio)]
    return min(damping, default=np.inf)


def _mass_schedule(
    configuration: FlightEnvelopeConfiguration, mass_kg: float
) -> tuple[float, float]:
    model = configuration.base_scenario.vehicle.mass_properties
    span = model.wet_mass_kg - model.dry_mass_kg
    fraction = (mass_kg - model.dry_mass_kg) / span
    centre_of_gravity_m = model.dry_cg_from_nose_m + fraction * (
        model.wet_cg_from_nose_m - model.dry_cg_from_nose_m
    )
    pitch_inertia = model.dry_inertia_body_kgm2[1, 1] + fraction * (
        model.wet_inertia_body_kgm2[1, 1] - model.dry_inertia_body_kgm2[1, 1]
    )
    return float(centre_of_gravity_m), float(pitch_inertia)


def make_operating_point(
    configuration: FlightEnvelopeConfiguration,
    mach: float,
    altitude_m: float,
    mass_kg: float,
) -> EnvelopeOperatingPoint:
    """Construct atmospheric and scheduled properties for one grid/query point."""
    if not np.all(np.isfinite([mach, altitude_m, mass_kg])) or mach <= 0.0 or mass_kg <= 0.0:
        raise ValueError("operating-point inputs must be finite with positive Mach and mass")
    atmosphere = configuration.base_scenario.environment.atmosphere.properties(altitude_m)
    airspeed = mach * atmosphere.speed_of_sound_mps
    dynamic_pressure = 0.5 * atmosphere.density_kgpm3 * airspeed**2
    centre_of_gravity, pitch_inertia = _mass_schedule(configuration, mass_kg)
    return EnvelopeOperatingPoint(
        mach=float(mach),
        altitude_m=float(altitude_m),
        mass_kg=float(mass_kg),
        airspeed_mps=float(airspeed),
        dynamic_pressure_pa=float(dynamic_pressure),
        centre_of_gravity_from_nose_m=centre_of_gravity,
        pitch_inertia_kgm2=pitch_inertia,
    )


def _pitch_dynamics(
    configuration: FlightEnvelopeConfiguration,
    operating_point: EnvelopeOperatingPoint,
    scales: _UncertaintyScales = _NOMINAL_SCALES,
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    aerodynamic_model = configuration.base_scenario.vehicle.aerodynamics
    area_m2 = aerodynamic_model.reference_area_m2
    length_m = aerodynamic_model.reference_length_m
    qbar = operating_point.dynamic_pressure_pa
    velocity = operating_point.airspeed_mps
    mass = operating_point.mass_kg
    inertia = operating_point.pitch_inertia_kgm2 * scales.inertia

    def dynamics(state: FloatArray, control: FloatArray) -> FloatArray:
        alpha_rad, pitch_rate_radps = state
        command_rad = control[0]
        coefficients = aerodynamic_model.coefficients(
            operating_point.mach,
            alpha_rad,
            0.0,
        )
        normal_coefficient = (
            scales.normal * coefficients.normal
            + scales.control * configuration.normal_control_derivative_per_rad * command_rad
        )
        nondimensional_pitch_rate = pitch_rate_radps * length_m / (2.0 * velocity)
        pitch_coefficient = (
            scales.pitch * coefficients.pitch
            + scales.pitch * configuration.pitch_rate_derivative * nondimensional_pitch_rate
            + scales.control * configuration.pitch_control_derivative_per_rad * command_rad
        )
        alpha_rate = pitch_rate_radps + qbar * area_m2 * normal_coefficient / (mass * velocity)
        pitch_acceleration = (
            qbar * area_m2 * length_m * pitch_coefficient + configuration.disturbance_moment_nm
        ) / inertia
        return np.array([alpha_rate, pitch_acceleration], dtype=np.float64)

    return dynamics


def analyze_envelope_point(
    configuration: FlightEnvelopeConfiguration,
    operating_point: EnvelopeOperatingPoint,
    *,
    scales: _UncertaintyScales = _NOMINAL_SCALES,
) -> EnvelopePointAnalysis:
    """Solve trim, derive the local model, and synthesize one feedback gain."""
    dynamics = _pitch_dynamics(configuration, operating_point, scales)
    actuator_limit = configuration.base_scenario.vehicle.actuator_limits.position_limit_rad
    trim = solve_trim(
        lambda decision: dynamics(
            np.array([decision[0], 0.0], dtype=np.float64),
            np.array([decision[1]], dtype=np.float64),
        ),
        [0.0, 0.0],
        lower_bounds=[-configuration.angle_of_attack_limit_rad, -actuator_limit],
        upper_bounds=[configuration.angle_of_attack_limit_rad, actuator_limit],
        tolerance=1.0e-10,
        maximum_iterations=60,
    )
    trim_state = np.array([trim.decision[0], 0.0], dtype=np.float64)
    trim_control = np.array([trim.decision[1]], dtype=np.float64)
    model = linearize_dynamics(dynamics, trim_state, trim_control)
    q_weight = np.diag(configuration.state_weight_diagonal)
    r_weight = np.array([[configuration.input_weight]], dtype=np.float64)
    design = continuous_lqr(model.system_matrix, model.input_matrix, q_weight, r_weight)
    closed_loop_matrix = model.system_matrix - model.input_matrix @ design.gain
    authority_fraction = max(0.0, 1.0 - abs(trim.decision[1]) / actuator_limit)
    aero = configuration.base_scenario.vehicle.aerodynamics
    moment_per_rad = (
        operating_point.dynamic_pressure_pa
        * aero.reference_area_m2
        * aero.reference_length_m
        * abs(configuration.pitch_control_derivative_per_rad)
        * scales.control
    )
    return EnvelopePointAnalysis(
        operating_point=operating_point,
        trim=trim,
        linear_model=model,
        open_loop_modes=analyze_modes(model.system_matrix),
        lqr=design,
        closed_loop_modes=analyze_modes(closed_loop_matrix),
        controllability_rank=int(
            np.linalg.matrix_rank(controllability_matrix(model.system_matrix, model.input_matrix))
        ),
        observability_rank=int(
            np.linalg.matrix_rank(observability_matrix(model.system_matrix, np.eye(2)))
        ),
        control_authority_fraction=float(authority_fraction),
        remaining_control_moment_nm=float(
            moment_per_rad * max(0.0, actuator_limit - abs(trim.decision[1]))
        ),
    )


def _midpoints(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(0.5 * (before + after) for before, after in pairwise(values))


def _verify_interpolated_schedule(
    configuration: FlightEnvelopeConfiguration,
    schedule: ScheduledStateFeedback,
) -> ScheduleVerification:
    stable_count = 0
    point_count = 0
    minimum_damping = np.inf
    worst_real = -np.inf
    for mach, altitude_m, mass_kg in product(
        _midpoints(configuration.mach_points),
        _midpoints(configuration.altitude_points_m),
        _midpoints(configuration.mass_points_kg),
    ):
        point = make_operating_point(configuration, mach, altitude_m, mass_kg)
        analysis = analyze_envelope_point(configuration, point)
        gain = schedule.gain(mach, altitude_m, mass_kg)
        eigenvalues = np.linalg.eigvals(
            analysis.linear_model.system_matrix - analysis.linear_model.input_matrix @ gain
        )
        point_count += 1
        stable = bool(np.all(eigenvalues.real < 0.0))
        stable_count += int(stable)
        modes = analyze_modes(
            analysis.linear_model.system_matrix - analysis.linear_model.input_matrix @ gain
        )
        minimum_damping = min(minimum_damping, _minimum_finite_damping(modes))
        worst_real = max(worst_real, float(np.max(eigenvalues.real)))
    return ScheduleVerification(point_count, stable_count, minimum_damping, worst_real)


def _verify_robustness(
    configuration: FlightEnvelopeConfiguration,
    schedule: ScheduledStateFeedback,
) -> RobustnessVerification:
    generator = np.random.default_rng(configuration.random_seed)
    stable_count = 0
    minimum_damping = np.inf
    worst_real = -np.inf
    for _sample_index in range(configuration.uncertainty_sample_count):
        mach = generator.uniform(configuration.mach_points[0], configuration.mach_points[-1])
        altitude_m = generator.uniform(
            configuration.altitude_points_m[0], configuration.altitude_points_m[-1]
        )
        mass_kg = generator.uniform(
            configuration.mass_points_kg[0], configuration.mass_points_kg[-1]
        )
        scales = _UncertaintyScales(
            normal=max(
                0.05,
                1.0 + generator.normal(scale=configuration.aerodynamic_derivative_sigma_fraction),
            ),
            pitch=max(
                0.05,
                1.0 + generator.normal(scale=configuration.aerodynamic_derivative_sigma_fraction),
            ),
            control=max(
                0.05,
                1.0 + generator.normal(scale=configuration.control_effectiveness_sigma_fraction),
            ),
            inertia=max(
                0.05,
                1.0 + generator.normal(scale=configuration.inertia_sigma_fraction),
            ),
        )
        point = make_operating_point(configuration, mach, altitude_m, mass_kg)
        analysis = analyze_envelope_point(configuration, point, scales=scales)
        gain = schedule.gain(mach, altitude_m, mass_kg)
        closed_loop = (
            analysis.linear_model.system_matrix - analysis.linear_model.input_matrix @ gain
        )
        eigenvalues = np.linalg.eigvals(closed_loop)
        stable = bool(np.all(eigenvalues.real < 0.0))
        stable_count += int(stable)
        minimum_damping = min(minimum_damping, _minimum_finite_damping(analyze_modes(closed_loop)))
        worst_real = max(worst_real, float(np.max(eigenvalues.real)))
    return RobustnessVerification(
        sample_count=configuration.uncertainty_sample_count,
        stable_sample_count=stable_count,
        stable_fraction=stable_count / configuration.uncertainty_sample_count,
        minimum_damping_ratio=minimum_damping,
        worst_real_eigenvalue_radps=worst_real,
    )


def analyze_flight_envelope(
    configuration: FlightEnvelopeConfiguration,
) -> FlightEnvelopeResult:
    """Run the full grid, build its gain schedule, and verify interpolation/uncertainty."""
    analyses: list[EnvelopePointAnalysis] = []
    shape = (
        len(configuration.mach_points),
        len(configuration.altitude_points_m),
        len(configuration.mass_points_kg),
        2,
    )
    gain_values = np.empty(shape, dtype=np.float64)
    for mach_index, mach in enumerate(configuration.mach_points):
        for altitude_index, altitude_m in enumerate(configuration.altitude_points_m):
            for mass_index, mass_kg in enumerate(configuration.mass_points_kg):
                point = make_operating_point(configuration, mach, altitude_m, mass_kg)
                analysis = analyze_envelope_point(configuration, point)
                analyses.append(analysis)
                gain_values[mach_index, altitude_index, mass_index] = analysis.lqr.gain[0]
    schedule = ScheduledStateFeedback(
        configuration.mach_points,
        configuration.altitude_points_m,
        configuration.mass_points_kg,
        gain_values,
    )
    schedule_verification = _verify_interpolated_schedule(configuration, schedule)
    robustness_verification = _verify_robustness(configuration, schedule)
    return FlightEnvelopeResult(
        configuration=configuration,
        analyses=tuple(analyses),
        gain_schedule=schedule,
        schedule_verification=schedule_verification,
        robustness_verification=robustness_verification,
    )
