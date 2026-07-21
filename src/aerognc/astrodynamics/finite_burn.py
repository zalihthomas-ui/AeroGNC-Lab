"""Adaptive two-body finite-burn execution with exact maneuver boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.astrodynamics.maneuvers import (
    FiniteBurn,
    vector_in_inertial_frame,
)
from aerognc.mathematics.adaptive_integrators import (
    AdaptiveOptions,
    AdaptiveStatistics,
    integrate_adaptive,
)
from aerognc.mathematics.integrators import DerivativeFunction, EventOccurrence
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class FiniteBurnExecution:
    """Seven-state trajectory and burn/mass accounting evidence."""

    time_s: FloatArray
    state: FloatArray
    events: tuple[EventOccurrence, EventOccurrence]
    segment_statistics: tuple[AdaptiveStatistics, ...]
    propellant_used_kg: float
    ideal_delivered_delta_v_mps: float
    mass_balance_error_kg: float


def execute_two_body_finite_burn(
    initial_state: npt.ArrayLike,
    burn: FiniteBurn,
    *,
    gravitational_parameter_m3_s2: float,
    dry_mass_kg: float,
    end_time_s: float,
    integration_options: AdaptiveOptions | None = None,
) -> FiniteBurnExecution:
    """Propagate coast/burn/coast segments without stepping across burn boundaries."""
    state = np.asarray(initial_state, dtype=np.float64)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError("finite-burn initial_state must contain seven finite values")
    if not np.isfinite(gravitational_parameter_m3_s2) or gravitational_parameter_m3_s2 <= 0.0:
        raise ValueError("gravitational_parameter_m3_s2 must be positive and finite")
    if not np.isfinite(dry_mass_kg) or dry_mass_kg <= 0.0 or state[6] <= dry_mass_kg:
        raise ValueError("initial mass must exceed a positive finite dry_mass_kg")
    if not np.isfinite(end_time_s) or end_time_s <= burn.end_time_s:
        raise ValueError("end_time_s must be finite and later than burn end")
    required_propellant_kg = burn.mass_flow_rate_kg_s * burn.duration_s
    if required_propellant_kg > state[6] - dry_mass_kg + 1.0e-12:
        raise FloatingPointError(
            "finite burn requires more propellant than available above dry mass"
        )
    options = AdaptiveOptions() if integration_options is None else integration_options
    initial_mass_kg = float(state[6])
    time_values = [0.0]
    state_values = [state.copy()]
    statistics: list[AdaptiveStatistics] = []
    occurrences: list[EventOccurrence] = []
    boundaries = ((0.0, burn.start_time_s, False), (burn.start_time_s, burn.end_time_s, True))
    segments = (*boundaries, (burn.end_time_s, end_time_s, False))

    def make_derivative(active_burn: bool) -> DerivativeFunction:
        def derivative(_time_s: float, current_state: FloatArray) -> FloatArray:
            position = current_state[:3]
            radius_m = float(np.linalg.norm(position))
            if radius_m <= 0.0:
                raise FloatingPointError("finite-burn trajectory reached central singularity")
            output = np.zeros(7, dtype=np.float64)
            output[:3] = current_state[3:6]
            output[3:6] = -gravitational_parameter_m3_s2 * position / radius_m**3
            if active_burn:
                direction = vector_in_inertial_frame(
                    burn.direction,
                    burn.frame,
                    current_state[:3],
                    current_state[3:6],
                )
                direction /= np.linalg.norm(direction)
                output[3:6] += burn.thrust_n / current_state[6] * direction
                output[6] = -burn.mass_flow_rate_kg_s
            return output

        return derivative

    for start_time_s, stop_time_s, active_burn in segments:
        if stop_time_s <= start_time_s:
            continue
        if start_time_s == burn.start_time_s:
            occurrences.append(EventOccurrence("burn_start", start_time_s, state.copy()))

        integrated = integrate_adaptive(
            make_derivative(active_burn),
            state,
            (start_time_s, stop_time_s),
            options=options,
        )
        time_values.extend(float(value) for value in integrated.time_s[1:])
        state_values.extend(value.copy() for value in integrated.state[1:])
        statistics.append(integrated.statistics)
        state = integrated.state[-1].copy()
        if state[6] < dry_mass_kg - 1.0e-10:
            raise FloatingPointError("finite-burn integration crossed the dry-mass floor")
        if stop_time_s == burn.end_time_s:
            occurrences.append(EventOccurrence("burn_end", stop_time_s, state.copy()))

    if len(occurrences) != 2:
        raise RuntimeError("finite-burn boundary event accounting is inconsistent")
    final_mass_kg = float(state_values[-1][6])
    propellant_used_kg = initial_mass_kg - final_mass_kg
    ideal_delta_v_mps = burn.specific_impulse_s * 9.80665 * np.log(initial_mass_kg / final_mass_kg)
    return FiniteBurnExecution(
        np.asarray(time_values, dtype=np.float64),
        np.vstack(state_values),
        (occurrences[0], occurrences[1]),
        tuple(statistics),
        propellant_used_kg,
        float(ideal_delta_v_mps),
        propellant_used_kg - required_propellant_kg,
    )
