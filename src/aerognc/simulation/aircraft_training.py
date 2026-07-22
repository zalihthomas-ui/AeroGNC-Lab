"""Civilian flight presets, preflight calculations, and transparent training scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.environment.orbital_atmosphere import ReferenceOrbitalAtmosphere
from aerognc.simulation.aircraft_telemetry import AircraftTelemetry
from aerognc.vehicle.fixed_wing import (
    STANDARD_GRAVITY_MPS2,
    AircraftControlCommand,
    AircraftState,
    aircraft_stall_speed_mps,
    longitudinal_trim_command,
)

AircraftPresetName = Literal[
    "level_flight",
    "coordinated_turn",
    "stall_demonstration",
    "crosswind_response",
    "high_altitude_research",
]


@dataclass(frozen=True, slots=True)
class AircraftPreset:
    """One understandable setup preset; every changed value remains editable."""

    key: AircraftPresetName
    display_name: str
    purpose: str


AIRCRAFT_PRESETS: tuple[AircraftPreset, ...] = (
    AircraftPreset("level_flight", "Level Flight", "Stable first flight near the trim point."),
    AircraftPreset(
        "coordinated_turn",
        "Coordinated 360 Turn",
        "Start with safe energy for a manual bank-and-rudder exercise.",
    ),
    AircraftPreset(
        "stall_demonstration",
        "Stall Demonstration",
        "High-altitude, low-speed setup for synthetic stall and recovery practice.",
    ),
    AircraftPreset(
        "crosswind_response",
        "Crosswind + Gust",
        "Seeded turbulence and a finite crosswind pulse for disturbance rejection.",
    ),
    AircraftPreset(
        "high_altitude_research",
        "High-Altitude Research",
        "Thin-atmosphere starting point for the public-safe 100 km exploration aid.",
    ),
)


def apply_aircraft_preset(
    configuration: AircraftSandboxConfiguration,
    preset: AircraftPresetName,
) -> AircraftSandboxConfiguration:
    """Return a validated configuration variant for one civilian exercise."""
    if preset == "level_flight":
        return replace(
            configuration,
            initial=replace(
                configuration.initial,
                altitude_m=1_500.0,
                true_airspeed_mps=82.0,
                heading_rad=np.deg2rad(90.0),
                flight_path_angle_rad=0.0,
                bank_angle_rad=0.0,
                angle_of_attack_rad=np.deg2rad(2.8),
            ),
            initial_throttle=0.28,
            wind_north_mps=2.0,
            wind_east_mps=4.0,
            turbulence_std_ned_mps=(0.0, 0.0, 0.0),
            gust_amplitude_ned_mps=(0.0, 0.0, 0.0),
        )
    if preset == "coordinated_turn":
        return replace(
            configuration,
            initial=replace(
                configuration.initial,
                altitude_m=2_500.0,
                true_airspeed_mps=105.0,
                heading_rad=0.0,
                bank_angle_rad=np.deg2rad(20.0),
                angle_of_attack_rad=np.deg2rad(3.5),
            ),
            initial_throttle=0.46,
            turbulence_std_ned_mps=(0.0, 0.0, 0.0),
            gust_amplitude_ned_mps=(0.0, 0.0, 0.0),
        )
    if preset == "stall_demonstration":
        return replace(
            configuration,
            initial=replace(
                configuration.initial,
                altitude_m=3_500.0,
                true_airspeed_mps=62.0,
                heading_rad=np.deg2rad(90.0),
                flight_path_angle_rad=np.deg2rad(2.0),
                bank_angle_rad=0.0,
                angle_of_attack_rad=np.deg2rad(11.0),
            ),
            initial_throttle=0.24,
            turbulence_std_ned_mps=(0.0, 0.0, 0.0),
            gust_amplitude_ned_mps=(0.0, 0.0, 0.0),
        )
    if preset == "crosswind_response":
        return replace(
            configuration,
            initial=replace(
                configuration.initial,
                altitude_m=2_000.0,
                true_airspeed_mps=92.0,
                heading_rad=0.0,
                bank_angle_rad=0.0,
                angle_of_attack_rad=np.deg2rad(3.0),
            ),
            initial_throttle=0.36,
            wind_north_mps=0.0,
            wind_east_mps=12.0,
            turbulence_std_ned_mps=(1.2, 1.8, 0.7),
            turbulence_correlation_time_s=2.5,
            gust_start_time_s=12.0,
            gust_duration_s=8.0,
            gust_amplitude_ned_mps=(0.0, 8.0, 0.0),
        )
    if preset == "high_altitude_research":
        return replace(
            configuration,
            initial=replace(
                configuration.initial,
                altitude_m=13_000.0,
                true_airspeed_mps=210.0,
                heading_rad=np.deg2rad(90.0),
                flight_path_angle_rad=np.deg2rad(8.0),
                bank_angle_rad=0.0,
                angle_of_attack_rad=np.deg2rad(3.0),
            ),
            initial_throttle=1.0,
            turbulence_std_ned_mps=(0.4, 0.4, 0.2),
            gust_amplitude_ned_mps=(0.0, 0.0, 0.0),
        )
    raise ValueError(f"unknown aircraft preset: {preset}")


@dataclass(frozen=True, slots=True)
class AircraftPreflight:
    """Calculated setup facts displayed before a live or batch run."""

    wing_loading_kgpm2: float
    maximum_air_breathing_thrust_to_weight: float
    stall_speed_1g_mps: float
    fuel_mass_kg: float
    estimated_fuel_endurance_s: float
    initial_mach: float
    warning: str


def aircraft_preflight(configuration: AircraftSandboxConfiguration) -> AircraftPreflight:
    """Calculate transparent first-order preflight values from the selected inputs."""
    atmosphere = ReferenceOrbitalAtmosphere(configuration.planet.atmosphere_density_scale)
    properties = atmosphere.properties(configuration.initial.altitude_m)
    stall_speed = aircraft_stall_speed_mps(
        configuration.mass.initial_mass_kg,
        properties.density_kgpm3,
        configuration,
    )
    fuel_mass_kg = configuration.mass.initial_mass_kg - configuration.mass.dry_mass_kg
    fuel_flow_kgps = configuration.mass.maximum_fuel_flow_kgps * configuration.initial_throttle
    endurance_s = np.inf if fuel_flow_kgps <= 1.0e-12 else fuel_mass_kg / fuel_flow_kgps
    return AircraftPreflight(
        wing_loading_kgpm2=configuration.mass.initial_mass_kg / configuration.geometry.wing_area_m2,
        maximum_air_breathing_thrust_to_weight=configuration.propulsion.maximum_thrust_n
        / (configuration.mass.initial_mass_kg * STANDARD_GRAVITY_MPS2),
        stall_speed_1g_mps=stall_speed,
        fuel_mass_kg=fuel_mass_kg,
        estimated_fuel_endurance_s=float(endurance_s),
        initial_mach=configuration.initial.true_airspeed_mps / properties.speed_of_sound_mps,
        warning=(
            "Synthetic educational estimate only; no structural, handling-quality, or "
            "certification claim."
        ),
    )


TrainingTask = Literal[
    "altitude_speed_hold",
    "coordinated_360_turn",
    "stall_recovery",
    "research_altitude_crossing",
]


@dataclass(frozen=True, slots=True)
class TrainingEvaluation:
    """Scored civilian exercise with visible pass criteria and raw metrics."""

    task: TrainingTask
    passed: bool
    score_percent: float
    metrics: dict[str, float]
    criteria: tuple[str, ...]
    interpretation: str


def _control_smoothness(commands: Sequence[AircraftControlCommand], time_s: np.ndarray) -> float:
    if len(commands) < 2:
        return 0.0
    axes = np.asarray([[item.roll, item.pitch, item.yaw] for item in commands])
    rates = np.diff(axes, axis=0) / np.diff(time_s)[:, None]
    return float(np.sqrt(np.mean(rates**2)))


def evaluate_training_task(
    task: TrainingTask,
    samples: Sequence[AircraftTelemetry],
    commands: Sequence[AircraftControlCommand],
    *,
    target_altitude_m: float | None = None,
    target_airspeed_mps: float | None = None,
) -> TrainingEvaluation:
    """Score one non-operational flying exercise from recorded telemetry."""
    if len(samples) < 2 or len(commands) != len(samples):
        raise ValueError("training evaluation needs equal telemetry/command sequences")
    time_s = np.asarray([sample.time_s for sample in samples])
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("training telemetry times must be strictly increasing")
    altitude = np.asarray([sample.altitude_m for sample in samples])
    airspeed = np.asarray([sample.true_airspeed_mps for sample in samples])
    smoothness = _control_smoothness(commands, time_s)
    if task == "altitude_speed_hold":
        target_altitude = altitude[0] if target_altitude_m is None else target_altitude_m
        target_airspeed = airspeed[0] if target_airspeed_mps is None else target_airspeed_mps
        altitude_rms = float(np.sqrt(np.mean((altitude - target_altitude) ** 2)))
        airspeed_rms = float(np.sqrt(np.mean((airspeed - target_airspeed) ** 2)))
        passed = altitude_rms <= 75.0 and airspeed_rms <= 7.5 and smoothness <= 1.5
        score = 100.0 - 0.45 * altitude_rms - 4.0 * airspeed_rms - 8.0 * smoothness
        return TrainingEvaluation(
            task,
            passed,
            float(np.clip(score, 0.0, 100.0)),
            {
                "altitude_rms_error_m": altitude_rms,
                "airspeed_rms_error_mps": airspeed_rms,
                "control_smoothness_per_s": smoothness,
            },
            ("altitude RMS <= 75 m", "airspeed RMS <= 7.5 m/s", "smoothness <= 1.5 /s"),
            "Tracks initial altitude and speed; this is a handling exercise, not navigation.",
        )
    if task == "coordinated_360_turn":
        headings = np.unwrap(np.deg2rad([sample.heading_deg for sample in samples]))
        heading_change_deg = float(abs(np.rad2deg(headings[-1] - headings[0])))
        sideslip_rms_deg = float(
            np.sqrt(np.mean(np.asarray([sample.sideslip_angle_deg for sample in samples]) ** 2))
        )
        altitude_loss_m = float(max(0.0, altitude[0] - np.min(altitude)))
        passed = (
            heading_change_deg >= 350.0 and sideslip_rms_deg <= 5.0 and altitude_loss_m <= 250.0
        )
        score = (
            min(100.0, heading_change_deg / 3.6) - 5.0 * sideslip_rms_deg - 0.08 * altitude_loss_m
        )
        return TrainingEvaluation(
            task,
            passed,
            float(np.clip(score, 0.0, 100.0)),
            {
                "absolute_heading_change_deg": heading_change_deg,
                "sideslip_rms_deg": sideslip_rms_deg,
                "maximum_altitude_loss_m": altitude_loss_m,
                "control_smoothness_per_s": smoothness,
            },
            ("heading change >= 350 deg", "sideslip RMS <= 5 deg", "altitude loss <= 250 m"),
            "Evaluates a civilian coordinated turn with no pursuit or target logic.",
        )
    if task == "stall_recovery":
        stalled = np.asarray([sample.stalled for sample in samples])
        onset_indices = np.flatnonzero(stalled)
        recovery_time_s = float(time_s[-1] - time_s[0])
        altitude_loss_m = float(max(0.0, altitude[0] - np.min(altitude)))
        recovered = False
        if onset_indices.size:
            onset = int(onset_indices[0])
            recovered_indices = np.flatnonzero(np.logical_not(stalled[onset + 1 :]))
            if recovered_indices.size:
                recovery = onset + 1 + int(recovered_indices[0])
                recovery_time_s = float(time_s[recovery] - time_s[onset])
                altitude_loss_m = float(
                    max(
                        0.0,
                        altitude[onset] - np.min(altitude[onset : recovery + 1]),
                    )
                )
                recovered = True
        passed = recovered and recovery_time_s <= 15.0 and altitude_loss_m <= 500.0
        score = 100.0 - 4.0 * recovery_time_s - 0.08 * altitude_loss_m if recovered else 0.0
        return TrainingEvaluation(
            task,
            passed,
            float(np.clip(score, 0.0, 100.0)),
            {
                "recovery_time_s": recovery_time_s,
                "altitude_loss_m": altitude_loss_m,
                "control_smoothness_per_s": smoothness,
            },
            ("stall must occur then clear", "recovery <= 15 s", "altitude loss <= 500 m"),
            "Uses the project's synthetic coefficient-break stall model; spins are omitted.",
        )
    if task == "research_altitude_crossing":
        maximum_altitude_m = float(np.max(altitude))
        passed = maximum_altitude_m >= 100_000.0
        return TrainingEvaluation(
            task,
            passed,
            float(np.clip(maximum_altitude_m / 1_000.0, 0.0, 100.0)),
            {
                "maximum_altitude_m": maximum_altitude_m,
                "maximum_mach": max(sample.mach for sample in samples),
                "control_smoothness_per_s": smoothness,
            },
            ("maximum altitude >= 100000 m",),
            "Crossing 100 km is a research boundary and does not demonstrate orbit.",
        )
    raise ValueError(f"unknown training task: {task}")


def scripted_demo_command(
    time_s: float,
    state: AircraftState,
    configuration: AircraftSandboxConfiguration,
) -> AircraftControlCommand:
    """Return a reproducible gentle control demo through the normal command interface."""
    del state
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError("scripted demo time must be finite and nonnegative")
    trim = longitudinal_trim_command(configuration)
    if time_s < 5.0:
        return trim
    if time_s < 12.0:
        return replace(trim, roll=0.28, yaw=0.10)
    if time_s < 19.0:
        return replace(trim, roll=-0.28, yaw=-0.10)
    if time_s < 27.0:
        return replace(trim, pitch=trim.pitch + 0.18, throttle=0.55)
    if time_s < 34.0:
        return replace(trim, pitch=trim.pitch - 0.10, throttle=0.35)
    return trim
