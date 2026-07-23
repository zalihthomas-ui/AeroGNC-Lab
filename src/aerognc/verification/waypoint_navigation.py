"""Independent truth scoring for the waypoint estimated-navigation provider."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.navigation.estimated_provider import (
    EstimatedNavigationParameters,
    EstimatedNavigationProvider,
)
from aerognc.navigation.state import NavigationState


@dataclass(frozen=True, slots=True)
class WaypointNavigationCampaignLimits:
    """Quantitative dropout/recovery acceptance bounds."""

    maximum_pre_outage_position_rms_m: float = 3.0
    maximum_outage_position_error_m: float = 15.0
    maximum_recovery_position_rms_m: float = 3.0
    maximum_recovery_velocity_rms_mps: float = 0.5
    minimum_observed_gnss_age_s: float = 19.5
    maximum_observed_gnss_age_s: float = 21.0

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("navigation campaign limits must be positive and finite")
        if self.maximum_observed_gnss_age_s < self.minimum_observed_gnss_age_s:
            raise ValueError("maximum GNSS age bound must not precede its minimum")


@dataclass(frozen=True, slots=True)
class WaypointNavigationCampaignResult:
    """Truth-separated campaign metrics and estimator-only final diagnostics."""

    duration_s: float
    dropout_start_s: float
    dropout_end_s: float
    pre_outage_position_rms_m: float
    outage_maximum_position_error_m: float
    recovery_position_rms_m: float
    recovery_velocity_rms_mps: float
    maximum_observed_gnss_age_s: float
    minimum_covariance_eigenvalue: float
    all_states_valid: bool
    final_diagnostics: dict[str, object]
    limits: WaypointNavigationCampaignLimits

    @property
    def passed(self) -> bool:
        """Return whether estimation, outage bridging, and recovery meet every bound."""
        gnss_integrity = self.final_diagnostics.get("gnss_integrity")
        healthy = isinstance(gnss_integrity, dict) and gnss_integrity.get("health") == "healthy"
        accepted = (
            int(gnss_integrity.get("accepted_count", 0)) > 0
            if isinstance(gnss_integrity, dict)
            else False
        )
        return bool(
            self.all_states_valid
            and self.minimum_covariance_eigenvalue >= -1.0e-9
            and self.pre_outage_position_rms_m <= self.limits.maximum_pre_outage_position_rms_m
            and self.outage_maximum_position_error_m <= self.limits.maximum_outage_position_error_m
            and self.recovery_position_rms_m <= self.limits.maximum_recovery_position_rms_m
            and self.recovery_velocity_rms_mps <= self.limits.maximum_recovery_velocity_rms_mps
            and self.limits.minimum_observed_gnss_age_s
            <= self.maximum_observed_gnss_age_s
            <= self.limits.maximum_observed_gnss_age_s
            and healthy
            and accepted
        )

    def summary(self) -> dict[str, object]:
        """Return a portable, deterministic validation record."""
        return {
            "schema_version": "1.0",
            "scope": "simulation-only truth-isolated estimated waypoint navigation",
            "passed": self.passed,
            "dropout_window_s": [self.dropout_start_s, self.dropout_end_s],
            "metrics": {
                "pre_outage_position_rms_m": self.pre_outage_position_rms_m,
                "outage_maximum_position_error_m": self.outage_maximum_position_error_m,
                "recovery_position_rms_m": self.recovery_position_rms_m,
                "recovery_velocity_rms_mps": self.recovery_velocity_rms_mps,
                "maximum_observed_gnss_age_s": self.maximum_observed_gnss_age_s,
                "minimum_covariance_eigenvalue": self.minimum_covariance_eigenvalue,
                "all_states_valid": self.all_states_valid,
            },
            "limits": asdict(self.limits),
            "estimator_diagnostics": self.final_diagnostics,
            "interpretation": (
                "Truth is used only by this scoring campaign. The controller-facing provider "
                "receives delayed synthetic sensors and emits estimated state; passing is "
                "research evidence, not navigation certification."
            ),
        }


def _campaign_truth(time_s: float) -> NavigationState:
    radius_m = 600.0
    turn_rate_radps = 1.0 / 30.0
    phase = turn_rate_radps * time_s
    horizontal_speed_mps = radius_m * turn_rate_radps
    altitude_m = 120.0 + 15.0 * np.sin(0.015 * time_s)
    climb_rate_mps = 0.225 * np.cos(0.015 * time_s)
    climb_acceleration_mps2 = -0.003375 * np.sin(0.015 * time_s)
    pitch_rad = float(np.arctan2(climb_rate_mps, horizontal_speed_mps))
    pitch_rate_radps = float(
        climb_acceleration_mps2
        * horizontal_speed_mps
        / (horizontal_speed_mps**2 + climb_rate_mps**2)
    )
    roll_rad = float(np.arctan2(horizontal_speed_mps**2, 9.80665 * radius_m))
    yaw_rad = float((phase + np.pi) % (2.0 * np.pi) - np.pi)
    body_rates = np.array(
        [
            -turn_rate_radps * np.sin(pitch_rad),
            pitch_rate_radps * np.cos(roll_rad)
            + turn_rate_radps * np.sin(roll_rad) * np.cos(pitch_rad),
            -pitch_rate_radps * np.sin(roll_rad)
            + turn_rate_radps * np.cos(roll_rad) * np.cos(pitch_rad),
        ]
    )
    return NavigationState(
        position_ned_m=np.array(
            [
                radius_m * np.sin(phase),
                radius_m * (1.0 - np.cos(phase)),
                -altitude_m,
            ]
        ),
        velocity_ned_mps=np.array(
            [
                horizontal_speed_mps * np.cos(phase),
                horizontal_speed_mps * np.sin(phase),
                -climb_rate_mps,
            ]
        ),
        quaternion_nb=euler321_to_quaternion(roll_rad, pitch_rad, yaw_rad),
        angular_rate_body_radps=body_rates,
        airspeed_mps=float(np.hypot(horizontal_speed_mps, climb_rate_mps)),
    )


def run_waypoint_navigation_campaign(
    parameters: EstimatedNavigationParameters,
    *,
    duration_s: float = 120.0,
    limits: WaypointNavigationCampaignLimits | None = None,
) -> WaypointNavigationCampaignResult:
    """Run a deterministic turning/climbing trajectory through GNSS outage/recovery."""
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("navigation campaign duration_s must be positive and finite")
    if len(parameters.gnss.dropout_intervals_s) != 1:
        raise ValueError("navigation campaign requires exactly one GNSS dropout interval")
    dropout_start_s, dropout_end_s = parameters.gnss.dropout_intervals_s[0]
    if dropout_start_s < 20.0 or dropout_end_s + 10.0 > duration_s:
        raise ValueError("navigation campaign needs pre-outage and recovery scoring windows")

    provider = EstimatedNavigationProvider(parameters)
    sample_count = round(duration_s / parameters.step_s) + 1
    times = np.arange(sample_count, dtype=np.float64) * parameters.step_s
    position_errors = np.empty((sample_count, 3), dtype=np.float64)
    velocity_errors = np.empty((sample_count, 3), dtype=np.float64)
    valid = np.empty(sample_count, dtype=np.bool_)
    gnss_ages = np.empty(sample_count, dtype=np.float64)
    covariance_eigenvalues = np.empty(sample_count, dtype=np.float64)

    for index, time_s in enumerate(times):
        truth = _campaign_truth(float(time_s))
        estimate = provider.update(truth, parameters.step_s)
        position_errors[index] = estimate.position_ned_m - truth.position_ned_m
        velocity_errors[index] = estimate.velocity_ned_mps - truth.velocity_ned_mps
        valid[index] = estimate.valid
        diagnostics = provider.diagnostics()
        gnss_ages[index] = float(cast(float, diagnostics["gnss_age_s"]))
        covariance_eigenvalues[index] = float(
            cast(float, diagnostics["covariance_minimum_eigenvalue"])
        )

    pre_mask = (times >= dropout_start_s - 20.0) & (times < dropout_start_s)
    outage_mask = (times >= dropout_start_s) & (times <= dropout_end_s)
    recovery_mask = (times >= dropout_end_s + 5.0) & (times <= dropout_end_s + 25.0)
    position_norm = np.linalg.norm(position_errors, axis=1)
    velocity_norm = np.linalg.norm(velocity_errors, axis=1)

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))

    return WaypointNavigationCampaignResult(
        duration_s=float(times[-1]),
        dropout_start_s=dropout_start_s,
        dropout_end_s=dropout_end_s,
        pre_outage_position_rms_m=rms(position_norm[pre_mask]),
        outage_maximum_position_error_m=float(np.max(position_norm[outage_mask])),
        recovery_position_rms_m=rms(position_norm[recovery_mask]),
        recovery_velocity_rms_mps=rms(velocity_norm[recovery_mask]),
        maximum_observed_gnss_age_s=float(np.max(gnss_ages)),
        minimum_covariance_eigenvalue=float(np.min(covariance_eigenvalues)),
        all_states_valid=bool(np.all(valid)),
        final_diagnostics=dict(provider.diagnostics()),
        limits=limits or WaypointNavigationCampaignLimits(),
    )


def write_waypoint_navigation_campaign(
    result: WaypointNavigationCampaignResult,
    path: str | Path,
) -> Path:
    """Write campaign evidence as deterministic JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


__all__ = [
    "WaypointNavigationCampaignLimits",
    "WaypointNavigationCampaignResult",
    "run_waypoint_navigation_campaign",
    "write_waypoint_navigation_campaign",
]
