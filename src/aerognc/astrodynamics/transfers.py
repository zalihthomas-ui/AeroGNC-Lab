"""Directly implemented two-body transfer and hyperbolic-flyby calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HohmannTransfer:
    """Coplanar circular-to-circular Hohmann transfer characteristics."""

    transfer_time_s: float
    departure_delta_v_mps: float
    arrival_delta_v_mps: float
    departure_transfer_speed_mps: float
    arrival_transfer_speed_mps: float


@dataclass(frozen=True, slots=True)
class FlybyGeometry:
    """Unpowered planet-centred hyperbolic flyby characteristics."""

    eccentricity: float
    turn_angle_rad: float
    impact_parameter_m: float
    periapsis_speed_mps: float


def design_hohmann_transfer(
    primary_mu_m3_s2: float, departure_radius_m: float, arrival_radius_m: float
) -> HohmannTransfer:
    """Calculate an ideal coplanar Hohmann transfer directly from vis-viva."""
    values = np.array([primary_mu_m3_s2, departure_radius_m, arrival_radius_m])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Hohmann inputs must be positive and finite")
    if departure_radius_m == arrival_radius_m:
        raise ValueError("Hohmann orbit radii must differ")
    transfer_axis_m = 0.5 * (departure_radius_m + arrival_radius_m)
    departure_circular_speed = np.sqrt(primary_mu_m3_s2 / departure_radius_m)
    arrival_circular_speed = np.sqrt(primary_mu_m3_s2 / arrival_radius_m)
    departure_transfer_speed = np.sqrt(
        primary_mu_m3_s2 * (2.0 / departure_radius_m - 1.0 / transfer_axis_m)
    )
    arrival_transfer_speed = np.sqrt(
        primary_mu_m3_s2 * (2.0 / arrival_radius_m - 1.0 / transfer_axis_m)
    )
    transfer_time_s = np.pi * np.sqrt(transfer_axis_m**3 / primary_mu_m3_s2)
    return HohmannTransfer(
        transfer_time_s=float(transfer_time_s),
        departure_delta_v_mps=float(abs(departure_transfer_speed - departure_circular_speed)),
        arrival_delta_v_mps=float(abs(arrival_circular_speed - arrival_transfer_speed)),
        departure_transfer_speed_mps=float(departure_transfer_speed),
        arrival_transfer_speed_mps=float(arrival_transfer_speed),
    )


def evaluate_hyperbolic_flyby(
    body_mu_m3_s2: float, periapsis_radius_m: float, excess_speed_mps: float
) -> FlybyGeometry:
    """Calculate ideal unpowered flyby geometry from periapsis and v-infinity."""
    values = np.array([body_mu_m3_s2, periapsis_radius_m, excess_speed_mps])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("flyby inputs must be positive and finite")
    eccentricity = 1.0 + periapsis_radius_m * excess_speed_mps**2 / body_mu_m3_s2
    turn_angle_rad = 2.0 * np.arcsin(1.0 / eccentricity)
    impact_parameter_m = (
        body_mu_m3_s2 / excess_speed_mps**2 * np.sqrt(max(eccentricity**2 - 1.0, 0.0))
    )
    periapsis_speed_mps = np.sqrt(excess_speed_mps**2 + 2.0 * body_mu_m3_s2 / periapsis_radius_m)
    return FlybyGeometry(
        eccentricity=float(eccentricity),
        turn_angle_rad=float(turn_angle_rad),
        impact_parameter_m=float(impact_parameter_m),
        periapsis_speed_mps=float(periapsis_speed_mps),
    )
