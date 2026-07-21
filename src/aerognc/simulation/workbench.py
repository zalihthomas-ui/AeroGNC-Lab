"""Validated services used by the beginner-friendly desktop workbench.

The graphical interface intentionally contains no engineering equations.  It turns
plain-language form values into the same immutable configurations exercised by the
CLI and automated verification suite.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from aerognc.catalogs import ConfirmedExoplanet, ExoplanetCatalog, load_exoplanet_catalog
from aerognc.configuration.orbit_tour_loader import (
    OrbitTourConfiguration,
    load_orbit_tour_configuration,
)
from aerognc.configuration.planetary_catalog import PlanetaryCatalog
from aerognc.configuration.six_dof_loader import SixDofConfiguration, load_six_dof_configuration
from aerognc.simulation.logging import SimulationResult
from aerognc.simulation.orbit_assisted_tour import (
    OrbitTourSimulation,
    simulate_orbit_assisted_tour,
)
from aerognc.simulation.six_dof_simulator import simulate_six_dof
from aerognc.visualisation.playback import MAXIMUM_PLAYBACK_SPEED, MINIMUM_PLAYBACK_SPEED

SECONDS_PER_DAY = 86_400.0


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _bounded(value: float, name: str, minimum: float, maximum: float) -> float:
    result = _finite(value, name)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


def _finite_triplet(values: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    result = tuple(_finite(value, f"{name}[{index}]") for index, value in enumerate(values))
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RocketWorkbenchInputs:
    """Editable 6-DOF rocket inputs with units encoded in every field name."""

    duration_s: float = 8.0
    step_s: float = 0.005
    initial_speed_mps: float = 10.0
    initial_euler321_deg: tuple[float, float, float] = (0.0, 86.0, 15.0)
    initial_angular_rate_body_degps: tuple[float, float, float] = (0.5, -0.3, 0.2)
    playback_speed: float = 2.0

    def validate(self, base: SixDofConfiguration) -> None:
        """Reject values outside the verified model and playback domains."""
        reference_end_s = float(base.reference_schedule.time_s[-1])
        maximum_duration_s = min(base.base.simulation.maximum_time_s, reference_end_s)
        _bounded(self.duration_s, "duration [s]", 0.25, maximum_duration_s)
        _bounded(self.step_s, "integration step [s]", 0.0001, 0.02)
        if self.step_s > self.duration_s:
            raise ValueError("integration step [s] must not exceed duration [s]")
        _bounded(self.initial_speed_mps, "initial speed [m/s]", 0.0, 500.0)
        euler = _finite_triplet(self.initial_euler321_deg, "initial Euler angle [deg]")
        if any(abs(value) > 180.0 for value in euler):
            raise ValueError("each initial Euler angle [deg] must be within -180..180")
        rates = _finite_triplet(
            self.initial_angular_rate_body_degps,
            "initial body rate [deg/s]",
        )
        if any(abs(value) > 90.0 for value in rates):
            raise ValueError("each initial body rate [deg/s] must be within -90..90")
        _bounded(
            self.playback_speed,
            "playback speed",
            MINIMUM_PLAYBACK_SPEED,
            MAXIMUM_PLAYBACK_SPEED,
        )


def build_rocket_configuration(
    base_path: str | Path,
    inputs: RocketWorkbenchInputs,
) -> SixDofConfiguration:
    """Return a validated immutable 6-DOF configuration from friendly inputs."""
    base = load_six_dof_configuration(base_path)
    inputs.validate(base)
    return replace(
        base,
        duration_s=float(inputs.duration_s),
        step_s=float(inputs.step_s),
        initial_speed_mps=float(inputs.initial_speed_mps),
        initial_euler321_deg=_finite_triplet(
            inputs.initial_euler321_deg, "initial Euler angle [deg]"
        ),
        initial_angular_rate_body_degps=_finite_triplet(
            inputs.initial_angular_rate_body_degps,
            "initial body rate [deg/s]",
        ),
    )


def run_rocket_workbench(
    base_path: str | Path,
    inputs: RocketWorkbenchInputs,
) -> tuple[SixDofConfiguration, SimulationResult]:
    """Build and run one deterministic closed-loop 6-DOF rocket case."""
    configuration = build_rocket_configuration(base_path, inputs)
    return configuration, simulate_six_dof(configuration)


@dataclass(frozen=True, slots=True)
class OrbitTourWorkbenchInputs:
    """Plain-language controls for the fictional capture/dwell/departure tour."""

    departure_body: str = "Asteria"
    assist_body: str = "Neria"
    destination_body: str = "Caelus"
    departure_day: float = 0.0
    assist_arrival_day: float = 240.0
    destination_arrival_day: float = 2035.4
    departure_parking_altitude_km: float = 300.0
    assist_parking_altitude_km: float = 300.0
    destination_parking_altitude_km: float = 300.0
    assist_dwell_revolutions: int = 2
    initial_mass_kg: float = 130_000.0
    dry_mass_kg: float = 8_000.0
    specific_impulse_s: float = 1_200.0
    maximum_total_delta_v_mps: float = 32_000.0
    minimum_final_mass_kg: float = 8_500.0
    playback_days_per_second: float = 80.0

    def validate(self, catalog: PlanetaryCatalog) -> None:
        """Reject ambiguous routes and physically inconsistent spacecraft inputs."""
        available = {body.name for body in catalog.bodies}
        route = (self.departure_body, self.assist_body, self.destination_body)
        if any(name not in available for name in route):
            missing = sorted(set(route) - available)
            raise ValueError(f"unknown fictional world(s): {', '.join(missing)}")
        if len(set(route)) != 3:
            raise ValueError("departure, orbit-assist, and destination worlds must differ")
        departure_day = _bounded(self.departure_day, "departure day", 0.0, 100_000.0)
        assist_day = _bounded(self.assist_arrival_day, "orbit-assist arrival day", 0.001, 100_000.0)
        destination_day = _bounded(
            self.destination_arrival_day,
            "destination arrival day",
            0.002,
            100_000.0,
        )
        if not departure_day < assist_day < destination_day:
            raise ValueError("mission days must increase: departure < assist < destination")
        for value, name in (
            (self.departure_parking_altitude_km, "departure parking altitude [km]"),
            (self.assist_parking_altitude_km, "assist parking altitude [km]"),
            (self.destination_parking_altitude_km, "destination parking altitude [km]"),
        ):
            _bounded(value, name, 1.0, 1_000_000.0)
        if isinstance(self.assist_dwell_revolutions, bool) or not isinstance(
            self.assist_dwell_revolutions, int
        ):
            raise ValueError("assist dwell revolutions must be a whole number")
        if not 1 <= self.assist_dwell_revolutions <= 100:
            raise ValueError("assist dwell revolutions must be between 1 and 100")
        initial_mass = _bounded(self.initial_mass_kg, "initial mass [kg]", 1.0, 1.0e9)
        dry_mass = _bounded(self.dry_mass_kg, "dry mass [kg]", 1.0, 1.0e9)
        if dry_mass >= initial_mass:
            raise ValueError("dry mass [kg] must be lower than initial mass [kg]")
        _bounded(self.specific_impulse_s, "specific impulse [s]", 1.0, 100_000.0)
        _bounded(
            self.maximum_total_delta_v_mps,
            "maximum total delta-v [m/s]",
            1.0,
            1.0e7,
        )
        minimum_mass = _bounded(
            self.minimum_final_mass_kg,
            "minimum final mass [kg]",
            1.0,
            initial_mass,
        )
        if minimum_mass < dry_mass:
            raise ValueError("minimum final mass [kg] cannot be below dry mass [kg]")
        _bounded(
            self.playback_days_per_second,
            "playback rate [days/s]",
            0.1,
            10_000.0,
        )


def build_orbit_tour_configuration(
    base_path: str | Path,
    inputs: OrbitTourWorkbenchInputs,
) -> OrbitTourConfiguration:
    """Return a validated tour configuration without editing the source YAML."""
    base = load_orbit_tour_configuration(base_path)
    inputs.validate(base.catalog)
    return replace(
        base,
        departure_body=inputs.departure_body,
        assist_body=inputs.assist_body,
        destination_body=inputs.destination_body,
        departure_time_s=float(inputs.departure_day) * SECONDS_PER_DAY,
        assist_arrival_time_s=float(inputs.assist_arrival_day) * SECONDS_PER_DAY,
        destination_arrival_time_s=float(inputs.destination_arrival_day) * SECONDS_PER_DAY,
        departure_parking_altitude_m=float(inputs.departure_parking_altitude_km) * 1_000.0,
        assist_parking_altitude_m=float(inputs.assist_parking_altitude_km) * 1_000.0,
        destination_parking_altitude_m=float(inputs.destination_parking_altitude_km) * 1_000.0,
        assist_dwell_revolutions=inputs.assist_dwell_revolutions,
        initial_mass_kg=float(inputs.initial_mass_kg),
        dry_mass_kg=float(inputs.dry_mass_kg),
        specific_impulse_s=float(inputs.specific_impulse_s),
        maximum_total_delta_v_mps=float(inputs.maximum_total_delta_v_mps),
        minimum_final_mass_kg=float(inputs.minimum_final_mass_kg),
    )


def run_orbit_tour_workbench(
    base_path: str | Path,
    inputs: OrbitTourWorkbenchInputs,
) -> OrbitTourSimulation:
    """Build and run one deterministic fictional planetary tour."""
    configuration = build_orbit_tour_configuration(base_path, inputs)
    return simulate_orbit_assisted_tour(configuration)


@dataclass(frozen=True, slots=True)
class CatalogWorkbenchInputs:
    """Search fields for the observational confirmed-exoplanet snapshot."""

    text: str = ""
    maximum_distance_pc: float | None = None
    discovery_method: str | None = None
    minimum_discovery_year: int | None = None
    maximum_discovery_year: int | None = None
    limit: int = 250

    def validate(self) -> None:
        """Validate optional catalog bounds before applying them."""
        if self.maximum_distance_pc is not None:
            _bounded(self.maximum_distance_pc, "maximum distance [pc]", 0.001, 1.0e9)
        if (
            self.minimum_discovery_year is not None
            and not 1700 <= self.minimum_discovery_year <= 2200
        ):
            raise ValueError("minimum discovery year must be between 1700 and 2200")
        if (
            self.maximum_discovery_year is not None
            and not 1700 <= self.maximum_discovery_year <= 2200
        ):
            raise ValueError("maximum discovery year must be between 1700 and 2200")
        if (
            self.minimum_discovery_year is not None
            and self.maximum_discovery_year is not None
            and self.maximum_discovery_year < self.minimum_discovery_year
        ):
            raise ValueError("maximum discovery year must not precede minimum year")
        if not 1 <= self.limit <= 5_000:
            raise ValueError("result limit must be between 1 and 5000")


def search_exoplanet_catalog(
    catalog: ExoplanetCatalog,
    inputs: CatalogWorkbenchInputs,
) -> tuple[ConfirmedExoplanet, ...]:
    """Apply validated user filters to an already verified catalog snapshot."""
    inputs.validate()
    return catalog.search(
        text=inputs.text,
        maximum_distance_pc=inputs.maximum_distance_pc,
        discovery_method=inputs.discovery_method,
        minimum_discovery_year=inputs.minimum_discovery_year,
        maximum_discovery_year=inputs.maximum_discovery_year,
        limit=inputs.limit,
    )


def load_workbench_catalog(
    csv_path: str | Path,
    metadata_path: str | Path,
) -> ExoplanetCatalog:
    """Load and checksum-verify the bundled observational catalog snapshot."""
    return load_exoplanet_catalog(Path(csv_path), Path(metadata_path))
