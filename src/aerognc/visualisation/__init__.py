"""Consistent publication-style plotting with lazy backend selection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aerognc.catalogs import ConfirmedExoplanet, ExoplanetCatalog
    from aerognc.gnc.flight_envelope import FlightEnvelopeResult
    from aerognc.simulation.advanced_navigation import (
        AdvancedNavigationResult,
        NavigationConsistencyResult,
    )
    from aerognc.simulation.guided_ascent import AscentGuidanceOptimizationResult
    from aerognc.simulation.logging import SimulationResult
    from aerognc.simulation.orbit_assisted_tour import OrbitTourSimulation
    from aerognc.verification.flight_data_identification import (
        FlightDataIdentificationResult,
    )
    from aerognc.verification.launch_window import LaunchWindowRun


def plot_three_dof_results(
    result: SimulationResult, output_directory: str | Path
) -> tuple[Path, ...]:
    """Load the noninteractive plotter only when static figures are requested."""
    from aerognc.visualisation.three_dof import plot_three_dof_results as implementation

    return implementation(result, output_directory)


def plot_rotating_ascent(result: SimulationResult, output_directory: str | Path) -> Path:
    """Load the rotating-ascent plotter only when requested."""
    from aerognc.visualisation.rotating_ascent import plot_rotating_ascent as implementation

    return implementation(result, output_directory)


def plot_multistage_recovery(result: SimulationResult, output_directory: str | Path) -> Path:
    """Load the multistage/recovery plotter only when requested."""
    from aerognc.visualisation.multistage_recovery import (
        plot_multistage_recovery as implementation,
    )

    return implementation(result, output_directory)


def plot_flight_envelope(result: FlightEnvelopeResult, output_directory: str | Path) -> Path:
    """Load the envelope plotter only when requested."""
    from aerognc.visualisation.flight_envelope import plot_flight_envelope as implementation

    return implementation(result, output_directory)


def plot_ascent_guidance(
    result: AscentGuidanceOptimizationResult, output_directory: str | Path
) -> Path:
    """Load the constrained-ascent plotter only when requested."""
    from aerognc.visualisation.ascent_guidance import plot_ascent_guidance as implementation

    return implementation(result, output_directory)


def plot_advanced_navigation(
    result: AdvancedNavigationResult,
    consistency: NavigationConsistencyResult,
    output_directory: str | Path,
) -> Path:
    """Load the advanced-navigation plotter only when requested."""
    from aerognc.visualisation.advanced_navigation import (
        plot_advanced_navigation as implementation,
    )

    return implementation(result, consistency, output_directory)


def plot_flight_data_identification(
    result: FlightDataIdentificationResult,
    output_directory: str | Path,
) -> Path:
    """Load the flight-data identification plotter only when requested."""
    from aerognc.visualisation.flight_data_identification import (
        plot_flight_data_identification as implementation,
    )

    return implementation(result, output_directory)


def plot_orbit_assisted_tour(
    simulation: OrbitTourSimulation,
    output_directory: str | Path,
) -> Path:
    """Load the orbit-tour plotter only when requested."""
    from aerognc.visualisation.orbit_assisted_tour import (
        plot_orbit_assisted_tour as implementation,
    )

    return implementation(simulation, output_directory)


def plot_launch_window_optimization(
    run: LaunchWindowRun,
    output_directory: str | Path,
) -> Path:
    """Load the launch-window plotter only when requested."""
    from aerognc.visualisation.launch_window import (
        plot_launch_window_optimization as implementation,
    )

    return implementation(run, output_directory)


def plot_milky_way_catalog(
    catalog: ExoplanetCatalog,
    selection: tuple[ConfirmedExoplanet, ...],
    output_directory: str | Path,
) -> Path:
    """Load the observational catalog plotter only when requested."""
    from aerognc.visualisation.galaxy_catalog import plot_milky_way_catalog as implementation

    return implementation(catalog, selection, output_directory)


__all__ = [
    "plot_advanced_navigation",
    "plot_ascent_guidance",
    "plot_flight_data_identification",
    "plot_flight_envelope",
    "plot_launch_window_optimization",
    "plot_milky_way_catalog",
    "plot_multistage_recovery",
    "plot_orbit_assisted_tour",
    "plot_rotating_ascent",
    "plot_three_dof_results",
]
