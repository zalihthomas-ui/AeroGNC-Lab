"""Deterministic evidence products for the observational catalog layer."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from aerognc.catalogs import (
    ConfirmedExoplanet,
    ExoplanetCatalog,
    MilkyWayMetadata,
    SolarSystemPlanet,
)


def galaxy_catalog_payload(
    catalog: ExoplanetCatalog,
    selection: tuple[ConfirmedExoplanet, ...],
    galaxy: MilkyWayMetadata,
    solar_system: tuple[SolarSystemPlanet, ...],
) -> dict[str, object]:
    """Build a transparent, missing-data-aware catalog report."""
    return {
        "scope": {
            "observational_data": True,
            "simulation_ephemeris": False,
            "complete_milky_way_census": False,
            "note": catalog.provenance.scope_note,
        },
        "milky_way_context": asdict(galaxy),
        "solar_system_planets": [asdict(planet) for planet in solar_system],
        "provenance": asdict(catalog.provenance),
        "snapshot_summary": asdict(catalog.summary()),
        "selection_summary": asdict(catalog.summary(selection)),
        "selection_preview": [
            {
                "name": planet.name,
                "host_name": planet.host_name,
                "discovery_method": planet.discovery_method,
                "discovery_year": planet.discovery_year,
                "distance_pc": planet.system_distance_pc,
                "orbital_period_days": planet.orbital_period_days,
            }
            for planet in selection[:25]
        ],
    }


def write_galaxy_catalog_outputs(
    catalog: ExoplanetCatalog,
    selection: tuple[ConfirmedExoplanet, ...],
    galaxy: MilkyWayMetadata,
    solar_system: tuple[SolarSystemPlanet, ...],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write a JSON summary and the complete filtered selection as CSV."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "galaxy_catalog_summary.json"
    report_path.write_text(
        json.dumps(
            galaxy_catalog_payload(catalog, selection, galaxy, solar_system),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selection_path = output / "exoplanet_selection.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "planet_name",
                "host_name",
                "discovery_method",
                "discovery_year",
                "orbital_period_days",
                "semimajor_axis_au",
                "radius_earth",
                "mass_earth",
                "system_distance_pc",
                "right_ascension_deg",
                "declination_deg",
            )
        )
        for planet in selection:
            writer.writerow(
                (
                    planet.name,
                    planet.host_name,
                    planet.discovery_method,
                    planet.discovery_year,
                    planet.orbital_period_days,
                    planet.semimajor_axis_au,
                    planet.radius_earth,
                    planet.mass_earth,
                    planet.system_distance_pc,
                    planet.right_ascension_deg,
                    planet.declination_deg,
                )
            )
    return report_path, selection_path
