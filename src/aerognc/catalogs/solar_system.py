"""Reader for the eight IAU Solar System planets and selected public properties."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SolarSystemPlanet:
    """Selected mean properties; values are descriptive, not an ephemeris."""

    order_from_sun: int
    name: str
    category: str
    semimajor_axis_au: float
    sidereal_orbit_period_days: float
    mean_radius_km: float
    source_url: str


def load_solar_system_planets(path: Path) -> tuple[SolarSystemPlanet, ...]:
    """Load and validate the ordered eight-planet table."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    planets: list[SolarSystemPlanet] = []
    for row in rows:
        planet = SolarSystemPlanet(
            order_from_sun=int(row["order_from_sun"]),
            name=row["name"].strip(),
            category=row["category"].strip(),
            semimajor_axis_au=float(row["semimajor_axis_au"]),
            sidereal_orbit_period_days=float(row["sidereal_orbit_period_days"]),
            mean_radius_km=float(row["mean_radius_km"]),
            source_url=row["source_url"].strip(),
        )
        if (
            not planet.name
            or planet.semimajor_axis_au <= 0.0
            or planet.sidereal_orbit_period_days <= 0.0
            or planet.mean_radius_km <= 0.0
            or not planet.source_url.startswith("https://")
        ):
            raise ValueError("Solar System planet rows must contain physical sourced values")
        planets.append(planet)
    if tuple(planet.order_from_sun for planet in planets) != tuple(range(1, 9)):
        raise ValueError("Solar System catalog must contain exactly the eight ordered planets")
    if len({planet.name for planet in planets}) != 8:
        raise ValueError("Solar System planet names must be unique")
    return tuple(planets)
