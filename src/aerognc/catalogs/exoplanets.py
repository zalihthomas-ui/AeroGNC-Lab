"""Validated loader and query service for NASA confirmed-exoplanet snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from aerognc.catalogs.galactic import heliocentric_galactic_xyz_pc


@dataclass(frozen=True, slots=True)
class CatalogProvenance:
    """Acquisition metadata paired with an immutable catalog snapshot."""

    source_name: str
    source_url: str
    source_table: str
    query: str
    retrieved_utc: str
    row_count: int
    sha256: str
    scope_note: str


@dataclass(frozen=True, slots=True)
class ConfirmedExoplanet:
    """One confirmed-planet row; unavailable reported values remain ``None``."""

    name: str
    host_name: str
    system_planet_count: int
    discovery_method: str
    discovery_year: int | None
    orbital_period_days: float | None
    semimajor_axis_au: float | None
    radius_earth: float | None
    mass_earth: float | None
    system_distance_pc: float | None
    right_ascension_deg: float | None
    declination_deg: float | None
    stellar_spectral_type: str | None
    stellar_temperature_k: float | None
    stellar_mass_solar: float | None
    stellar_radius_solar: float | None

    @property
    def has_3d_position(self) -> bool:
        """Whether distance and sky direction support a heliocentric position."""
        return (
            self.system_distance_pc is not None
            and self.right_ascension_deg is not None
            and self.declination_deg is not None
        )

    def galactic_position_pc(self) -> NDArray[np.float64]:
        """Return heliocentric Galactic Cartesian position [pc]."""
        if not self.has_3d_position:
            raise ValueError(f"{self.name!r} has no complete three-dimensional position")
        assert self.system_distance_pc is not None
        assert self.right_ascension_deg is not None
        assert self.declination_deg is not None
        return heliocentric_galactic_xyz_pc(
            self.right_ascension_deg,
            self.declination_deg,
            self.system_distance_pc,
        )


@dataclass(frozen=True, slots=True)
class CatalogSummary:
    """Compact completeness and discovery summary for a snapshot or selection."""

    planet_count: int
    host_count: int
    positioned_planet_count: int
    planets_with_orbital_period: int
    planets_with_mass: int
    planets_with_radius: int
    discovery_year_min: int | None
    discovery_year_max: int | None
    nearest_reported_distance_pc: float | None
    farthest_reported_distance_pc: float | None


@dataclass(frozen=True, slots=True)
class ExoplanetCatalog:
    """A validated, searchable point-in-time catalog."""

    provenance: CatalogProvenance
    planets: tuple[ConfirmedExoplanet, ...]

    def search(
        self,
        *,
        text: str = "",
        maximum_distance_pc: float | None = None,
        discovery_method: str | None = None,
        minimum_discovery_year: int | None = None,
        maximum_discovery_year: int | None = None,
        limit: int | None = None,
    ) -> tuple[ConfirmedExoplanet, ...]:
        """Filter deterministically by human-readable catalog fields."""
        if maximum_distance_pc is not None and maximum_distance_pc <= 0.0:
            raise ValueError("maximum distance must be positive")
        if (
            minimum_discovery_year is not None
            and maximum_discovery_year is not None
            and maximum_discovery_year < minimum_discovery_year
        ):
            raise ValueError("maximum discovery year must not precede minimum year")
        if limit is not None and limit <= 0:
            raise ValueError("catalog result limit must be positive")
        query = text.strip().casefold()
        method = discovery_method.strip().casefold() if discovery_method else None
        matches: list[ConfirmedExoplanet] = []
        for planet in self.planets:
            if (
                query
                and query not in planet.name.casefold()
                and query not in planet.host_name.casefold()
            ):
                continue
            if method is not None and planet.discovery_method.casefold() != method:
                continue
            if maximum_distance_pc is not None and (
                planet.system_distance_pc is None or planet.system_distance_pc > maximum_distance_pc
            ):
                continue
            if minimum_discovery_year is not None and (
                planet.discovery_year is None or planet.discovery_year < minimum_discovery_year
            ):
                continue
            if maximum_discovery_year is not None and (
                planet.discovery_year is None or planet.discovery_year > maximum_discovery_year
            ):
                continue
            matches.append(planet)
            if limit is not None and len(matches) >= limit:
                break
        return tuple(matches)

    def summary(
        self,
        planets: tuple[ConfirmedExoplanet, ...] | None = None,
    ) -> CatalogSummary:
        """Compute missing-data-aware summary statistics."""
        selected = self.planets if planets is None else planets
        years = [planet.discovery_year for planet in selected if planet.discovery_year is not None]
        distances = [
            planet.system_distance_pc
            for planet in selected
            if planet.system_distance_pc is not None
        ]
        return CatalogSummary(
            planet_count=len(selected),
            host_count=len({planet.host_name for planet in selected}),
            positioned_planet_count=sum(planet.has_3d_position for planet in selected),
            planets_with_orbital_period=sum(
                planet.orbital_period_days is not None for planet in selected
            ),
            planets_with_mass=sum(planet.mass_earth is not None for planet in selected),
            planets_with_radius=sum(planet.radius_earth is not None for planet in selected),
            discovery_year_min=min(years) if years else None,
            discovery_year_max=max(years) if years else None,
            nearest_reported_distance_pc=min(distances) if distances else None,
            farthest_reported_distance_pc=max(distances) if distances else None,
        )

    def galactic_positions_pc(
        self,
        planets: tuple[ConfirmedExoplanet, ...] | None = None,
    ) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        """Return names and an ``N x 3`` position matrix for complete rows."""
        selected = self.planets if planets is None else planets
        positioned = tuple(planet for planet in selected if planet.has_3d_position)
        if not positioned:
            return (), np.empty((0, 3), dtype=np.float64)
        return (
            tuple(planet.name for planet in positioned),
            np.vstack([planet.galactic_position_pc() for planet in positioned]),
        )


def _optional_float(value: str | None, field_name: str) -> float | None:
    if value is None or not value.strip():
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"catalog field {field_name} must be finite when present")
    return result


def _optional_int(value: str | None, field_name: str) -> int | None:
    parsed = _optional_float(value, field_name)
    if parsed is None:
        return None
    integer = int(parsed)
    if float(integer) != parsed:
        raise ValueError(f"catalog field {field_name} must be integral")
    return integer


def _required_text(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"catalog field {key} is required")
    return value.strip()


def _planet_from_row(row: dict[str, str | None]) -> ConfirmedExoplanet:
    system_planet_count = _optional_int(row.get("sy_pnum"), "sy_pnum")
    if system_planet_count is None or system_planet_count <= 0:
        raise ValueError("catalog system planet count must be positive")
    return ConfirmedExoplanet(
        name=_required_text(row, "pl_name"),
        host_name=_required_text(row, "hostname"),
        system_planet_count=system_planet_count,
        discovery_method=_required_text(row, "discoverymethod"),
        discovery_year=_optional_int(row.get("disc_year"), "disc_year"),
        orbital_period_days=_optional_float(row.get("pl_orbper"), "pl_orbper"),
        semimajor_axis_au=_optional_float(row.get("pl_orbsmax"), "pl_orbsmax"),
        radius_earth=_optional_float(row.get("pl_rade"), "pl_rade"),
        mass_earth=_optional_float(row.get("pl_bmasse"), "pl_bmasse"),
        system_distance_pc=_optional_float(row.get("sy_dist"), "sy_dist"),
        right_ascension_deg=_optional_float(row.get("ra"), "ra"),
        declination_deg=_optional_float(row.get("dec"), "dec"),
        stellar_spectral_type=(row.get("st_spectype") or "").strip() or None,
        stellar_temperature_k=_optional_float(row.get("st_teff"), "st_teff"),
        stellar_mass_solar=_optional_float(row.get("st_mass"), "st_mass"),
        stellar_radius_solar=_optional_float(row.get("st_rad"), "st_rad"),
    )


def _metadata_value(mapping: dict[str, object], key: str, expected_type: type[str]) -> str:
    value = mapping.get(key)
    if not isinstance(value, expected_type) or not value:
        raise ValueError(f"catalog metadata field {key} must be a non-empty string")
    return value


def load_exoplanet_catalog(csv_path: Path, metadata_path: Path) -> ExoplanetCatalog:
    """Load a snapshot, validate its checksum, metadata, schema, and uniqueness."""
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("catalog CSV and provenance JSON must both exist")
    metadata_object = cast(object, json.loads(metadata_path.read_text(encoding="utf-8")))
    if not isinstance(metadata_object, dict):
        raise ValueError("catalog metadata root must be an object")
    metadata = cast(dict[str, object], metadata_object)
    row_count = metadata.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count <= 0:
        raise ValueError("catalog metadata row_count must be a positive integer")
    expected_sha256 = _metadata_value(metadata, "sha256", str)
    actual_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("catalog CSV SHA-256 does not match its provenance record")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        planets = tuple(_planet_from_row(row) for row in reader)
    if len(planets) != row_count:
        raise ValueError("catalog row count does not match its provenance record")
    names = [planet.name for planet in planets]
    if names != sorted(names, key=str.casefold):
        raise ValueError("catalog rows must be sorted by planet name")
    if len(names) != len(set(names)):
        raise ValueError("catalog contains duplicate planet names")
    provenance = CatalogProvenance(
        source_name=_metadata_value(metadata, "source_name", str),
        source_url=_metadata_value(metadata, "source_url", str),
        source_table=_metadata_value(metadata, "source_table", str),
        query=_metadata_value(metadata, "query", str),
        retrieved_utc=_metadata_value(metadata, "retrieved_utc", str),
        row_count=row_count,
        sha256=expected_sha256,
        scope_note=_metadata_value(metadata, "scope_note", str),
    )
    return ExoplanetCatalog(provenance=provenance, planets=planets)
