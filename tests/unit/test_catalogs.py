import shutil
from pathlib import Path

import numpy as np
import pytest

from aerognc.catalogs import (
    galactic_to_icrs_deg,
    icrs_to_galactic_deg,
    load_exoplanet_catalog,
    load_milky_way_metadata,
    load_solar_system_planets,
)

CATALOG_DIRECTORY = Path("data/catalogs")
CATALOG_CSV = CATALOG_DIRECTORY / "nasa_confirmed_exoplanets.csv"
CATALOG_METADATA = CATALOG_DIRECTORY / "nasa_confirmed_exoplanets.metadata.json"


def test_icrs_galactic_known_centre_and_round_trip() -> None:
    longitude_deg, latitude_deg = icrs_to_galactic_deg(266.4051, -28.936175)

    assert min(longitude_deg, 360.0 - longitude_deg) < 1.0e-3
    assert abs(latitude_deg) < 1.0e-3

    initial = (123.4, -42.5)
    galactic = icrs_to_galactic_deg(*initial)
    recovered = galactic_to_icrs_deg(*galactic)
    assert recovered == pytest.approx(initial, abs=2.0e-12)


@pytest.mark.parametrize(
    ("longitude_deg", "latitude_deg"),
    [(-1.0, 0.0), (360.0, 0.0), (0.0, -90.1), (0.0, 90.1)],
)
def test_coordinate_transform_rejects_invalid_angles(
    longitude_deg: float,
    latitude_deg: float,
) -> None:
    with pytest.raises(ValueError):
        icrs_to_galactic_deg(longitude_deg, latitude_deg)


def test_bundled_confirmed_exoplanet_snapshot_is_valid_and_searchable() -> None:
    catalog = load_exoplanet_catalog(CATALOG_CSV, CATALOG_METADATA)
    summary = catalog.summary()

    assert catalog.provenance.source_name == "NASA Exoplanet Archive"
    assert catalog.provenance.row_count == 6_324
    assert summary.planet_count == 6_324
    assert summary.host_count > 4_000
    assert summary.positioned_planet_count > 6_000
    assert summary.discovery_year_max == 2026
    trappist = catalog.search(text="trappist-1")
    assert [planet.name for planet in trappist] == [
        "TRAPPIST-1 b",
        "TRAPPIST-1 c",
        "TRAPPIST-1 d",
        "TRAPPIST-1 e",
        "TRAPPIST-1 f",
        "TRAPPIST-1 g",
        "TRAPPIST-1 h",
    ]
    nearby = catalog.search(maximum_distance_pc=5.0)
    assert nearby
    assert all(
        planet.system_distance_pc is not None and planet.system_distance_pc <= 5.0
        for planet in nearby
    )
    assert np.all(np.isfinite(trappist[0].galactic_position_pc()))


def test_catalog_checksum_detects_modified_snapshot(tmp_path: Path) -> None:
    modified = tmp_path / "catalog.csv"
    shutil.copyfile(CATALOG_CSV, modified)
    with modified.open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(ValueError, match="SHA-256"):
        load_exoplanet_catalog(modified, CATALOG_METADATA)


def test_catalog_filters_validate_bounds() -> None:
    catalog = load_exoplanet_catalog(CATALOG_CSV, CATALOG_METADATA)

    with pytest.raises(ValueError, match="distance"):
        catalog.search(maximum_distance_pc=0.0)
    with pytest.raises(ValueError, match="year"):
        catalog.search(minimum_discovery_year=2020, maximum_discovery_year=2010)
    with pytest.raises(ValueError, match="limit"):
        catalog.search(limit=0)


def test_milky_way_context_and_solar_system_planets_are_explicitly_scoped() -> None:
    galaxy = load_milky_way_metadata(CATALOG_DIRECTORY / "milky_way_metadata.yaml")
    planets = load_solar_system_planets(CATALOG_DIRECTORY / "solar_system_planets.csv")

    assert galaxy.name == "Milky Way"
    assert galaxy.star_count_lower_estimate == 100_000_000_000
    assert "not a complete" in galaxy.scope_note
    assert [planet.name for planet in planets] == [
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ]
