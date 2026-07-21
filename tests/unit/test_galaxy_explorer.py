from pathlib import Path

import matplotlib
import pytest

from aerognc.catalogs import load_exoplanet_catalog
from aerognc.visualisation.galaxy_explorer import (
    GalaxyExplorer3D,
    build_exoplanet_host_points,
)

matplotlib.use("Agg")


def _trappist_selection():  # type: ignore[no-untyped-def]
    root = Path("data/catalogs")
    catalog = load_exoplanet_catalog(
        root / "nasa_confirmed_exoplanets.csv",
        root / "nasa_confirmed_exoplanets.metadata.json",
    )
    return catalog.search(text="TRAPPIST-1")


def test_host_points_collapse_multiplanet_system() -> None:
    points = build_exoplanet_host_points(_trappist_selection())

    assert len(points) == 1
    assert points[0].host_name == "TRAPPIST-1"
    assert len(points[0].planet_names) == 7


def test_galaxy_explorer_constructs_and_rejects_noninteractive_show() -> None:
    explorer = GalaxyExplorer3D(_trappist_selection())
    try:
        assert len(explorer.points) == 1
        with pytest.raises(RuntimeError, match="cannot open"):
            explorer.show()
    finally:
        explorer.close()


def test_galaxy_explorer_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="no rows"):
        GalaxyExplorer3D(())
