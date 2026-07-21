import json
from pathlib import Path

from aerognc.catalogs import (
    load_exoplanet_catalog,
    load_milky_way_metadata,
    load_solar_system_planets,
)
from aerognc.verification.galaxy_catalog import write_galaxy_catalog_outputs
from aerognc.visualisation.galaxy_catalog import plot_milky_way_catalog

CATALOG_DIRECTORY = Path("data/catalogs")


def test_catalog_workflow_writes_scoped_report_selection_and_figure(tmp_path: Path) -> None:
    catalog = load_exoplanet_catalog(
        CATALOG_DIRECTORY / "nasa_confirmed_exoplanets.csv",
        CATALOG_DIRECTORY / "nasa_confirmed_exoplanets.metadata.json",
    )
    galaxy = load_milky_way_metadata(CATALOG_DIRECTORY / "milky_way_metadata.yaml")
    solar_system = load_solar_system_planets(CATALOG_DIRECTORY / "solar_system_planets.csv")
    selection = catalog.search(maximum_distance_pc=50.0)
    report_path, selection_path = write_galaxy_catalog_outputs(
        catalog,
        selection,
        galaxy,
        solar_system,
        tmp_path,
    )
    figure_path = plot_milky_way_catalog(catalog, selection, tmp_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["scope"] == {
        "complete_milky_way_census": False,
        "note": catalog.provenance.scope_note,
        "observational_data": True,
        "simulation_ephemeris": False,
    }
    assert payload["snapshot_summary"]["planet_count"] == 6_324
    assert payload["selection_summary"]["planet_count"] == len(selection)
    assert len(payload["solar_system_planets"]) == 8
    assert selection_path.stat().st_size > 5_000
    assert figure_path.stat().st_size > 75_000
