from pathlib import Path

import pytest

from aerognc.catalogs import load_exoplanet_catalog
from aerognc.simulation.workbench import (
    CatalogWorkbenchInputs,
    OrbitTourWorkbenchInputs,
    RocketWorkbenchInputs,
    build_orbit_tour_configuration,
    build_rocket_configuration,
    run_orbit_tour_workbench,
    run_rocket_workbench,
    search_exoplanet_catalog,
)
from aerognc.visualisation.workbench import WorkbenchPaths

SIX_DOF = Path("configs/six_dof_nominal.yaml")
ORBIT_TOUR = Path("configs/orbit_assisted_tour.yaml")


def test_rocket_workbench_builds_and_runs_user_initial_condition() -> None:
    inputs = RocketWorkbenchInputs(
        duration_s=0.3,
        step_s=0.01,
        initial_speed_mps=12.0,
        initial_euler321_deg=(1.0, 85.0, 12.0),
        initial_angular_rate_body_degps=(0.1, 0.2, -0.1),
        playback_speed=4.0,
    )
    configuration, result = run_rocket_workbench(SIX_DOF, inputs)

    assert configuration.initial_speed_mps == 12.0
    assert result.time_s[-1] == pytest.approx(0.3)
    assert result.columns["north_m"].size == 31


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (RocketWorkbenchInputs(step_s=0.03), "integration step"),
        (RocketWorkbenchInputs(duration_s=9.0), "duration"),
        (RocketWorkbenchInputs(playback_speed=100.0), "playback speed"),
        (
            RocketWorkbenchInputs(initial_angular_rate_body_degps=(0.0, 100.0, 0.0)),
            "body rate",
        ),
    ],
)
def test_rocket_workbench_rejects_inputs_outside_supported_domain(
    inputs: RocketWorkbenchInputs,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_rocket_configuration(SIX_DOF, inputs)


def test_orbit_tour_workbench_builds_and_runs_verified_preset() -> None:
    inputs = OrbitTourWorkbenchInputs()
    configuration = build_orbit_tour_configuration(ORBIT_TOUR, inputs)
    simulation = run_orbit_tour_workbench(ORBIT_TOUR, inputs)

    assert configuration.assist_dwell_revolutions == 2
    assert simulation.assessment.all_pass
    assert simulation.tour.final_mass_kg >= inputs.minimum_final_mass_kg


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            OrbitTourWorkbenchInputs(destination_body="Neria"),
            "worlds must differ",
        ),
        (
            OrbitTourWorkbenchInputs(assist_arrival_day=2100.0),
            "mission days must increase",
        ),
        (
            OrbitTourWorkbenchInputs(dry_mass_kg=130_000.0),
            "dry mass",
        ),
        (
            OrbitTourWorkbenchInputs(minimum_final_mass_kg=1_000.0),
            "minimum final mass",
        ),
    ],
)
def test_orbit_tour_workbench_rejects_inconsistent_inputs(
    inputs: OrbitTourWorkbenchInputs,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_orbit_tour_configuration(ORBIT_TOUR, inputs)


def test_catalog_workbench_filters_verified_snapshot() -> None:
    root = Path("data/catalogs")
    catalog = load_exoplanet_catalog(
        root / "nasa_confirmed_exoplanets.csv",
        root / "nasa_confirmed_exoplanets.metadata.json",
    )
    selected = search_exoplanet_catalog(
        catalog,
        CatalogWorkbenchInputs(text="TRAPPIST-1", maximum_distance_pc=20.0, limit=10),
    )

    assert len(selected) == 7
    assert {planet.host_name for planet in selected} == {"TRAPPIST-1"}


def test_catalog_workbench_rejects_reversed_years() -> None:
    root = Path("data/catalogs")
    catalog = load_exoplanet_catalog(
        root / "nasa_confirmed_exoplanets.csv",
        root / "nasa_confirmed_exoplanets.metadata.json",
    )
    with pytest.raises(ValueError, match="must not precede"):
        search_exoplanet_catalog(
            catalog,
            CatalogWorkbenchInputs(minimum_discovery_year=2025, maximum_discovery_year=2020),
        )


def test_workbench_path_validation_lists_missing_resource(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    paths = WorkbenchPaths(*(missing for _index in range(8)))

    with pytest.raises(FileNotFoundError, match=r"missing\.yaml"):
        paths.validate()
