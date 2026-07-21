from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from aerognc.astrodynamics.maneuvers import ImpulsiveManeuver
from aerognc.astrodynamics.mission_design import compute_porkchop_grid
from aerognc.configuration.planetary_catalog import load_planetary_catalog
from aerognc.simulation.mission_planner import MissionPlanRequest, plan_mission
from aerognc.visualisation.mission_control import mission_phase
from aerognc.visualisation.mission_design import plot_porkchop_grid

CATALOG_PATH = Path("configs/fictional_planetary_system.yaml")


def test_catalog_and_direct_planner_reach_destination_and_prepare_playback() -> None:
    catalog = load_planetary_catalog(CATALOG_PATH)
    request = MissionPlanRequest("direct", "Asteria", "Neria", 0.0, 260.0, sample_count=120)

    plan = plan_mission(catalog, request)

    assert plan.metrics.feasible
    assert plan.metrics.destination_miss_distance_m < 1.0
    assert plan.metrics.departure_c3_m2_s2 > 0.0
    assert plan.mission.result.time_s.size == 120
    assert mission_phase(plan.mission, 100.0) == "DIRECT INTERPLANETARY TRANSFER"
    assert plan.mission.result.events[-1].name == "mission_end"


def test_manual_rtn_correction_changes_path_mass_and_event_timeline() -> None:
    catalog = load_planetary_catalog(CATALOG_PATH)
    correction = ImpulsiveManeuver("correction", 100.0 * 86_400.0, (0.0, 5.0, 0.0), "rtn", 450.0)
    nominal = plan_mission(
        catalog,
        MissionPlanRequest("direct", "Asteria", "Neria", 0.0, 260.0, sample_count=120),
    )
    corrected = plan_mission(
        catalog,
        MissionPlanRequest(
            "direct",
            "Asteria",
            "Neria",
            0.0,
            260.0,
            sample_count=120,
            maneuvers=(correction,),
        ),
    )

    assert (
        corrected.metrics.destination_miss_distance_m > nominal.metrics.destination_miss_distance_m
    )
    assert corrected.mission.result.columns["mass_kg"][-1] < corrected.request.initial_mass_kg
    assert "maneuver_correction" in {event.name for event in corrected.mission.result.events}


def test_porkchop_figure_contains_feasible_best_point() -> None:
    catalog = load_planetary_catalog(CATALOG_PATH)
    departure = catalog.body("Asteria", role="departure")
    destination = catalog.body("Neria", role="destination")
    grid = compute_porkchop_grid(
        departure,
        destination,
        catalog.primary.gravitational_parameter_m3_s2,
        np.linspace(0.0, 20.0, 4) * 86_400.0,
        np.linspace(240.0, 300.0, 5) * 86_400.0,
        maximum_c3_m2_s2=100.0e6,
        maximum_arrival_excess_speed_mps=20_000.0,
    )

    assert np.any(grid.feasible)
    figure = plot_porkchop_grid(grid, title="test")
    try:
        assert len(figure.axes) >= 2
    finally:
        figure.clear()
