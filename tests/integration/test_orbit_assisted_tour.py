import json
from pathlib import Path

from aerognc.configuration.orbit_tour_loader import load_orbit_tour_configuration
from aerognc.simulation.orbit_assisted_tour import (
    simulate_orbit_assisted_tour,
    write_orbit_tour_results,
)
from aerognc.visualisation.orbit_assisted_tour import plot_orbit_assisted_tour

CONFIGURATION_PATH = Path("configs/orbit_assisted_tour.yaml")


def test_orbit_tour_runs_all_phases_and_writes_evidence(tmp_path: Path) -> None:
    configuration = load_orbit_tour_configuration(CONFIGURATION_PATH)
    simulation = simulate_orbit_assisted_tour(configuration)
    csv_path, report_path = write_orbit_tour_results(simulation, tmp_path)
    figure_path = plot_orbit_assisted_tour(simulation, tmp_path)

    assert simulation.assessment.all_pass
    assert set(simulation.result.columns["phase_code"]) == {0.0, 1.0, 2.0}
    assert simulation.result.time_s.size == (
        configuration.first_leg_samples
        + configuration.parking_orbit_samples
        + configuration.second_leg_samples
    )
    assert simulation.tour.total_delta_v_mps < configuration.maximum_total_delta_v_mps
    assert simulation.tour.final_mass_kg > configuration.minimum_final_mass_kg
    assert simulation.result.maximum_summary["destination_lambert_endpoint_error"]["value"] < 0.1
    assert csv_path.stat().st_size > 100_000
    assert figure_path.stat().st_size > 30_000
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["requirements"]["all_pass"] is True
    assert len(payload["burns"]) == 5
    assert payload["limitations"]
