import numpy as np
import pytest

from aerognc.configuration import load_rotating_ascent_configuration
from aerognc.simulation.rotating_ascent import RotatingAscentModel, simulate_rotating_ascent


def test_rotating_ascent_is_deterministic_and_detects_ordered_events() -> None:
    configuration = load_rotating_ascent_configuration("configs/rotating_planet_ascent.yaml")
    first = simulate_rotating_ascent(configuration)
    second = simulate_rotating_ascent(configuration)

    assert np.array_equal(first.time_s, second.time_s)
    for name in first.columns:
        assert np.array_equal(first.columns[name], second.columns[name])
    assert [event.name for event in first.events] == ["burnout", "apogee", "ground_impact"]
    assert first.maximum_summary["altitude"]["value"] == pytest.approx(1091.8, abs=2.0)
    assert (
        first.columns["mass_kg"].min()
        >= configuration.base_configuration.vehicle.mass_properties.dry_mass_kg
    )
    assert np.all(np.isfinite(first.columns["latitude_deg"]))
    assert np.all(np.isfinite(first.columns["longitude_deg"]))
    assert abs(first.columns["latitude_deg"][-1] - first.columns["latitude_deg"][0]) > 1.0e-6

    model = RotatingAscentModel(configuration)
    initial = model.diagnostics(0.0, model.initial_state())
    assert initial.geodetic.latitude_rad == pytest.approx(
        configuration.launch_site.geodetic.latitude_rad, abs=1.0e-12
    )
    assert initial.geodetic.longitude_rad == pytest.approx(
        configuration.launch_site.geodetic.longitude_rad, abs=1.0e-12
    )
