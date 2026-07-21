import numpy as np
import pytest

from aerognc.configuration import load_interplanetary_configuration
from aerognc.simulation.interplanetary import simulate_interplanetary


@pytest.fixture(scope="module")
def mission():  # type: ignore[no-untyped-def]
    configuration = load_interplanetary_configuration("configs/interplanetary_gravity_assist.yaml")
    return simulate_interplanetary(configuration)


def test_gravity_assist_mission_encounters_assist_and_reaches_destination(mission) -> None:  # type: ignore[no-untyped-def]
    result = mission.result
    assert [event.name for event in result.events] == [
        "departure_injection",
        "assist_entry",
        "assist_closest_approach",
        "assist_exit",
        "destination_arrival",
        "mission_end",
    ]
    closest_assist_m = float(result.maximum_summary["assist_closest_approach"]["value"])
    assert closest_assist_m > mission.configuration.body_with_role("assist").radius_m
    assert closest_assist_m < mission.configuration.assist_encounter_radius_m
    assert float(result.maximum_summary["assist_heliocentric_speed_gain"]["value"]) > 9_000.0
    assert float(result.maximum_summary["assist_central_energy_gain"]["value"]) > 1.0e8
    assert abs(float(result.maximum_summary["assist_relative_speed_change"]["value"])) < 2.0
    assert result.maximum_summary["destination_arrival"]["value"] == 1.0
    assert (
        float(result.maximum_summary["destination_closest_approach"]["value"])
        < mission.configuration.destination_arrival_radius_m
    )


def test_interplanetary_result_is_finite_ordered_and_reproducible(mission) -> None:  # type: ignore[no-untyped-def]
    repeated = simulate_interplanetary(mission.configuration).result
    assert np.all(np.diff(mission.result.time_s) > 0.0)
    assert np.all(np.isfinite(np.column_stack(tuple(mission.result.columns.values()))))
    np.testing.assert_array_equal(repeated.time_s, mission.result.time_s)
    for name in mission.result.columns:
        np.testing.assert_array_equal(repeated.columns[name], mission.result.columns[name])
