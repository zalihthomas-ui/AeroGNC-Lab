import numpy as np
import pytest

from aerognc.vehicle.propulsion import ThrustCurve
from aerognc.vehicle.staging import MultistageVehicle, StageDefinition


def _motor(thrust_n: float, propellant_kg: float, duration_s: float = 2.0) -> ThrustCurve:
    return ThrustCurve(
        [0.0, 0.2, duration_s - 0.2, duration_s],
        [0.0, thrust_n, thrust_n, 0.0],
        propellant_kg,
    )


def test_ordered_staging_reports_events_thrust_and_exact_jettison() -> None:
    vehicle = MultistageVehicle(
        payload_mass_kg=4.0,
        stages=(
            StageDefinition("booster", 3.0, _motor(100.0, 2.0), 0.0, 2.2),
            StageDefinition("sustainer", 1.5, _motor(40.0, 1.0), 2.2),
        ),
    )

    assert vehicle.mass_kg(-1.0) == pytest.approx(11.5)
    assert vehicle.active_stage_name(1.0) == "booster"
    assert vehicle.active_stage_name(3.0) == "sustainer"
    assert vehicle.thrust_n(1.0) == pytest.approx(100.0)
    assert vehicle.mass_kg(np.nextafter(2.2, -np.inf)) - vehicle.mass_kg(2.2) == pytest.approx(3.0)
    assert [(event.stage_name, event.kind) for event in vehicle.events()] == [
        ("booster", "ignition"),
        ("booster", "burnout"),
        ("booster", "separation"),
        ("sustainer", "ignition"),
        ("sustainer", "burnout"),
    ]
    assert vehicle.continuity_report().passed


def test_staging_rejects_overlap_and_nonzero_endpoint_thrust() -> None:
    with pytest.raises(ValueError, match="start at zero"):
        StageDefinition("bad", 1.0, ThrustCurve([0.0, 1.0], [1.0, 0.0], 1.0), 0.0)
    first = StageDefinition("one", 1.0, _motor(10.0, 1.0), 0.0, 2.0)
    overlapping = StageDefinition("two", 1.0, _motor(10.0, 1.0), 1.0)
    with pytest.raises(ValueError, match="ordered"):
        MultistageVehicle(payload_mass_kg=1.0, stages=(first, overlapping))
