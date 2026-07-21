import numpy as np

from aerognc.simulation.multistage_recovery import simulate_multistage_recovery
from aerognc.vehicle.propulsion import ThrustCurve
from aerognc.vehicle.recovery import RecoveryDevice
from aerognc.vehicle.staging import MultistageVehicle, StageDefinition


def _motor(thrust_n: float, propellant_mass_kg: float) -> ThrustCurve:
    return ThrustCurve(
        [0.0, 0.2, 1.8, 2.0],
        [0.0, thrust_n, thrust_n, 0.0],
        propellant_mass_kg,
    )


def test_multistage_ascent_separation_recovery_and_touchdown() -> None:
    vehicle = MultistageVehicle(
        payload_mass_kg=4.0,
        stages=(
            StageDefinition("booster", 3.0, _motor(190.0, 2.0), 0.0, 2.2),
            StageDefinition("sustainer", 1.5, _motor(90.0, 1.0), 2.2),
        ),
    )
    recovery = RecoveryDevice(
        trigger_time_s=5.0,
        deployment_delay_s=0.5,
        reefing_time_s=0.8,
        reefed_hold_time_s=1.0,
        inflation_time_s=1.2,
        reefed_area_m2=0.25,
        full_area_m2=1.0,
        drag_coefficient=1.4,
    )
    result = simulate_multistage_recovery(vehicle, recovery, step_s=0.01)
    event_names = [event.name for event in result.events]

    assert event_names[:3] == ["booster_ignition", "booster_burnout", "booster_separation"]
    assert "sustainer_ignition" in event_names
    assert "apogee" in event_names
    assert event_names[-1] == "ground_contact"
    assert result.maximum_summary["altitude"]["value"] > 50.0
    assert result.maximum_summary["opening_load"]["value"] > 0.0
    assert result.maximum_summary["dry_mass_margin"]["value"] >= -1.0e-10
    assert result.columns["altitude_m"][-1] == 0.0
    assert np.max(result.columns["recovery_drag_area_m2"]) == recovery.full_area_m2
