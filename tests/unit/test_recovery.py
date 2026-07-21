import pytest

from aerognc.vehicle.recovery import RecoveryDevice, simulate_vertical_recovery


def _device() -> RecoveryDevice:
    return RecoveryDevice(
        trigger_time_s=1.0,
        deployment_delay_s=0.5,
        reefing_time_s=1.0,
        reefed_hold_time_s=2.0,
        inflation_time_s=1.5,
        reefed_area_m2=0.3,
        full_area_m2=1.2,
        drag_coefficient=1.4,
    )


def test_recovery_area_schedule_is_continuous_and_bounded() -> None:
    device = _device()
    assert device.drag_area_m2(1.5) == 0.0
    assert device.drag_area_m2(2.0) == pytest.approx(0.15)
    assert device.drag_area_m2(2.5) == pytest.approx(0.3)
    assert device.drag_area_m2(4.5) == pytest.approx(0.3)
    assert device.drag_area_m2(5.25) == pytest.approx(0.75)
    assert device.drag_area_m2(6.0) == pytest.approx(1.2)
    assert device.drag_force_down_n(6.0, 10.0, 1.0) < 0.0


def test_vertical_recovery_reports_opening_load_and_ground_contact() -> None:
    result = simulate_vertical_recovery(
        _device(),
        initial_altitude_m=500.0,
        initial_velocity_down_mps=8.0,
        mass_kg=8.0,
        step_s=0.02,
    )

    assert result.ground_contact.name == "ground_contact"
    assert result.altitude_m[-1] == pytest.approx(0.0, abs=1.0e-10)
    assert result.maximum_opening_load_n > 0.0
    assert result.touchdown_speed_mps < 12.0
    assert [event.name for event in result.events] == [
        "deployment_start",
        "reefed",
        "full_inflation_start",
        "fully_inflated",
    ]
