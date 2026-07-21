import json

import numpy as np
import pytest

from aerognc.configuration import load_aircraft_configuration
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    AircraftState,
    FixedWingFlightModel,
    aircraft_initial_state,
)
from aerognc.visualisation.aircraft_controls import (
    ControlBindings,
    PilotControlProfile,
    TriggerThrottleOwnership,
    VirtualPilotStick,
    apply_stability_assist,
    load_pilot_profile,
    shape_pilot_axis,
    shape_pilot_command,
    write_pilot_profile,
)


def test_axis_shaping_has_deadzone_expo_and_bounds() -> None:
    assert shape_pilot_axis(0.04, sensitivity=1.0, expo=0.3, deadzone=0.08) == 0.0
    assert 0.0 < shape_pilot_axis(0.5, sensitivity=1.0, expo=0.3, deadzone=0.08) < 0.5
    assert shape_pilot_axis(2.0, sensitivity=2.0, expo=0.0, deadzone=0.0) == 1.0


def test_virtual_keyboard_stick_ramps_recenters_and_retains_trim() -> None:
    profile = PilotControlProfile()
    stick = VirtualPilotStick()
    first = stick.update({"right", "."}, 0.1, profile)
    assert first == pytest.approx((0.28, 0.0, 0.0))
    assert stick.pitch_trim > 0.0
    stick.update(set(), 0.1, profile)
    assert stick.roll == 0.0
    retained_trim = stick.pitch_trim
    stick.clear()
    assert stick.pitch_trim == retained_trim
    stick.reset()
    assert stick.pitch_trim == 0.0


def test_shaping_and_assistance_are_bounded_and_direct_mode_preserves_axes() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    state = AircraftState.from_array(aircraft_initial_state(configuration))
    raw = AircraftControlCommand(0.7, -0.4, 0.5, 0.6, True)
    direct_profile = PilotControlProfile(control_mode="direct", input_expo=0.0, analog_deadzone=0.0)
    shaped = shape_pilot_command(raw, direct_profile)
    direct = apply_stability_assist(model, state, shaped, direct_profile, pitch_trim=0.1)
    assert direct.roll == pytest.approx(shaped.roll)
    assert direct.pitch == pytest.approx(shaped.pitch + 0.1)
    assert direct.yaw == pytest.approx(shaped.yaw)
    assisted = apply_stability_assist(model, state, raw, PilotControlProfile(), wings_level=True)
    assert np.all(np.abs([assisted.roll, assisted.pitch, assisted.yaw]) <= 1.0)
    assert assisted == apply_stability_assist(
        model, state, raw, PilotControlProfile(), wings_level=True
    )


def test_trigger_throttle_latches_until_disconnect() -> None:
    ownership = TriggerThrottleOwnership()
    assert ownership.update(0) is None
    assert ownership.update(128) == pytest.approx(128.0 / 255.0)
    assert ownership.update(0) == 0.0
    assert ownership.update(0, connected=False) is None
    assert ownership.update(0) is None


def test_control_bindings_normalize_and_reject_duplicates() -> None:
    assert ControlBindings(roll_left="LEFT").roll_left == "left"
    with pytest.raises(ValueError, match="unique"):
        ControlBindings(roll_left="q", roll_right="Q")


def test_pilot_profile_round_trip_and_strict_schema(tmp_path) -> None:
    profile = PilotControlProfile(name="Test", invert_pitch=True)
    path = write_pilot_profile(profile, tmp_path / "pilot.json")
    assert load_pilot_profile(path) == profile
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_pilot_profile(path)
