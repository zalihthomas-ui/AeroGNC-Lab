import pytest

from aerognc.visualisation.pilot_input import XInputGamepad, normalized_stick


def test_xinput_stick_normalization_applies_deadzone_and_bounds() -> None:
    assert normalized_stick(0, 7_849) == 0.0
    assert normalized_stick(7_849, 7_849) == 0.0
    assert normalized_stick(32_767, 7_849) == pytest.approx(1.0)
    assert normalized_stick(-32_768, 7_849) == pytest.approx(-1.0)
    assert 0.0 < normalized_stick(20_000, 7_849) < 1.0


def test_xinput_adapter_is_optional_and_returns_bounded_snapshot() -> None:
    snapshot = XInputGamepad().poll()

    assert isinstance(snapshot.connected, bool)
    assert -1.0 <= snapshot.roll <= 1.0
    assert -1.0 <= snapshot.pitch <= 1.0
    assert -1.0 <= snapshot.yaw <= 1.0
    assert snapshot.throttle is None or 0.0 <= snapshot.throttle <= 1.0


def test_xinput_rejects_invalid_index_and_deadzone() -> None:
    with pytest.raises(ValueError, match="user_index"):
        XInputGamepad(4)
    with pytest.raises(ValueError, match="deadzone"):
        normalized_stick(0, 32_767)
