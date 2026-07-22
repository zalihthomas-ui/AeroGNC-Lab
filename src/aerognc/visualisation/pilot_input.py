"""Optional Windows XInput adapter for the interactive aircraft sandbox."""

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from aerognc.visualisation.aircraft_controls import TriggerThrottleOwnership

XINPUT_GAMEPAD_A = 0x1000
XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE = 7_849
XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE = 8_689


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_ushort),
        ("left_trigger", ctypes.c_ubyte),
        ("right_trigger", ctypes.c_ubyte),
        ("left_thumb_x", ctypes.c_short),
        ("left_thumb_y", ctypes.c_short),
        ("right_thumb_x", ctypes.c_short),
        ("right_thumb_y", ctypes.c_short),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [("packet_number", ctypes.c_ulong), ("gamepad", _XInputGamepad)]


class _GetStateFunction(Protocol):
    def __call__(self, user_index: int, state: object) -> int: ...


@dataclass(frozen=True, slots=True)
class GamepadSnapshot:
    """Normalized gamepad inputs; throttle is absent until RT is intentionally used."""

    connected: bool
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float | None = None
    rocket_assist: bool = False


def normalized_stick(value: int, deadzone: int) -> float:
    """Map a signed XInput thumb value to [-1, 1] with a radial-axis deadzone."""
    if deadzone < 0 or deadzone >= 32_767:
        raise ValueError("deadzone must lie in [0, 32767)")
    clipped = int(np.clip(value, -32_768, 32_767))
    magnitude = abs(clipped)
    if magnitude <= deadzone:
        return 0.0
    normalized = (magnitude - deadzone) / (32_767 - deadzone)
    return float(np.copysign(min(normalized, 1.0), clipped))


class XInputGamepad:
    """Best-effort controller 0 adapter; unavailable platforms fail without blocking flight."""

    def __init__(self, user_index: int = 0) -> None:
        if isinstance(user_index, bool) or not 0 <= user_index <= 3:
            raise ValueError("XInput user_index must lie in [0, 3]")
        self.user_index = user_index
        self._get_state: _GetStateFunction | None = None
        self._throttle_ownership = TriggerThrottleOwnership()
        self.backend_name = "unavailable"
        if platform.system() != "Windows":
            return
        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            return
        for library_name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
            try:
                library = loader(library_name)
            except OSError:
                continue
            function = library.XInputGetState
            function.argtypes = [ctypes.c_ulong, ctypes.POINTER(_XInputState)]
            function.restype = ctypes.c_ulong
            self._get_state = cast(_GetStateFunction, function)
            self.backend_name = library_name
            break

    @property
    def available(self) -> bool:
        """Return whether an XInput library was loaded (connection is checked per poll)."""
        return self._get_state is not None

    def poll(self) -> GamepadSnapshot:
        """Read controller zero once; return a disconnected snapshot on any OS/API error."""
        if self._get_state is None:
            self._throttle_ownership.update(0, connected=False)
            return GamepadSnapshot(False)
        state = _XInputState()
        try:
            status = self._get_state(self.user_index, ctypes.byref(state))
        except (OSError, ValueError):
            self._throttle_ownership.update(0, connected=False)
            return GamepadSnapshot(False)
        if status != 0:
            self._throttle_ownership.update(0, connected=False)
            return GamepadSnapshot(False)
        gamepad = state.gamepad
        right_trigger = int(gamepad.right_trigger)
        throttle = self._throttle_ownership.update(right_trigger)
        return GamepadSnapshot(
            True,
            normalized_stick(int(gamepad.left_thumb_x), XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE),
            -normalized_stick(int(gamepad.left_thumb_y), XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE),
            normalized_stick(int(gamepad.right_thumb_x), XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE),
            throttle,
            bool(int(gamepad.buttons) & XINPUT_GAMEPAD_A),
        )
