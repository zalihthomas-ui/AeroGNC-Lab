"""Progressive pilot-input shaping and optional civilian stability assistance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np

from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    AircraftState,
    FixedWingFlightModel,
)

AircraftControlMode = Literal["direct", "stability_assisted"]
AIRCRAFT_CONTROL_MODES: tuple[AircraftControlMode, ...] = (
    "direct",
    "stability_assisted",
)


@dataclass(frozen=True, slots=True)
class ControlBindings:
    """Editable keyboard bindings for pilot and view actions."""

    roll_left: str = "left"
    roll_right: str = "right"
    pitch_up: str = "up"
    pitch_down: str = "down"
    yaw_left: str = "a"
    yaw_right: str = "d"
    throttle_up: str = "w"
    throttle_down: str = "s"
    rocket_assist: str = "r"
    wings_level: str = "l"
    trim_nose_down: str = ","
    trim_nose_up: str = "."

    def __post_init__(self) -> None:
        values = tuple(value.strip().casefold() for value in asdict(self).values())
        if any(not value for value in values):
            raise ValueError("aircraft control bindings cannot be empty")
        if len(set(values)) != len(values):
            raise ValueError("aircraft control bindings must be unique")
        for field_name, value in zip(self.__dataclass_fields__, values, strict=True):
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class PilotControlProfile:
    """Deterministic input response and assistance settings."""

    name: str = "Accessible stability-assisted"
    control_mode: AircraftControlMode = "stability_assisted"
    roll_sensitivity: float = 0.82
    pitch_sensitivity: float = 0.78
    yaw_sensitivity: float = 0.70
    input_expo: float = 0.30
    analog_deadzone: float = 0.08
    invert_pitch: bool = False
    keyboard_ramp_per_s: float = 2.8
    keyboard_recentering_per_s: float = 3.8
    throttle_rate_per_s: float = 0.35
    trim_rate_per_s: float = 0.16
    roll_rate_damping_s: float = 0.28
    pitch_rate_damping_s: float = 0.36
    yaw_rate_damping_s: float = 0.22
    bank_level_gain: float = 0.85
    pitch_hold_gain: float = 0.65
    bindings: ControlBindings = field(default_factory=ControlBindings)

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.roll_sensitivity,
                self.pitch_sensitivity,
                self.yaw_sensitivity,
                self.input_expo,
                self.analog_deadzone,
                self.keyboard_ramp_per_s,
                self.keyboard_recentering_per_s,
                self.throttle_rate_per_s,
                self.trim_rate_per_s,
                self.roll_rate_damping_s,
                self.pitch_rate_damping_s,
                self.yaw_rate_damping_s,
                self.bank_level_gain,
                self.pitch_hold_gain,
            ],
            dtype=np.float64,
        )
        if not self.name.strip() or not np.all(np.isfinite(values)):
            raise ValueError("pilot profile name and values must be nonempty and finite")
        if self.control_mode not in AIRCRAFT_CONTROL_MODES:
            raise ValueError(f"control mode must be one of {AIRCRAFT_CONTROL_MODES}")
        if np.any(values[:3] <= 0.0) or np.any(values[:3] > 2.0):
            raise ValueError("pilot axis sensitivity must lie in (0, 2]")
        if not 0.0 <= self.input_expo <= 0.9:
            raise ValueError("pilot input expo must lie in [0, 0.9]")
        if not 0.0 <= self.analog_deadzone < 0.5:
            raise ValueError("pilot analog deadzone must lie in [0, 0.5)")
        if np.any(values[5:9] <= 0.0):
            raise ValueError("keyboard, throttle, and trim rates must be positive")
        if np.any(values[9:] < 0.0):
            raise ValueError("stability-assist gains must be nonnegative")


def shape_pilot_axis(
    value: float,
    *,
    sensitivity: float,
    expo: float,
    deadzone: float,
) -> float:
    """Apply deadzone, normalized cubic expo, sensitivity, and hard bounds."""
    values = np.asarray([value, sensitivity, expo, deadzone], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("pilot-axis shaping values must be finite")
    if sensitivity <= 0.0 or not 0.0 <= expo <= 0.9 or not 0.0 <= deadzone < 1.0:
        raise ValueError("invalid pilot-axis shaping settings")
    clipped = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(clipped)
    if magnitude <= deadzone:
        return 0.0
    normalized = (magnitude - deadzone) / (1.0 - deadzone)
    curved = (1.0 - expo) * normalized + expo * normalized**3
    return float(np.clip(np.copysign(curved * sensitivity, clipped), -1.0, 1.0))


def _move_toward(current: float, target: float, maximum_delta: float) -> float:
    return float(current + np.clip(target - current, -maximum_delta, maximum_delta))


@dataclass(slots=True)
class VirtualPilotStick:
    """Progressive keyboard axes that ramp and recenter like a virtual joystick."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    pitch_trim: float = 0.0

    def clear(self) -> None:
        """Release every virtual flight axis while retaining deliberate trim."""
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def reset(self) -> None:
        """Release every virtual axis and restore neutral pitch trim."""
        self.clear()
        self.pitch_trim = 0.0

    def update(
        self,
        pressed_keys: set[str],
        duration_s: float,
        profile: PilotControlProfile,
    ) -> tuple[float, float, float]:
        """Advance keyboard axes and trim for one real-time input interval."""
        if not np.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("virtual-stick duration must be finite and nonnegative")
        bindings = profile.bindings
        target_roll = float(
            (bindings.roll_right in pressed_keys) - (bindings.roll_left in pressed_keys)
        )
        target_pitch = float(
            (bindings.pitch_up in pressed_keys) - (bindings.pitch_down in pressed_keys)
        )
        target_yaw = float(
            (bindings.yaw_right in pressed_keys) - (bindings.yaw_left in pressed_keys)
        )
        targets = (target_roll, target_pitch, target_yaw)
        current = (self.roll, self.pitch, self.yaw)
        updated: list[float] = []
        for axis, target in zip(current, targets, strict=True):
            rate = (
                profile.keyboard_ramp_per_s if target != 0.0 else profile.keyboard_recentering_per_s
            )
            updated.append(_move_toward(axis, target, rate * duration_s))
        self.roll, self.pitch, self.yaw = updated
        trim_direction = float(
            (bindings.trim_nose_up in pressed_keys) - (bindings.trim_nose_down in pressed_keys)
        )
        self.pitch_trim = float(
            np.clip(
                self.pitch_trim + trim_direction * profile.trim_rate_per_s * duration_s,
                -0.5,
                0.5,
            )
        )
        return self.roll, self.pitch, self.yaw


def shape_pilot_command(
    raw_command: AircraftControlCommand,
    profile: PilotControlProfile,
) -> AircraftControlCommand:
    """Shape normalized pilot axes without modifying throttle or rocket state."""
    pitch_sign = -1.0 if profile.invert_pitch else 1.0
    return AircraftControlCommand(
        shape_pilot_axis(
            raw_command.roll,
            sensitivity=profile.roll_sensitivity,
            expo=profile.input_expo,
            deadzone=profile.analog_deadzone,
        ),
        pitch_sign
        * shape_pilot_axis(
            raw_command.pitch,
            sensitivity=profile.pitch_sensitivity,
            expo=profile.input_expo,
            deadzone=profile.analog_deadzone,
        ),
        shape_pilot_axis(
            raw_command.yaw,
            sensitivity=profile.yaw_sensitivity,
            expo=profile.input_expo,
            deadzone=profile.analog_deadzone,
        ),
        raw_command.throttle,
        raw_command.rocket_assist,
    )


def apply_stability_assist(
    model: FixedWingFlightModel,
    state: AircraftState,
    command: AircraftControlCommand,
    profile: PilotControlProfile,
    *,
    pitch_trim: float = 0.0,
    wings_level: bool = False,
) -> AircraftControlCommand:
    """Apply bounded rate damping and neutral-stick attitude stabilization."""
    trimmed_pitch_command = float(np.clip(command.pitch + pitch_trim, -1.0, 1.0))
    if profile.control_mode == "direct":
        return AircraftControlCommand(
            command.roll,
            trimmed_pitch_command,
            command.yaw,
            command.throttle,
            command.rocket_assist,
        )
    roll, pitch, _heading = model.local_attitude_rad(state.as_array())
    p_rate, q_rate, r_rate = state.angular_rate_body_radps
    roll_command = command.roll - profile.roll_rate_damping_s * p_rate
    pitch_command = trimmed_pitch_command - profile.pitch_rate_damping_s * q_rate
    yaw_command = command.yaw - profile.yaw_rate_damping_s * r_rate
    if wings_level or abs(command.roll) < 0.04:
        roll_command -= profile.bank_level_gain * roll
    if abs(command.pitch) < 0.04:
        reference_pitch = (
            model.configuration.initial.flight_path_angle_rad
            + model.configuration.initial.angle_of_attack_rad
        )
        pitch_command += profile.pitch_hold_gain * (reference_pitch - pitch)
    return AircraftControlCommand(
        float(np.clip(roll_command, -1.0, 1.0)),
        float(np.clip(pitch_command, -1.0, 1.0)),
        float(np.clip(yaw_command, -1.0, 1.0)),
        command.throttle,
        command.rocket_assist,
    )


@dataclass(slots=True)
class TriggerThrottleOwnership:
    """Retain controller-throttle ownership after the right trigger is first used."""

    claimed: bool = False

    def update(self, trigger_value: int, *, connected: bool = True) -> float | None:
        """Return no override before first use, then a stable normalized throttle."""
        if not connected:
            self.claimed = False
            return None
        if not 0 <= trigger_value <= 255:
            raise ValueError("controller trigger value must lie in [0, 255]")
        if trigger_value > 8:
            self.claimed = True
        if not self.claimed:
            return None
        return trigger_value / 255.0


def write_pilot_profile(profile: PilotControlProfile, path: str | Path) -> Path:
    """Write one versioned local control profile as readable JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", **asdict(profile)}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_pilot_profile(path: str | Path) -> PilotControlProfile:
    """Load a strict version-1 local control profile."""
    payload: object = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.pop("schema_version", None) != "1.0":
        raise ValueError("pilot profile must use schema_version 1.0")
    bindings_payload = payload.pop("bindings", None)
    if not isinstance(bindings_payload, dict):
        raise ValueError("pilot profile bindings must be an object")
    expected = set(PilotControlProfile.__dataclass_fields__) - {"bindings"}
    if set(payload) != expected or set(bindings_payload) != set(
        ControlBindings.__dataclass_fields__
    ):
        raise ValueError("pilot profile fields do not match the version-1 schema")
    typed_payload = cast(Mapping[str, object], payload)
    typed_bindings = cast(Mapping[str, object], bindings_payload)

    def text_value(data: Mapping[str, object], key: str) -> str:
        value = data[key]
        if not isinstance(value, str):
            raise ValueError(f"pilot profile {key} must be text")
        return value

    def number_value(key: str) -> float:
        value = typed_payload[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"pilot profile {key} must be a number")
        return float(value)

    def boolean_value(key: str) -> bool:
        value = typed_payload[key]
        if not isinstance(value, bool):
            raise ValueError(f"pilot profile {key} must be true or false")
        return value

    mode_value = text_value(typed_payload, "control_mode")
    if mode_value not in AIRCRAFT_CONTROL_MODES:
        raise ValueError(f"control mode must be one of {AIRCRAFT_CONTROL_MODES}")
    return PilotControlProfile(
        name=text_value(typed_payload, "name"),
        control_mode=mode_value,
        roll_sensitivity=number_value("roll_sensitivity"),
        pitch_sensitivity=number_value("pitch_sensitivity"),
        yaw_sensitivity=number_value("yaw_sensitivity"),
        input_expo=number_value("input_expo"),
        analog_deadzone=number_value("analog_deadzone"),
        invert_pitch=boolean_value("invert_pitch"),
        keyboard_ramp_per_s=number_value("keyboard_ramp_per_s"),
        keyboard_recentering_per_s=number_value("keyboard_recentering_per_s"),
        throttle_rate_per_s=number_value("throttle_rate_per_s"),
        trim_rate_per_s=number_value("trim_rate_per_s"),
        roll_rate_damping_s=number_value("roll_rate_damping_s"),
        pitch_rate_damping_s=number_value("pitch_rate_damping_s"),
        yaw_rate_damping_s=number_value("yaw_rate_damping_s"),
        bank_level_gain=number_value("bank_level_gain"),
        pitch_hold_gain=number_value("pitch_hold_gain"),
        bindings=ControlBindings(
            roll_left=text_value(typed_bindings, "roll_left"),
            roll_right=text_value(typed_bindings, "roll_right"),
            pitch_up=text_value(typed_bindings, "pitch_up"),
            pitch_down=text_value(typed_bindings, "pitch_down"),
            yaw_left=text_value(typed_bindings, "yaw_left"),
            yaw_right=text_value(typed_bindings, "yaw_right"),
            throttle_up=text_value(typed_bindings, "throttle_up"),
            throttle_down=text_value(typed_bindings, "throttle_down"),
            rocket_assist=text_value(typed_bindings, "rocket_assist"),
            wings_level=text_value(typed_bindings, "wings_level"),
            trim_nose_down=text_value(typed_bindings, "trim_nose_down"),
            trim_nose_up=text_value(typed_bindings, "trim_nose_up"),
        ),
    )
