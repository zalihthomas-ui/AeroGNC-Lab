"""Deterministic packet-level software loopback for future HIL preparation.

This is a logical-time software-in-the-loop exercise.  It performs no operating-
system networking and makes no real-time or physical-HIL claim.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.mathematics.vectors import FloatArray
from aerognc.simulation.hil import (
    ActuatorCommandPayload,
    EmulatedLink,
    HilPacket,
    LinkImpairmentConfiguration,
    PacketReceiver,
    PacketType,
    PlantStatePayload,
    encode_packet,
)


@dataclass(frozen=True, slots=True)
class SoftwareLoopbackConfiguration:
    """Logical scheduler, deadline, watchdog, and two independent link definitions."""

    sample_period_s: float = 0.01
    command_deadline_s: float = 0.02
    command_timeout_s: float = 0.03
    state_link: LinkImpairmentConfiguration = field(default_factory=LinkImpairmentConfiguration)
    command_link: LinkImpairmentConfiguration = field(default_factory=LinkImpairmentConfiguration)

    def __post_init__(self) -> None:
        for value, name in (
            (self.sample_period_s, "sample_period_s"),
            (self.command_deadline_s, "command_deadline_s"),
            (self.command_timeout_s, "command_timeout_s"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class SoftwareLoopbackResult:
    """Deterministic logical-time transport and watchdog evidence."""

    sample_count: int
    state_packets_accepted: int
    state_packets_dropped: int
    state_packets_duplicated: int
    state_packets_stale: int
    command_packets_accepted: int
    command_packets_dropped: int
    command_packets_duplicated: int
    command_packets_stale: int
    command_deadline_misses: int
    watchdog_activations: int
    pending_state_packets: int
    pending_command_packets: int
    mean_logical_latency_s: float
    p95_logical_latency_s: float
    maximum_logical_latency_s: float
    applied_command_checksum: float
    applied_commands: FloatArray

    def as_dict(self) -> dict[str, int | float | str]:
        """Return stable summary fields without the full command history."""
        return {
            "model": "deterministic logical-time software loopback; no physical HIL",
            "sample_count": self.sample_count,
            "state_packets_accepted": self.state_packets_accepted,
            "state_packets_dropped": self.state_packets_dropped,
            "state_packets_duplicated": self.state_packets_duplicated,
            "state_packets_stale": self.state_packets_stale,
            "command_packets_accepted": self.command_packets_accepted,
            "command_packets_dropped": self.command_packets_dropped,
            "command_packets_duplicated": self.command_packets_duplicated,
            "command_packets_stale": self.command_packets_stale,
            "command_deadline_misses": self.command_deadline_misses,
            "watchdog_activations": self.watchdog_activations,
            "pending_state_packets": self.pending_state_packets,
            "pending_command_packets": self.pending_command_packets,
            "mean_logical_latency_s": self.mean_logical_latency_s,
            "p95_logical_latency_s": self.p95_logical_latency_s,
            "maximum_logical_latency_s": self.maximum_logical_latency_s,
            "applied_command_checksum": self.applied_command_checksum,
        }


def run_software_loopback(
    plant_states: Sequence[PlantStatePayload],
    controller: Callable[[PlantStatePayload], ActuatorCommandPayload],
    configuration: SoftwareLoopbackConfiguration,
) -> SoftwareLoopbackResult:
    """Exchange typed packets and apply a zero-command fail-silent watchdog."""
    if not plant_states:
        raise ValueError("software loopback requires at least one plant state")
    state_link = EmulatedLink(configuration.state_link)
    command_link = EmulatedLink(configuration.command_link)
    state_receiver = PacketReceiver(
        PacketType.PLANT_STATE,
        timeout_s=configuration.command_timeout_s,
    )
    command_receiver = PacketReceiver(
        PacketType.ACTUATOR_COMMAND,
        timeout_s=configuration.command_timeout_s,
    )
    latest_command = ActuatorCommandPayload(0.0, 0.0, 0.0)
    logical_latencies_s: list[float] = []
    applied_commands = np.zeros((len(plant_states), 3), dtype=np.float64)
    command_sequence = 0
    command_deadline_misses = 0
    watchdog_activations = 0

    for state_sequence, state in enumerate(plant_states):
        current_time_s = state_sequence * configuration.sample_period_s
        state_frame = encode_packet(
            HilPacket(
                PacketType.PLANT_STATE,
                state_sequence & 0xFFFFFFFF,
                current_time_s,
                state.encode(),
            )
        )
        state_link.transmit(state_frame, current_time_s)

        for frame in state_link.receive_ready(current_time_s):
            packet = state_receiver.accept(frame, current_time_s)
            if packet is None:
                continue
            state_payload = PlantStatePayload.decode(packet.payload)
            command_payload = controller(state_payload)
            command_frame = encode_packet(
                HilPacket(
                    PacketType.ACTUATOR_COMMAND,
                    command_sequence & 0xFFFFFFFF,
                    packet.timestamp_s,
                    command_payload.encode(),
                )
            )
            command_link.transmit(command_frame, current_time_s)
            command_sequence += 1

        for frame in command_link.receive_ready(current_time_s):
            packet = command_receiver.accept(frame, current_time_s)
            if packet is None:
                continue
            latest_command = ActuatorCommandPayload.decode(packet.payload)
            latency_s = current_time_s - packet.timestamp_s
            logical_latencies_s.append(latency_s)
            command_deadline_misses += int(latency_s > configuration.command_deadline_s)

        if command_receiver.is_timed_out(current_time_s):
            applied = ActuatorCommandPayload(0.0, 0.0, 0.0)
            watchdog_activations += 1
        else:
            applied = latest_command
        applied_commands[state_sequence] = (
            applied.roll_command,
            applied.pitch_command,
            applied.yaw_command,
        )

    latency_array = np.asarray(logical_latencies_s, dtype=np.float64)
    if latency_array.size:
        mean_latency_s = float(np.mean(latency_array))
        p95_latency_s = float(np.percentile(latency_array, 95.0))
        maximum_latency_s = float(np.max(latency_array))
    else:
        mean_latency_s = p95_latency_s = maximum_latency_s = 0.0
    return SoftwareLoopbackResult(
        sample_count=len(plant_states),
        state_packets_accepted=state_receiver.accepted_count,
        state_packets_dropped=state_link.dropped_count,
        state_packets_duplicated=state_link.duplicated_count,
        state_packets_stale=state_receiver.stale_count,
        command_packets_accepted=command_receiver.accepted_count,
        command_packets_dropped=command_link.dropped_count,
        command_packets_duplicated=command_link.duplicated_count,
        command_packets_stale=command_receiver.stale_count,
        command_deadline_misses=command_deadline_misses,
        watchdog_activations=watchdog_activations,
        pending_state_packets=state_link.pending_count,
        pending_command_packets=command_link.pending_count,
        mean_logical_latency_s=mean_latency_s,
        p95_logical_latency_s=p95_latency_s,
        maximum_logical_latency_s=maximum_latency_s,
        applied_command_checksum=float(np.sum(applied_commands)),
        applied_commands=applied_commands,
    )


def synthetic_loopback_states(
    sample_count: int,
    sample_period_s: float,
) -> tuple[PlantStatePayload, ...]:
    """Build a deterministic, fictional bounded attitude/rate input sequence."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    states: list[PlantStatePayload] = []
    for index in range(sample_count):
        time_s = index * sample_period_s
        roll_rad = np.deg2rad(2.0 * np.sin(0.7 * time_s))
        pitch_rad = np.deg2rad(-1.5 * np.cos(0.5 * time_s))
        yaw_rad = np.deg2rad(0.8 * np.sin(0.3 * time_s))
        states.append(
            PlantStatePayload(
                position_ned_m=np.array([3.0 * time_s, 0.2 * time_s, -10.0 - time_s]),
                velocity_ned_mps=np.array([3.0, 0.2, -1.0]),
                attitude_quaternion_nb=euler321_to_quaternion(
                    float(roll_rad),
                    float(pitch_rad),
                    float(yaw_rad),
                ),
                angular_velocity_body_radps=np.array(
                    [
                        0.024 * np.cos(0.7 * time_s),
                        0.013 * np.sin(0.5 * time_s),
                        0.004 * np.cos(0.3 * time_s),
                    ]
                ),
            )
        )
    return tuple(states)


def normalized_rate_damping_controller(state: PlantStatePayload) -> ActuatorCommandPayload:
    """Small public-safe controller used only to exercise the transport boundary."""
    command = np.clip(-12.0 * state.angular_velocity_body_radps, -1.0, 1.0)
    return ActuatorCommandPayload(float(command[0]), float(command[1]), float(command[2]))


def run_software_loopback_demo(
    configuration: SoftwareLoopbackConfiguration,
    *,
    sample_count: int,
) -> SoftwareLoopbackResult:
    """Run the deterministic built-in transport/controller example."""
    states = synthetic_loopback_states(sample_count, configuration.sample_period_s)
    return run_software_loopback(states, normalized_rate_damping_controller, configuration)


def write_software_loopback_report(result: SoftwareLoopbackResult, path: str | Path) -> Path:
    """Write deterministic machine-readable software-loopback evidence."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
