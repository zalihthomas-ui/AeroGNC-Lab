"""Hardware-independent packets and an impairment emulator for future HIL work.

This module performs no serial or UDP I/O. It defines a versioned binary boundary
between controller and plant processes so timing requirements can be measured before
hardware is selected.
"""

from __future__ import annotations

import heapq
import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from aerognc.mathematics.vectors import FloatArray, as_vector

MAGIC = b"AGNC"
PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 4096
HEADER = struct.Struct("<4sBBHId")
CHECKSUM = struct.Struct("<I")
PLANT_STATE = struct.Struct("<13d")
ACTUATOR_COMMAND = struct.Struct("<3d")
SEQUENCE_MODULUS = 1 << 32
SEQUENCE_HALF_RANGE = 1 << 31


class PacketType(IntEnum):
    """Version-one future plant/controller message identifiers."""

    PLANT_STATE = 1
    ACTUATOR_COMMAND = 2
    HEARTBEAT = 3


@dataclass(frozen=True, slots=True)
class HilPacket:
    """Protocol packet before transport framing."""

    packet_type: PacketType
    sequence: int
    timestamp_s: float
    payload: bytes

    def __post_init__(self) -> None:
        if not 0 <= self.sequence <= 0xFFFFFFFF:
            raise ValueError("sequence must fit an unsigned 32-bit integer")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")


@dataclass(frozen=True, slots=True)
class PlantStatePayload:
    """NED/FRD 13-state plant message, matching the rigid-body state convention."""

    position_ned_m: FloatArray
    velocity_ned_mps: FloatArray
    attitude_quaternion_nb: FloatArray
    angular_velocity_body_radps: FloatArray

    def __post_init__(self) -> None:
        position = as_vector(self.position_ned_m, 3, name="position_ned_m")
        velocity = as_vector(self.velocity_ned_mps, 3, name="velocity_ned_mps")
        quaternion = as_vector(self.attitude_quaternion_nb, 4, name="attitude_quaternion_nb")
        angular_velocity = as_vector(
            self.angular_velocity_body_radps, 3, name="angular_velocity_body_radps"
        )
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-10):
            raise ValueError("attitude_quaternion_nb must have unit norm")
        object.__setattr__(self, "position_ned_m", position.copy())
        object.__setattr__(self, "velocity_ned_mps", velocity.copy())
        object.__setattr__(self, "attitude_quaternion_nb", quaternion.copy())
        object.__setattr__(self, "angular_velocity_body_radps", angular_velocity.copy())

    def encode(self) -> bytes:
        """Encode 13 little-endian IEEE-754 float64 values."""
        values = np.concatenate(
            (
                self.position_ned_m,
                self.velocity_ned_mps,
                self.attitude_quaternion_nb,
                self.angular_velocity_body_radps,
            )
        )
        return PLANT_STATE.pack(*values)

    @classmethod
    def decode(cls, payload: bytes) -> PlantStatePayload:
        """Decode and validate a plant-state payload."""
        if len(payload) != PLANT_STATE.size:
            raise ValueError(f"plant-state payload must be {PLANT_STATE.size} bytes")
        values = np.asarray(PLANT_STATE.unpack(payload), dtype=np.float64)
        return cls(values[0:3], values[3:6], values[6:10], values[10:13])


@dataclass(frozen=True, slots=True)
class ActuatorCommandPayload:
    """Fictional vehicle roll, pitch, and yaw commands, each normalised to [-1, 1]."""

    roll_command: float
    pitch_command: float
    yaw_command: float

    def __post_init__(self) -> None:
        command = np.asarray(
            [self.roll_command, self.pitch_command, self.yaw_command], dtype=np.float64
        )
        if not np.all(np.isfinite(command)) or np.any(np.abs(command) > 1.0):
            raise ValueError("actuator commands must be finite and within [-1, 1]")

    def encode(self) -> bytes:
        """Encode three normalised little-endian float64 commands."""
        return ACTUATOR_COMMAND.pack(self.roll_command, self.pitch_command, self.yaw_command)

    @classmethod
    def decode(cls, payload: bytes) -> ActuatorCommandPayload:
        """Decode and validate an actuator-command payload."""
        if len(payload) != ACTUATOR_COMMAND.size:
            raise ValueError(f"actuator-command payload must be {ACTUATOR_COMMAND.size} bytes")
        return cls(*ACTUATOR_COMMAND.unpack(payload))


def encode_packet(packet: HilPacket) -> bytes:
    """Encode a packet with header and CRC-32 integrity check."""
    header = HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(packet.packet_type),
        len(packet.payload),
        packet.sequence,
        packet.timestamp_s,
    )
    protected = header + packet.payload
    return protected + CHECKSUM.pack(zlib.crc32(protected) & 0xFFFFFFFF)


def decode_packet(data: bytes) -> HilPacket:
    """Decode a complete datagram/frame and reject malformed or corrupt data."""
    minimum_length = HEADER.size + CHECKSUM.size
    if len(data) < minimum_length:
        raise ValueError("packet is shorter than the protocol header and checksum")
    magic, version, packet_type, payload_length, sequence, timestamp_s = HEADER.unpack_from(data)
    expected_length = HEADER.size + payload_length + CHECKSUM.size
    if len(data) != expected_length:
        raise ValueError(f"packet length is {len(data)} bytes; header declares {expected_length}")
    if magic != MAGIC:
        raise ValueError("packet magic is invalid")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    protected = data[: -CHECKSUM.size]
    expected_checksum = CHECKSUM.unpack_from(data, len(protected))[0]
    actual_checksum = zlib.crc32(protected) & 0xFFFFFFFF
    if actual_checksum != expected_checksum:
        raise ValueError("packet CRC-32 check failed")
    try:
        kind = PacketType(packet_type)
    except ValueError as error:
        raise ValueError(f"unknown packet type {packet_type}") from error
    return HilPacket(
        packet_type=kind,
        sequence=sequence,
        timestamp_s=timestamp_s,
        payload=data[HEADER.size : -CHECKSUM.size],
    )


def sequence_is_newer(candidate: int, reference: int) -> bool:
    """Compare unsigned 32-bit sequence numbers across one wrap boundary.

    The exactly half-range case is deliberately treated as ambiguous/not newer.  A
    receiver should be reset if endpoints can become separated by 2^31 packets.
    """
    for value, name in ((candidate, "candidate"), (reference, "reference")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} sequence must be an integer")
        if not 0 <= value < SEQUENCE_MODULUS:
            raise ValueError(f"{name} sequence must fit an unsigned 32-bit integer")
    difference = (candidate - reference) % SEQUENCE_MODULUS
    return 0 < difference < SEQUENCE_HALF_RANGE


class PacketReceiver:
    """Fail-silent packet gate with type, sequence, and local-time watchdog checks."""

    def __init__(self, expected_type: PacketType, *, timeout_s: float) -> None:
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("receiver timeout_s must be finite and positive")
        self.expected_type = expected_type
        self.timeout_s = float(timeout_s)
        self.last_sequence: int | None = None
        self.last_receive_time_s: float | None = None
        self.accepted_count = 0
        self.invalid_count = 0
        self.wrong_type_count = 0
        self.stale_count = 0
        self.last_rejection_reason: str | None = None

    @staticmethod
    def _local_time(current_time_s: float) -> float:
        if not np.isfinite(current_time_s) or current_time_s < 0.0:
            raise ValueError("receiver current_time_s must be finite and nonnegative")
        return float(current_time_s)

    def accept(self, data: bytes, current_time_s: float) -> HilPacket | None:
        """Return a valid newer expected packet, otherwise update rejection counters."""
        receive_time_s = self._local_time(current_time_s)
        try:
            packet = decode_packet(data)
        except ValueError as error:
            self.invalid_count += 1
            self.last_rejection_reason = str(error)
            return None
        if packet.packet_type != self.expected_type:
            self.wrong_type_count += 1
            self.last_rejection_reason = (
                f"expected {self.expected_type.name}, received {packet.packet_type.name}"
            )
            return None
        if self.last_sequence is not None and not sequence_is_newer(
            packet.sequence, self.last_sequence
        ):
            self.stale_count += 1
            self.last_rejection_reason = "duplicate, stale, or ambiguous sequence"
            return None
        self.last_sequence = packet.sequence
        self.last_receive_time_s = receive_time_s
        self.accepted_count += 1
        self.last_rejection_reason = None
        return packet

    def is_timed_out(self, current_time_s: float) -> bool:
        """Return true before first acceptance or after the configured silence time."""
        checked_time_s = self._local_time(current_time_s)
        return (
            self.last_receive_time_s is None
            or checked_time_s - self.last_receive_time_s > self.timeout_s
        )


@dataclass(frozen=True, slots=True)
class LinkImpairmentConfiguration:
    """Seeded software-only transport impairment settings."""

    latency_s: float = 0.0
    jitter_standard_deviation_s: float = 0.0
    loss_probability: float = 0.0
    duplicate_probability: float = 0.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise ValueError("latency_s must be finite and nonnegative")
        if (
            not np.isfinite(self.jitter_standard_deviation_s)
            or self.jitter_standard_deviation_s < 0.0
        ):
            raise ValueError("jitter_standard_deviation_s must be finite and nonnegative")
        if not np.isfinite(self.loss_probability) or not 0.0 <= self.loss_probability <= 1.0:
            raise ValueError("loss_probability must be in [0, 1]")
        if (
            not np.isfinite(self.duplicate_probability)
            or not 0.0 <= self.duplicate_probability <= 1.0
        ):
            raise ValueError("duplicate_probability must be in [0, 1]")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")


@dataclass(order=True, slots=True)
class _QueuedFrame:
    delivery_time_s: float
    insertion_order: int
    data: bytes = field(compare=False)


class EmulatedLink:
    """Deterministic in-memory latency, Gaussian-jitter, and packet-loss emulator."""

    def __init__(self, configuration: LinkImpairmentConfiguration) -> None:
        self.configuration = configuration
        self._random = np.random.default_rng(configuration.random_seed)
        self._queue: list[_QueuedFrame] = []
        self._insertion_order = 0
        self.transmitted_count = 0
        self.dropped_count = 0
        self.duplicated_count = 0

    def _queue_frame(self, data: bytes, current_time_s: float) -> None:
        jitter_s = self._random.normal(0.0, self.configuration.jitter_standard_deviation_s)
        delay_s = max(0.0, self.configuration.latency_s + jitter_s)
        heapq.heappush(
            self._queue,
            _QueuedFrame(current_time_s + delay_s, self._insertion_order, bytes(data)),
        )
        self._insertion_order += 1

    def transmit(self, data: bytes, current_time_s: float) -> bool:
        """Schedule a frame, returning false when the configured link drops it."""
        if not np.isfinite(current_time_s) or current_time_s < 0.0:
            raise ValueError("current_time_s must be finite and nonnegative")
        self.transmitted_count += 1
        if self._random.random() < self.configuration.loss_probability:
            self.dropped_count += 1
            return False
        self._queue_frame(data, current_time_s)
        if (
            self.configuration.duplicate_probability > 0.0
            and self._random.random() < self.configuration.duplicate_probability
        ):
            self._queue_frame(data, current_time_s)
            self.duplicated_count += 1
        return True

    def receive_ready(self, current_time_s: float) -> tuple[bytes, ...]:
        """Return all frames due at or before the supplied emulation time."""
        if not np.isfinite(current_time_s) or current_time_s < 0.0:
            raise ValueError("current_time_s must be finite and nonnegative")
        ready: list[bytes] = []
        while self._queue and self._queue[0].delivery_time_s <= current_time_s:
            ready.append(heapq.heappop(self._queue).data)
        return tuple(ready)

    @property
    def pending_count(self) -> int:
        """Number of accepted frames awaiting their delivery time."""
        return len(self._queue)
