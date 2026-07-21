"""Bounded UDP packet transport for localhost-only software HIL exercises."""

from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aerognc.simulation.hil import (
    CHECKSUM,
    HEADER,
    MAX_PAYLOAD_BYTES,
    ActuatorCommandPayload,
    HilPacket,
    PacketReceiver,
    PacketType,
    PlantStatePayload,
    encode_packet,
)
from aerognc.simulation.software_loopback import (
    normalized_rate_damping_controller,
    synthetic_loopback_states,
)

Ipv4Address = tuple[str, int]


def _address(value: Ipv4Address, label: str, *, allow_zero_port: bool) -> Ipv4Address:
    host, port = value
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError(f"{label} host must be a numeric IP address") from error
    if parsed.version != 4:
        raise ValueError(f"{label} currently supports IPv4 only")
    minimum_port = 0 if allow_zero_port else 1
    if isinstance(port, bool) or not isinstance(port, int) or not minimum_port <= port <= 65535:
        raise ValueError(f"{label} port must lie in [{minimum_port}, 65535]")
    return str(parsed), port


@dataclass(frozen=True, slots=True)
class UdpTransportStatistics:
    """Observable endpoint counters including codec and sequence rejections."""

    datagrams_sent: int
    datagrams_received: int
    receive_timeouts: int
    source_rejections: int
    packets_accepted: int
    invalid_packets: int
    wrong_type_packets: int
    stale_packets: int


class UdpPacketEndpoint:
    """One typed, source-filtered UDP endpoint with a bounded receive timeout."""

    def __init__(
        self,
        expected_type: PacketType,
        *,
        bind_address: Ipv4Address = ("127.0.0.1", 0),
        allowed_source: Ipv4Address | None = None,
        receive_timeout_s: float = 0.1,
        watchdog_timeout_s: float = 0.2,
    ) -> None:
        if not np.isfinite(receive_timeout_s) or not 0.0 < receive_timeout_s <= 10.0:
            raise ValueError("UDP receive_timeout_s must lie in (0, 10] seconds")
        bind = _address(bind_address, "UDP bind address", allow_zero_port=True)
        self.expected_type = expected_type
        self.receiver = PacketReceiver(expected_type, timeout_s=watchdog_timeout_s)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind(bind)
            self._socket.settimeout(float(receive_timeout_s))
        except BaseException:
            self._socket.close()
            raise
        local = self._socket.getsockname()
        self.local_address: Ipv4Address = (str(local[0]), int(local[1]))
        self._allowed_source = (
            None
            if allowed_source is None
            else _address(allowed_source, "UDP allowed source", allow_zero_port=False)
        )
        self._latest_packet: HilPacket | None = None
        self._closed = False
        self._sent_count = 0
        self._received_count = 0
        self._timeout_count = 0
        self._source_rejection_count = 0

    def set_allowed_source(self, source: Ipv4Address) -> None:
        """Set or replace the exact accepted peer address."""
        self._allowed_source = _address(source, "UDP allowed source", allow_zero_port=False)

    def send(self, packet: HilPacket, destination: Ipv4Address) -> int:
        """Encode and send one complete datagram to an explicit numeric address."""
        if self._closed:
            raise RuntimeError("UDP endpoint is closed")
        target = _address(destination, "UDP destination", allow_zero_port=False)
        data = encode_packet(packet)
        sent = self._socket.sendto(data, target)
        if sent != len(data):
            raise OSError(f"UDP sent {sent} of {len(data)} packet bytes")
        self._sent_count += 1
        return sent

    def receive(self, current_time_s: float) -> HilPacket | None:
        """Receive once, enforce source/type/CRC/sequence, or return None on timeout/rejection."""
        if self._closed:
            raise RuntimeError("UDP endpoint is closed")
        maximum_datagram_bytes = HEADER.size + MAX_PAYLOAD_BYTES + CHECKSUM.size
        try:
            data, raw_source = self._socket.recvfrom(maximum_datagram_bytes + 1)
        except TimeoutError:
            self._timeout_count += 1
            return None
        self._received_count += 1
        source: Ipv4Address = (str(raw_source[0]), int(raw_source[1]))
        if self._allowed_source is not None and source != self._allowed_source:
            self._source_rejection_count += 1
            return None
        packet = self.receiver.accept(data, current_time_s)
        if packet is not None:
            self._latest_packet = packet
        return packet

    def actuator_command_or_zero(self, current_time_s: float) -> ActuatorCommandPayload:
        """Decode the latest fresh command or return the fail-silent zero command."""
        if self.expected_type != PacketType.ACTUATOR_COMMAND:
            raise RuntimeError("actuator watchdog is only valid for command endpoints")
        if self._latest_packet is None or self.receiver.is_timed_out(current_time_s):
            return ActuatorCommandPayload(0.0, 0.0, 0.0)
        return ActuatorCommandPayload.decode(self._latest_packet.payload)

    @property
    def statistics(self) -> UdpTransportStatistics:
        """Return a snapshot of transport and packet-gate counters."""
        return UdpTransportStatistics(
            self._sent_count,
            self._received_count,
            self._timeout_count,
            self._source_rejection_count,
            self.receiver.accepted_count,
            self.receiver.invalid_count,
            self.receiver.wrong_type_count,
            self.receiver.stale_count,
        )

    def close(self) -> None:
        """Release the local socket; repeated calls are harmless."""
        if not self._closed:
            self._socket.close()
            self._closed = True

    def __enter__(self) -> UdpPacketEndpoint:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class UdpLoopbackResult:
    """Local OS-socket evidence without physical hardware or real-time claims."""

    sample_count: int
    state_endpoint: UdpTransportStatistics
    command_endpoint: UdpTransportStatistics
    watchdog_activations: int
    applied_command_checksum: float
    localhost_only: bool = True
    physical_hil_executed: bool = False
    real_time_guarantee: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return deterministic counters and explicit scope declarations."""
        return {
            "model": "operating-system UDP localhost loopback; no physical HIL",
            "sample_count": self.sample_count,
            "state_endpoint": asdict(self.state_endpoint),
            "command_endpoint": asdict(self.command_endpoint),
            "watchdog_activations": self.watchdog_activations,
            "applied_command_checksum": self.applied_command_checksum,
            "localhost_only": self.localhost_only,
            "physical_hil_executed": self.physical_hil_executed,
            "real_time_guarantee": self.real_time_guarantee,
        }


def run_udp_loopback_demo(
    *,
    sample_count: int = 100,
    sample_period_s: float = 0.01,
    receive_timeout_s: float = 0.1,
    watchdog_timeout_s: float = 0.03,
) -> UdpLoopbackResult:
    """Exchange typed state/command datagrams synchronously over localhost."""
    states = synthetic_loopback_states(sample_count, sample_period_s)
    with (
        UdpPacketEndpoint(
            PacketType.ACTUATOR_COMMAND,
            receive_timeout_s=receive_timeout_s,
            watchdog_timeout_s=watchdog_timeout_s,
        ) as plant_endpoint,
        UdpPacketEndpoint(
            PacketType.PLANT_STATE,
            receive_timeout_s=receive_timeout_s,
            watchdog_timeout_s=watchdog_timeout_s,
        ) as controller_endpoint,
    ):
        plant_endpoint.set_allowed_source(controller_endpoint.local_address)
        controller_endpoint.set_allowed_source(plant_endpoint.local_address)
        checksum = 0.0
        watchdog_activations = 0
        for sequence, state in enumerate(states):
            logical_time_s = sequence * sample_period_s
            plant_endpoint.send(
                HilPacket(
                    PacketType.PLANT_STATE,
                    sequence & 0xFFFFFFFF,
                    logical_time_s,
                    state.encode(),
                ),
                controller_endpoint.local_address,
            )
            state_packet = controller_endpoint.receive(logical_time_s)
            if state_packet is not None:
                decoded_state = PlantStatePayload.decode(state_packet.payload)
                command = normalized_rate_damping_controller(decoded_state)
                controller_endpoint.send(
                    HilPacket(
                        PacketType.ACTUATOR_COMMAND,
                        sequence & 0xFFFFFFFF,
                        logical_time_s,
                        command.encode(),
                    ),
                    plant_endpoint.local_address,
                )
            plant_endpoint.receive(logical_time_s)
            watchdog_activations += int(plant_endpoint.receiver.is_timed_out(logical_time_s))
            applied = plant_endpoint.actuator_command_or_zero(logical_time_s)
            checksum += applied.roll_command + applied.pitch_command + applied.yaw_command
        return UdpLoopbackResult(
            sample_count,
            controller_endpoint.statistics,
            plant_endpoint.statistics,
            watchdog_activations,
            checksum,
        )


def write_udp_loopback_report(result: UdpLoopbackResult, path: str | Path) -> Path:
    """Write deterministic localhost counter evidence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
