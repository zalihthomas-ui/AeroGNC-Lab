import json
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from aerognc.simulation.hil import (
    ActuatorCommandPayload,
    HilPacket,
    PacketType,
    PlantStatePayload,
    encode_packet,
)
from aerognc.simulation.udp_transport import (
    UdpPacketEndpoint,
    run_udp_loopback_demo,
    write_udp_loopback_report,
)


def _command_packet(sequence: int) -> HilPacket:
    return HilPacket(
        PacketType.ACTUATOR_COMMAND,
        sequence,
        0.01 * sequence,
        ActuatorCommandPayload(0.1, -0.2, 0.3).encode(),
    )


def test_udp_endpoint_accepts_only_expected_source_and_new_sequence() -> None:
    with (
        UdpPacketEndpoint(
            PacketType.ACTUATOR_COMMAND,
            receive_timeout_s=0.02,
            watchdog_timeout_s=0.05,
        ) as receiver,
        UdpPacketEndpoint(
            PacketType.PLANT_STATE,
            receive_timeout_s=0.02,
        ) as sender,
        UdpPacketEndpoint(
            PacketType.PLANT_STATE,
            receive_timeout_s=0.02,
        ) as rogue,
    ):
        receiver.set_allowed_source(sender.local_address)

        rogue.send(_command_packet(1), receiver.local_address)
        assert receiver.receive(0.0) is None
        assert receiver.statistics.source_rejections == 1

        sender.send(_command_packet(1), receiver.local_address)
        accepted = receiver.receive(0.01)
        assert accepted is not None
        assert ActuatorCommandPayload.decode(accepted.payload).pitch_command == pytest.approx(-0.2)

        sender.send(_command_packet(1), receiver.local_address)
        assert receiver.receive(0.02) is None
        assert receiver.statistics.stale_packets == 1


def test_udp_timeout_is_bounded_and_watchdog_fails_silent() -> None:
    with UdpPacketEndpoint(
        PacketType.ACTUATOR_COMMAND,
        receive_timeout_s=0.01,
        watchdog_timeout_s=0.02,
    ) as receiver:
        start = time.perf_counter()
        assert receiver.receive(0.0) is None
        elapsed_s = time.perf_counter() - start
        command = receiver.actuator_command_or_zero(0.1)

    assert elapsed_s < 0.5
    assert receiver.statistics.receive_timeouts == 1
    assert command == ActuatorCommandPayload(0.0, 0.0, 0.0)


def test_udp_configuration_rejects_hostnames_and_invalid_ports() -> None:
    with pytest.raises(ValueError, match="numeric IP"):
        UdpPacketEndpoint(PacketType.PLANT_STATE, bind_address=("localhost", 0))
    with pytest.raises(ValueError, match="port"):
        UdpPacketEndpoint(PacketType.PLANT_STATE, bind_address=("127.0.0.1", 70_000))
    with pytest.raises(ValueError, match="receive_timeout_s"):
        UdpPacketEndpoint(PacketType.PLANT_STATE, receive_timeout_s=0.0)


def test_udp_endpoint_rejects_wrong_type_and_invalid_crc() -> None:
    with (
        UdpPacketEndpoint(
            PacketType.ACTUATOR_COMMAND,
            receive_timeout_s=0.02,
        ) as receiver,
        UdpPacketEndpoint(
            PacketType.PLANT_STATE,
            receive_timeout_s=0.02,
        ) as sender,
    ):
        receiver.set_allowed_source(sender.local_address)
        state = PlantStatePayload(
            np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)
        )
        sender.send(
            HilPacket(PacketType.PLANT_STATE, 1, 0.0, state.encode()),
            receiver.local_address,
        )
        assert receiver.receive(0.0) is None
        assert receiver.statistics.wrong_type_packets == 1

        raw = bytearray(encode_packet(_command_packet(2)))
        raw[-1] ^= 0xFF
        sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender_socket.bind(("127.0.0.1", 0))
            source = sender_socket.getsockname()
            receiver.set_allowed_source((str(source[0]), int(source[1])))
            sender_socket.sendto(raw, receiver.local_address)
            assert receiver.receive(0.01) is None
            assert receiver.statistics.invalid_packets == 1
        finally:
            sender_socket.close()


def test_udp_localhost_loopback_is_reproducible_and_truthfully_scoped(tmp_path: Path) -> None:
    first = run_udp_loopback_demo(sample_count=20)
    second = run_udp_loopback_demo(sample_count=20)

    assert first.state_endpoint.packets_accepted == 20
    assert first.command_endpoint.packets_accepted == 20
    assert first.watchdog_activations == 0
    assert first.applied_command_checksum == pytest.approx(second.applied_command_checksum)
    assert first.localhost_only is True
    assert first.physical_hil_executed is False
    assert first.real_time_guarantee is False

    destination = write_udp_loopback_report(first, tmp_path / "udp.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["command_endpoint"]["packets_accepted"] == 20
    assert payload["physical_hil_executed"] is False
