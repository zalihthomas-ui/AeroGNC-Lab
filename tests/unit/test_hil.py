import numpy as np
import pytest

from aerognc.simulation.hil import (
    ActuatorCommandPayload,
    EmulatedLink,
    HilPacket,
    LinkImpairmentConfiguration,
    PacketReceiver,
    PacketType,
    PlantStatePayload,
    decode_packet,
    encode_packet,
    sequence_is_newer,
)


def test_hil_packet_and_typed_payload_round_trip() -> None:
    state = PlantStatePayload(
        position_ned_m=np.array([1.0, 2.0, -3.0]),
        velocity_ned_mps=np.array([4.0, -5.0, 6.0]),
        attitude_quaternion_nb=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_body_radps=np.array([0.1, -0.2, 0.3]),
    )
    packet = HilPacket(PacketType.PLANT_STATE, 17, 1.25, state.encode())

    restored_packet = decode_packet(encode_packet(packet))
    restored_state = PlantStatePayload.decode(restored_packet.payload)

    assert restored_packet == packet
    np.testing.assert_array_equal(restored_state.position_ned_m, state.position_ned_m)
    np.testing.assert_array_equal(restored_state.velocity_ned_mps, state.velocity_ned_mps)
    np.testing.assert_array_equal(
        restored_state.attitude_quaternion_nb, state.attitude_quaternion_nb
    )
    np.testing.assert_array_equal(
        restored_state.angular_velocity_body_radps, state.angular_velocity_body_radps
    )


def test_command_payload_limits_and_packet_crc() -> None:
    command = ActuatorCommandPayload(0.25, -0.5, 1.0)
    assert ActuatorCommandPayload.decode(command.encode()) == command
    with pytest.raises(ValueError, match="within"):
        ActuatorCommandPayload(1.01, 0.0, 0.0)

    encoded = bytearray(
        encode_packet(HilPacket(PacketType.ACTUATOR_COMMAND, 1, 0.0, command.encode()))
    )
    encoded[-5] ^= 0x01
    with pytest.raises(ValueError, match="CRC"):
        decode_packet(bytes(encoded))


def test_emulated_link_latency_and_seeded_impairments_are_reproducible() -> None:
    fixed_link = EmulatedLink(LinkImpairmentConfiguration(latency_s=0.1))
    assert fixed_link.transmit(b"frame", 1.0)
    assert fixed_link.receive_ready(1.099) == ()
    assert fixed_link.receive_ready(1.1) == (b"frame",)

    configuration = LinkImpairmentConfiguration(
        latency_s=0.03,
        jitter_standard_deviation_s=0.01,
        loss_probability=0.25,
        random_seed=42,
    )
    links = (EmulatedLink(configuration), EmulatedLink(configuration))
    acceptance: list[list[bool]] = [[], []]
    for index in range(30):
        for link_index, link in enumerate(links):
            acceptance[link_index].append(link.transmit(index.to_bytes(2, "little"), index * 0.005))
    assert acceptance[0] == acceptance[1]
    assert links[0].dropped_count == links[1].dropped_count
    assert links[0].receive_ready(10.0) == links[1].receive_ready(10.0)


def test_wrap_aware_sequence_order_and_receiver_watchdog() -> None:
    assert sequence_is_newer(0, 0xFFFFFFFF)
    assert sequence_is_newer(7, 6)
    assert not sequence_is_newer(6, 6)
    assert not sequence_is_newer(0xFFFFFFFF, 0)
    with pytest.raises(ValueError, match="unsigned"):
        sequence_is_newer(-1, 0)

    receiver = PacketReceiver(PacketType.HEARTBEAT, timeout_s=0.1)
    first = encode_packet(HilPacket(PacketType.HEARTBEAT, 0xFFFFFFFF, 1.0, b""))
    wrapped = encode_packet(HilPacket(PacketType.HEARTBEAT, 0, 1.01, b""))
    assert receiver.is_timed_out(0.0)
    assert receiver.accept(first, 1.0) is not None
    assert receiver.accept(first, 1.001) is None
    assert receiver.accept(wrapped, 1.01) is not None
    assert receiver.stale_count == 1
    assert not receiver.is_timed_out(1.1)
    assert receiver.is_timed_out(1.111)


def test_receiver_rejects_corrupt_and_wrong_type_frames() -> None:
    receiver = PacketReceiver(PacketType.ACTUATOR_COMMAND, timeout_s=0.1)
    assert receiver.accept(b"bad", 0.0) is None
    heartbeat = encode_packet(HilPacket(PacketType.HEARTBEAT, 0, 0.0, b""))
    assert receiver.accept(heartbeat, 0.0) is None
    assert receiver.invalid_count == 1
    assert receiver.wrong_type_count == 1


def test_emulated_link_duplicate_injection_is_seeded() -> None:
    configuration = LinkImpairmentConfiguration(
        latency_s=0.01,
        duplicate_probability=1.0,
        random_seed=3,
    )
    link = EmulatedLink(configuration)
    assert link.transmit(b"same", 0.0)
    assert link.duplicated_count == 1
    assert link.receive_ready(1.0) == (b"same", b"same")
