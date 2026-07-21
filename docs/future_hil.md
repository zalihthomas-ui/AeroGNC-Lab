# Future HIL Preparation

No physical hardware-in-the-loop (HIL) test has been performed. AeroGNC-Lab keeps
the controller, plant, transport, and logging boundaries separate so a future HIL
campaign can be designed after timing and I/O requirements are measured.

## Software boundary

`aerognc.simulation.hil` defines a transport-neutral binary frame. It can be placed
inside a UDP datagram or escaped/framed for a serial byte stream without changing
the controller payloads. Version 1 uses little-endian IEEE-754 values.

| Field | Type | Bytes | Meaning |
|---|---:|---:|---|
| Magic | `char[4]` | 4 | ASCII `AGNC` |
| Protocol version | `uint8` | 1 | Currently 1 |
| Packet type | `uint8` | 1 | State, command, or heartbeat |
| Payload length | `uint16` | 2 | Bytes, maximum 4096 |
| Sequence | `uint32` | 4 | Wrap-aware transport counter |
| Timestamp | `float64` | 8 | Monotonic simulation/acquisition time, s |
| Payload | bytes | variable | Type-specific data |
| CRC-32 | `uint32` | 4 | Header and payload integrity check |

The plant-state payload follows the normative 13-state convention: NED position
(3), NED velocity (3), scalar-first body-to-NED quaternion (4), and FRD body angular
rate (3), all as `float64`. The actuator payload contains three fictional normalised
roll/pitch/yaw commands in ([-1,1]). Bad magic, version, length, type, checksum,
quaternion norm, or command range fails explicitly.

## Software-only impairment tests

`EmulatedLink` injects seeded latency, nonnegative Gaussian-jittered delay, packet
loss, and duplication. It deliberately performs no operating-system I/O, making
protocol and controller tests deterministic. Out-of-order delivery is possible when
jitter makes a later packet due first. `PacketReceiver` rejects corrupt, wrong-type,
duplicate, stale and ambiguous half-range sequence packets, handles unsigned 32-bit
wrap, and exposes a local receive-time watchdog. Clock-skew injection and serial
resynchronisation remain future work.

`software-loopback` passes sampled synthetic 13-state records through one impaired
state link, calls the separated rate-damping controller only after a valid decode,
passes its typed command through a second independently seeded link, and applies a
zero command whenever the receiver watchdog expires. It reports accepted, dropped,
duplicated, stale and pending frames; logical latency percentiles; deadline misses;
watchdog activations; and a deterministic applied-command checksum.

```powershell
python -m aerognc.cli software-loopback --samples 500 --seed 218
```

These are logical-time transport results. They are reproducible protocol evidence,
not operating-system UDP timing, processor-in-the-loop, or physical HIL evidence.

## Localhost UDP boundary

`aerognc.simulation.udp_transport` places the same encoded packets in one complete
UDP datagram. Each endpoint uses an explicitly bounded receive timeout, accepts only
an exact numeric IPv4 peer when configured, and then applies the existing CRC, type,
wrap-aware sequence, and watchdog gates. Host names are deliberately rejected so a
test cannot silently change destination through name resolution. A stale or missing
command resolves to the fail-silent all-zero actuator command.

The command below runs a synchronous operating-system socket check on `127.0.0.1`
and writes observable transmit, receive, timeout, source-rejection, invalid, type,
and sequence counters:

```powershell
python -m aerognc.cli udp-loopback --samples 100
```

This verifies packet exchange through the local UDP stack. It is not a real-time
benchmark, processor-in-the-loop result, network-security claim, or physical HIL
test. The implementation intentionally has no remote-host discovery, background
service, or hardware-control capability.

## Current software timing evidence

The configured desktop SIL benchmark calls the actual state-feedback law 10,000
times with seeded inputs. The executed local run recorded zero misses against a 1 ms
application-level deadline. Mean, 95th-percentile, and maximum call times are written
by the `flight-analysis` command, but are deliberately omitted from deterministic
reference JSON because operating-system scheduling makes those values run-dependent.
This measurement scopes the next investigation; it is not a real-time guarantee and
does not measure transport, sensor I/O, or execution on a target processor.

## Planned readiness sequence

1. Profile controller worst-case execution time and allocation behavior at the
   intended update rate.
2. Derive maximum payload rate, deadline, tolerated jitter, loss, and clock error.
3. Test those limits with the software-only link and record margins.
4. Use the verified localhost UDP boundary to derive and test an authenticated,
   operationally approved transport profile; serial framing remains future work.
5. Perform processor-in-the-loop tests against recorded deterministic scenarios.
6. Only then derive hardware I/O, timer, numeric precision, and interface needs.
7. Review electrical safety and conduct physical HIL with an approved test plan.

An FMI 3.0 Co-Simulation variable contract is also available for review; see
[FMI interoperability preparation](fmi_interoperability.md). No FMU binary has been
built or executed.

No development board is recommended here because controller timing and I/O
requirements have not yet been measured on a target implementation. The existence
of this interface and the desktop SIL measurement are preparation, not evidence of
completed HIL verification.
