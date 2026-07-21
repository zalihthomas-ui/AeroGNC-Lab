# SITL & Hardware Integration Roadmap

> **Safety-first.** Every step below keeps real-vehicle output **disabled by
> default**. A backend that can command hardware must be gated behind an explicit
> configuration flag (e.g. `hardware.allow_real_vehicle_output: true`) that does
> not exist yet. Do not claim flight-readiness without staged, logged hardware
> testing. Autonomous landing stays disabled until independently validated.

The GNC logic is already isolated behind two interfaces so backends can be swapped
without touching guidance/control:

- `aerognc.simulation.waypoint_backends.VehicleBackend` — `initialize`, `read_state`,
  `send_actuator_commands`, `step`, `shutdown`, plus a declared `CommandLevel`
  (raw-actuator / body-rate / attitude / velocity / position / mission-waypoint).
- `aerognc.navigation.providers.NavigationProvider` — perfect truth or estimated.

## Stage 1 — JSBSim backend (optional dependency)

- Add `jsbsim` as an optional extra; import lazily and feature-flag it.
- Implement `JsbSimBackend(VehicleBackend)` mapping our normalized surface
  commands to JSBSim FCS inputs and reading its state into `NavigationState`.
- Reuse the existing guidance/autopilot/mission/safety layers unchanged.
- Validate: fly the bundled demo mission and compare ground track / completion
  against the internal backend.

## Stage 2 — ArduPilot SITL via MAVLink

- Add `pymavlink` as an optional extra. Implement a `MavlinkBackend` at
  `CommandLevel.MISSION_WAYPOINT` (upload our waypoints as a MAVLink mission) and/or
  `ATTITUDE`/`VELOCITY` (guided/offboard).
- Support: connect, heartbeat, telemetry, mode/arming, mission upload/clear/start,
  pause/resume, RTL, guided waypoint, parameter read, command ack, timeout,
  reconnect.
- Run ArduPilot SITL (`sim_vehicle.py -v ArduPlane`), point the backend at its
  MAVLink endpoint, and fly the mission in SITL. Compare against the internal run.

## Stage 3 — PX4 SITL via MAVLink

- Same `MavlinkBackend`, PX4 endpoint; use offboard/mission as appropriate.
- Verify mode transitions and failsafe interaction with our safety manager.

## Stage 4 — Hardware-in-the-loop (HIL)

- Reuse the project's HIL scaffolding (`simulation/hil.py`, `udp_transport.py`,
  `software_loopback.py`) for timing/latency/jitter/loss emulation.
- Measure controller timing (`gnc/flight_analysis.benchmark_controller_sil`) and
  target I/O requirements **before** selecting hardware (see `docs/future_hil.md`).

## Stage 5 — RC-aircraft integration

- Keep manual RC override always available (never permanently blocked by the
  experimental GNC). Provide stabilized / autonomous / return-home / emergency
  modes.
- Require the explicit `hardware.allow_real_vehicle_output` opt-in, a pre-arm
  checklist, geofence, and a command watchdog. Begin with bench tests, then
  tethered/low-risk flights with a safety pilot, logging every flight.

## Command-level matrix

| Backend | Command level | Real output |
|---|---|---|
| Internal fixed-wing (this build) | raw actuator | none (simulation) |
| JSBSim | raw actuator | none (simulation) |
| ArduPilot SITL | mission / attitude / velocity | none (simulation) |
| PX4 SITL | mission / offboard | none (simulation) |
| Hardware (future) | per flight controller | **gated, off by default** |
