# SITL & Physical-Output Safety Boundary Roadmap

> **Safety-first.** This roadmap covers internal simulation and local software-in-
> the-loop only. Real-vehicle output is structurally unavailable, no enabling flag
> exists, and autonomous landing is outside the accepted scope.

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
  against both internal backends using the same quantitative evidence pattern as
  `results/reference/waypoint_backend_comparison.json`.

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

## Stage 4 — Processor/software loop timing

- Reuse the project's HIL scaffolding (`simulation/hil.py`, `udp_transport.py`,
  `software_loopback.py`) for timing/latency/jitter/loss emulation.
- Measure controller timing (`gnc/flight_analysis.benchmark_controller_sil`) and
  emulated I/O budgets without connecting actuators (see `docs/future_hil.md`).

## Out-of-scope physical integration boundary

Physical aircraft, serial/CAN/radio links, actuator output, arming, and autonomous
landing are not implemented by this roadmap. Any proposal to change that boundary
requires a separately reviewed requirement set and safety case; SITL work must not
create a latent physical-output path.

## Command-level matrix

| Backend | Command level | Real output |
|---|---|---|
| Internal reduced fixed-wing | raw actuator | none (simulation) |
| Internal coefficient-driven 18-state | raw actuator | none (simulation) |
| JSBSim | raw actuator | none (simulation) |
| ArduPilot SITL | mission / attitude / velocity | none (simulation) |
| PX4 SITL | mission / offboard | none (simulation) |
| Physical hardware | unavailable / out of scope | **no path exists** |
