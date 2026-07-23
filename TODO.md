# AeroGNC-Lab — Waypoint Fixed-Wing GNC + UI Overhaul — Master TODO

> **Purpose of this file.** This is the durable, hand-off-able task tracker for a large
> multi-session effort. Any future session (human or AI) should read this file first,
> then `docs/waypoint_gnc/inspection_report.md` and
> `docs/waypoint_gnc/implementation_plan.md`, then continue from the first unchecked box.
>
> **Conventions for this file**
> - `[x]` done and verified (tests/lint/typing pass or behaviour observed).
> - `[~]` in progress / partially done — see the note beside it.
> - `[ ]` not started.
> - Every completed item should name the file(s) touched and how it was verified.
> - Do **not** mark a box done unless it actually works (no placeholders — repo policy).

---

## 0. Scope decisions & boundaries (READ FIRST)

- [x] **Weaponization carve-out recorded.** The user asked for weapons / targeting /
  kinetic anti-satellite ("take down another satellite"). We implement the *legitimate*
  capability — **orbital rendezvous & proximity operations (RPO)**, relative-orbit
  dynamics, and "how altitude/orbit change when a maneuver/drag/perturbation is
  introduced" — and we do **not** implement kinetic intercept-to-destroy, terminal
  homing / proportional navigation against a target, engagement/fuzing/kill logic, or
  removal of the repo's public-safety exclusions. This preserves the project's stated
  public-safety posture. See `docs/waypoint_gnc/inspection_report.md` §Boundary.
- [x] Confirm the primary deliverable is a **waypoint-based autonomous fixed-wing GNC
  platform**, integrated into the existing `aerognc` package, plus a UI overhaul.

---

## Phase 1 — Inspect, baseline, plan  (spec §2, §32 steps 1–4)

- [x] 1.1 Inspect repository structure, languages, frameworks. → `inspection_report.md`
- [x] 1.2 Identify existing dynamics / nav / guidance / control / actuator / mission /
  sim-loop / UI / plotting / config / logging / tests / entry points.
- [x] 1.3 Identify state & command representation, coordinate frames, SI-unit consistency.
- [ ] 1.4 Run the existing project and record baseline behaviour (nominal 3-DOF, aircraft
  sandbox). *(Needs a machine with the `.venv` active; record apogee/impact numbers.)*
- [ ] 1.5 Run existing test suite (`pytest`) and record pass/fail **before** any change.
- [x] 1.6 Write inspection report: architecture, reuse, modify, new modules, missing deps,
  compatibility risks, integration sequence. → `docs/waypoint_gnc/inspection_report.md`
- [x] 1.7 Write implementation plan + architecture diagram. → `docs/waypoint_gnc/implementation_plan.md`

---

## Phase 2 — Foundations: data models, geometry, I/O  (spec §5, §6, §24, §32 steps 5–7)

- [x] 2.1 **Waypoint data model** (`src/aerognc/mission/waypoint.py`): typed `Waypoint`,
  `WaypointAction`, `AltitudeReference`, `LoiterDirection`, `TurnType`; full validation
  (lat/lon/alt bounds, positive radii, airspeed envelope, loiter radius vs min turn radius).
- [x] 2.2 **Mission model** (`src/aerognc/mission/mission.py`): `HomePosition`,
  `MissionDefaults`, `MissionLimits`, `Mission`; whole-mission validation.
- [x] 2.3 **Coordinate/geometry utilities** (`src/aerognc/mathematics/local_frame.py`):
  `WGS84`, `LocalTangentFrame` (geodetic↔local NED via home ref), `great_circle_distance_m`,
  `initial_bearing_rad`, `flat_earth_offset_ned_m`, `wrap_to_pi`, `wrap_to_2pi`.
  Reuses existing `mathematics/geodesy.py`. (Geodetic↔ECEF↔NED, quaternion↔Euler already exist.)
- [x] 2.4 **Mission import/export + schema/version validation**
  (`src/aerognc/mission/mission_io.py`): `load_mission`, `save_mission`, dict round-trip,
  `mission_version` compatibility, actionable error messages.
- [x] 2.5 Example mission file (`missions/waypoint_demo.mission.yaml`).
- [x] 2.6 Unit tests: `tests/unit/test_local_frame.py`, `test_waypoint_model.py`,
  `test_mission_io.py`. **Verified: 38 tests pass; ruff + mypy strict clean.**
- [ ] 2.7 Add `undo/redo`-friendly mission edit operations (pure functions returning new
  `Mission`: add/edit/delete/reorder/duplicate/clear). *(model supports it; edit-command layer pending)*

---

## Phase 3 — Path manager & waypoint switching  (spec §7, §14, §32 step 8)

_Module: `src/aerognc/gnc/path_manager.py` (exported via `aerognc.gnc`). 18 tests in
`tests/unit/test_path_manager.py`. Full suite 415 unit tests pass; ruff + mypy --strict clean._

- [x] 3.1 `PathSegment` types: `LineSegment` (with altitude ramp), `OrbitSegment` (loiter
  circle), RTH handled in `PathManager.from_mission` (targets home horizontal). Dubins later.
- [x] 3.2 `PathManager`: builds legs from a mission, tracks the active segment, sequences
  waypoints, detects completion, exposes `PathManagerStatus` + `planned_path_ned` + `loiter_circles`.
- [x] 3.3 Turn anticipation: `coordinated_turn_radius_m` (`v²/(g·tan φ)`, φ→0 ⇒ inf) plus
  half-plane bisector switching for fly-through legs. *(wind-aware anticipation refined in Phase 4.)*
- [x] 3.4 Robust completion: acceptance radius + altitude tolerance + half-plane switching
  (fly-through) vs proximity (fly-over) + min-dwell + monotonic advance (no chatter; verified
  under seeded position noise).
- [x] 3.5 Fillet geometry and arc following (`FilletSegment`) with coordinated-turn/
  leg-length radius bounds, tangent half-plane switching, altitude/airspeed
  interpolation, and direction-consistent line/loiter entry and exit. Full Dubins
  routing remains an optional future planner rather than a waypoint-loop dependency.
- [x] 3.6 Unit tests: turn geometry, line/orbit geometry, fillet, segment layout, turn
  anticipation, fly-over proximity, march-to-completion without chatter, loiter dwell.

---

## Phase 4 — Guidance layer  (spec §8, §16, §32 step 9)

_Modules: `navigation/state.py` (NavigationState + FlightEnvironment, also Phase 7.1),
`gnc/waypoint_guidance.py`. 17 tests in `test_waypoint_guidance.py`; ruff + mypy strict clean._

- [x] 4.1 `GuidanceLaw` ABC + structured `GuidanceCommand` (course/heading/altitude/
  airspeed/climb-rate/roll-feedforward/cross-track/dist-to-wp/along-track).
- [x] 4.2 Backends `direct_bearing`, `line_of_sight`, `l1_guidance`, `vector_field`
  (`PathFollowingGuidance`, mode-selectable; vector-field default).
- [x] 4.3 Loiter-circle vector-field following (used for orbit legs in every mode);
  altitude interpolation via `PathSegment.commanded_down_m`; airspeed command from segment.
- [x] 4.4 Wind-aware `heading_command` (crab) kept strictly separate from ground
  `course_command` (`wind_corrected_heading_rad`); verified sign + calm-air fallback.
- [x] 4.5 Unit tests: line convergence (all modes), on-path course, saturation, orbit
  tangent/roll-feedforward sign, altitude/airspeed/climb commands, crosswind crab.
  (Closed-loop convergence scenario lands in Phase 15 with the integrated loop.)

---

## Phase 5 — Flight control (cascaded autopilot)  (spec §10, §11, §12, §32 steps 10–11)

_Module: `gnc/fixedwing_autopilot.py`. 10 tests; ruff + mypy strict clean._

- [x] 5.1 Lateral: course→roll (PI, bank-limited, + guidance roll feedforward)→aileron
  (roll error + roll-rate damping); yaw damper→rudder. Reuses `gnc/pid.PIDController`.
- [x] 5.2 Longitudinal: altitude→pitch (PI, pitch-limited)→elevator (+pitch-rate damping);
  airspeed→throttle (PI about trim). Reuses PID.
- [x] 5.3 Selectable TECS-style specific-energy sum/balance mode with reference slew,
  feedforward, output tracking, anti-windup, limits, and deterministic tuning.
- [x] 5.4 Anti-windup, output/integral limits (PID), integrator reset / bumpless re-engage
  (`reset()`), trim feedforward; gain-scheduling hook = injectable `AutopilotGains`.
- [x] 5.5 Bounded coefficient-plant trim integration with `flight_analysis.solve_trim`,
  analytic reduced-plant equilibrium, explicit reject/configured-fallback policy,
  residual provenance, and trim-initialized actuator/controller state.
- [x] 5.6 Normalized surface commands ([-1,1]) + throttle ([0,1]); documented sign
  convention shared with the internal backend and `vehicle/control_surfaces.py`.
- [x] 5.7 Strict runtime selection between legacy altitude/airspeed loops and total-
  energy control; existing LQR/state-feedback analysis remains independent.
- [x] 5.8 Unit tests: course→roll sign & bank limit, roll-rate damping, altitude→pitch &
  limit, airspeed→throttle bounds, yaw damper, output clipping, reset.

---

## Phase 6 — Actuator model extension  (spec §12)

_Module: `vehicle/control_surfaces.py`. 12 tests; ruff + mypy strict clean._

- [x] 6.1 `ControlSurfaceSet` for aileron/elevator/rudder + first-order throttle, wrapping
  `FirstOrderActuator` (min/max, rate, lag, delay) with neutral + trim offsets and
  normalized-command mapping. (Flaps/spoilers slots addable the same way.)
- [x] 6.2 Failure modes: stuck, reduced authority, reversed, oscillating, total loss
  (delayed handled via the actuator `command_delay_s`).
- [x] 6.3 Unit tests for each failure mode, limits, rate limit, trim, throttle settling.

---

## Phase 7 — Navigation modes  (spec §9)

_The frame-explicit state, perfect/noisy providers, and truth-isolated estimated mode
are implemented and tested. Estimated mode integrates sampled IMU/GNSS/barometer/
airspeed, fixed-lag rotating-NED ESKF replay, NIS gating, covariance/latency health,
and a deterministic 20 s GNSS-outage/recovery campaign. Geodetic/controller-facing
acceleration, alpha/beta, magnetometer, and terrain aiding remain optional refinements._

- [~] 7.1 `NavigationState` output struct (lat/lon/alt, NED pos/vel, groundspeed, airspeed,
  RPY, quaternion, rates, accel, course, heading, climb rate, α/β if available, validity).
- [x] 7.2 **Perfect-state mode** (simulator truth passthrough).
- [x] 7.3 **Estimated-state mode** reusing `delayed_error_state_ekf` and
  `strapdown_ins`; timestamped IMU/GNSS/barometer/pitot simulation reuses
  `vehicle/sensors.py`. Magnetometer remains optional.
- [x] 7.4 GNSS dropout/recovery, startup/cadence validation, NIS sensor health,
  measurement ages, and covariance-based validity.
- [x] 7.5 Controller must not read truth in estimated mode (enforce via interface).
- [x] 7.6 Deterministic reset, truth non-aliasing, innovation failure/recovery,
  quantitative outage campaign, and full coefficient-plant mission tests.

---

## Phase 8 — Mission manager (state machine)  (spec §13, §15, §32 step 12)

_Module: `mission/mission_manager.py`. 8 tests; ruff + mypy strict clean. Takeoff/landing states exist but are deferred/off by default (spec §34)._

- [x] 8.1 State machine: DISARMED→PREFLIGHT→READY→TAKEOFF→CLIMB→NAVIGATE→LOITER→
  RETURN_HOME→APPROACH→FLARE→LANDED / PAUSED / ABORT / EMERGENCY / MISSION_COMPLETE.
- [x] 8.2 Explicit transition conditions; validate mission before execution; set/advance
  active waypoint; trigger actions; pause/resume/RTH/abort; expose state to UI; log transitions.
- [~] 8.3 Takeoff module (ground/hand/catapult/air-start) — optional, off by default.
- [x] 8.4 Loiter module (CW/CCW, center/radius/duration/turns/indefinite, exit).
- [~] 8.5 Landing module (approach/final/glide-slope/flare/touchdown/rollout) —
  **disabled by default** until tested (spec §34).
- [x] 8.6 Unit tests: transitions, sequencing, pause/resume/abort/RTH.

---

## Phase 9 — Safety manager  (spec §19, §20, §32 step 13)

_Module: `mission/safety.py`. 8 tests; ruff + mypy strict clean. NaN/Inf enforced upstream (state ctor + backend); manual-override deferred to the hardware phase._

- [x] 9.1 Separate safety layer monitoring airspeed/bank/pitch/α/load/alt/geofence/
  nav-validity/GPS/comms/battery/cross-track/actuator-saturation/divergence/NaN-Inf.
- [x] 9.2 Responses: limit command / hold / loiter / RTH / abort / manual / terminate-sim.
- [x] 9.3 Every trigger logged (timestamp, type, state, threshold, action).
- [~] 9.4 Manual-override architecture (never permanently block manual control).
- [x] 9.5 Unit tests for each monitored condition + geofence.

---

## Phase 10 — Simulation backends & external interfaces  (spec §17, §18, §32 steps 18)
 
_Reduced and nonlinear coefficient-driven internal backends plus the integrated
runner are complete (`simulation/waypoint_backends.py`,
`simulation/waypoint_mission.py`). Both fly the nominal demo mission through
navigate->loiter->return-home->complete and pass a committed cross-model acceptance
case. SITL adapters remain optional/deferred. No real-hardware output path exists._

- [x] 10.1 `VehicleBackend` ABC (`initialize/read_state/send_actuator_commands/
  send_guidance_command/step/shutdown`) + declared command level per backend.
- [x] 10.2 **Backend A — Internal sim:** reduced mission plant plus the 18-state
  coefficient-driven fictional-aircraft plant, rotating-frame state conversion,
  non-duplicated actuator dynamics, strict configuration, and provenance. Primary,
  always available.
- [~] 10.3 Backend B — JSBSim (optional import, feature-flagged).
- [~] 10.4 Backend C — ArduPilot SITL via MAVLink (optional dep `pymavlink`).
- [~] 10.5 Backend D — PX4 SITL via MAVLink.
- [~] 10.6 MAVLink integration: connect/heartbeat/telemetry/mode/arming/mission up/clear/
  start/pause/resume/RTL/guided-wp/params/ack/timeout/reconnect.
- [x] 10.7 **Hard safety gate:** `hardware.allow_real_vehicle_output: false` by default;
  no real actuator/mission command without explicit opt-in (spec §18, §34).
- [x] 10.8 Internal backend contract, full-mission, air-start, provenance, and
  reduced-versus-coefficient comparison tests. Mock external-adapter contracts move
  with 10.3–10.6.

---

## Phase 11 — Interactive map-based mission planner (UI)  (spec §4, §21, §32 step 14)

_Module: `visualisation/mission_planner_map.py` — pure `PlannerModel` (projection, edits,
undo/redo, mission build/IO; 12 unit tests) + Tk `InteractiveMissionPlanner` (construction
smoke test). White-on-white fixed in the planner AND `workbench.py`. Launch:
`aerognc mission-planner`. Live flight-data/control panels and full draggable docking are
partial (list+property+map+run+overlay done; extra live telemetry panels deferred)._

- [x] 11.1 **Fix white-on-white bug:** plain `tk.Entry/Text/Listbox/Spinbox/Canvas` in
  `visualisation/workbench.py` inherit default white bg under the dark theme. Style them
  explicitly (bg/fg/insert/select colors) + add a reusable helper. **(spec: white-on-white)**
- [x] 11.2 Redesign input section: grouped, labelled SI units, validation-on-edit,
  advanced-fields disclosure, live validation status.
- [~] 11.3 **Live setup preview** panel: re-render planned track/altitude profile as inputs
  change (no full sim run).
- [x] 11.4 Map panel (Tk Canvas or Matplotlib): home, numbered waypoints, planned route,
  actual route, active segment, acceptance-radius circles, loiter circles, geofence,
  aircraft icon+heading, wind vector.
- [x] 11.5 Map interactions: left-click add, drag reposition, right-click menu, double-click
  set-home, wheel zoom, click-segment insert, select→property panel.
- [~] 11.6 **Draggable / resizable / professional layout** (panes, docking-style, min-size safe).
- [x] 11.7 Mission controls: add/drag/edit/delete/reorder/duplicate/clear/undo/redo/
  import/export/start/pause/resume/stop/return-home/abort.
- [~] 11.8 Live flight data, control data, actuator data, sim-control panels (spec §21).
- [x] 11.9 Widget-construction smoke tests (match existing workbench test style).

---

## Phase 12 — Visualization & plots  (spec §22, §32 step 15)
 
_`visualisation/waypoint_mission.py`: ground-track (planned vs actual), altitude, airspeed+cross-track, actuators; PNG + CSV/JSON export. GIF/MP4 replay and extra overlays deferred._

- [x] 12.1 Live + post-run plots: 2D ground track, 3D trajectory, alt/airspeed/groundspeed
  vs time, roll/pitch/heading cmd-vs-response, cross-track, dist-to-wp, 4 actuator cmds,
  flight-mode & mission-state timelines. Reuse `visualisation/style.py`.
- [~] 12.2 Optional overlays: wind, α, β, rates, integrator values, saturation.
- [x] 12.3 Export PNG/SVG/CSV/JSON + optional MP4/GIF replay (reuse Pillow GIF path).

---

## Phase 13 — Logging & replay  (spec §23, §32 step 16)
 
_Per-step structured log (`MissionSample`) + CSV/JSON export; metadata carries transitions, safety events, config. Git-hash stamping and a dedicated replay tool deferred._

- [x] 13.1 Structured logging of all categories (sim/est state, guidance/control/actuator
  cmds, waypoint events, state transitions, safety, sensors, backend comms, timing).
- [~] 13.2 Run metadata: configuration and mission SHA-256 plus guidance/navigation/
  backend selection, aircraft identity/configuration SHA-256, model type, aerodynamic
  backend, and wind are recorded; git hash, start time, and package version remain.
  (Reuse `project/` manifest patterns.)
- [~] 13.3 Replay tool reconstructing a mission from logs.

---

## Phase 14 — Configuration  (spec §25)

- [x] 14.1 Add a versioned waypoint runtime configuration for mission reference,
  solver, navigation/noise, environment, guidance, autopilot, safety, selectable
  reduced/coefficient internal vehicle, actuators/failures, output, and the
  simulation-only hardware gate. Reuse the strict `configuration/` boundary. →
  `configs/waypoint_gnc.yaml`, `configs/waypoint_gnc_coefficient.yaml`
- [x] 14.2 Validate every runtime section at startup with contextual errors; reject
  unknown keys, unsupported versions/backends, invalid physics, and real output.
- [ ] 14.3 Externalize optional planner UI preferences and structured logging policy
  only when a stable user requirement exists; they remain presentation concerns.

---

## Phase 15 — Testing  (spec §26, §32 step 17)
 
_Unit + integration (full chain) + scenario tests (nominal, all 4 guidance modes, crosswind, GPS dropout, RTH, geofence, elevator failure) pass. Remaining spec §26 scenarios and a coverage re-check are pending._

- [x] 15.1 Unit tests (coords ✔ partial, bearing/dist ✔, angle-wrap ✔, wp-validation ✔,
  path transitions, wp-completion, PID, saturation, anti-windup, actuator limits,
  mission transitions, safety).
- [x] 15.2 Integration tests along the full chain (planner→manager→guidance→control→
  actuator→dynamics→nav→guidance).
- [~] 15.3 Scenario tests (17 scenarios in spec §26) with fixed seeds.
- [x] 15.4 Keep coverage ≥ 75% (repo threshold); 668 tests pass with 81.15%
  branch-aware package coverage under pytest 9.0.3 on 2026-07-23.

---

## Phase 16 — Orbital RPO feature (legitimate satellite-to-satellite)  (user request, non-weapon)

_Module: `astrodynamics/relative_motion.py` (+ `visualisation/rpo.py`, `aerognc rpo` CLI, `docs/rendezvous_proximity_operations.md`). 9 tests. Non-weapon: approach/station-keep/orbit-change only. Interactive Tk sandbox page deferred (plots+CLI done)._

- [x] 16.1 Relative-orbit dynamics (Clohessy–Wiltshire / Hill's equations) reusing
  `astrodynamics/`. "How orbit/altitude changes when a burn/drag/perturbation is introduced."
- [x] 16.2 Rendezvous & proximity-ops guidance: navigate a chaser to approach / station-keep
  near a target satellite (V-bar/R-bar approach, safe hold points). **No intercept/kill.**
- [x] 16.3 Conjunction / close-approach distance reporting (safety, not targeting).
- [~] 16.4 UI sandbox page + plots; tests.

---

## Phase 17 — Backend/Frontend improvements previously proposed  (user: "carry out all improvements")

_`aerognc.api.fly_mission` façade done (lazily exposed as `aerognc.fly_mission`; 5 tests). Waypoint + RPO visualisation already consume plain result objects (partial 17.3). Acceleration extra, config-loader dedup, FastAPI/web dashboard, and the accessibility pass remain deferred (larger separate efforts; some need optional deps / a JS stack)._

- [x] 17.1 Stable Python API façade `aerognc.run(config) -> Result` (library use, not just CLI).
- [~] 17.2 Optional acceleration extra `[fast]` (vectorize integrator hot loops; optional Numba).
- [~] 17.3 Separate data from rendering in `visualisation/` (views consume plain result
  objects; enables web/notebook reuse + testability).
- [~] 17.4 Reduce config-loader duplication (shared validation harness).
- [~] 17.5 (Stretch) FastAPI service layer over simulators; (stretch) web dashboard;
  (stretch) notebook widgets. *Gated on §17.1/§17.3.*
- [~] 17.6 Accessibility pass: text summaries of plots, keyboard paths, contrast.

---

## Phase 18 — Documentation & GitHub presentation  (spec §29, §30, §33, §32 steps 19–20)

- [x] 18.1 Update README (purpose/features/arch diagram/screenshots/example mission/
  quick-start/backends/config/tests/limitations/**safety warning for real-aircraft use**).
- [~] 18.2 Guides: installation, architecture, coordinate-frame, mission-planning,
  controller, simulation, SITL integration, testing, safety, troubleshooting, roadmap.
- [x] 18.3 Mermaid diagrams; feature table; changelog entry (`CHANGELOG.md`).
- [~] 18.4 GitHub: issue/PR templates, contribution guide (exists), release notes for first
  waypoint-capable version. Use "Designed for simulation, SITL validation, and progressive
  preparation for hardware integration." Do **not** claim flight-certification.
- [x] 18.5 Final implementation summary + files added/modified/removed + run commands +
  perf results + limitations + next steps (ArduPilot/PX4/HIL/RC).

---

## Phase 19 — Repository, security, packaging & release hardening

- [x] 19.1 Remove duplicate feature-branch CI runs and separate quality, canonical
  coverage, compatibility/Windows, and clean-package jobs.
- [x] 19.2 Declare typed-package metadata and verify `py.typed` in the built wheel.
- [x] 19.3 Pin external Actions; add CODEOWNERS, Dependabot, CodeQL, dependency review,
  and a complete runtime/development `pip-audit` job.
- [x] 19.4 Enforce exact requirement/trace-row integrity and release tag/package/
  citation version agreement with automated tests.
- [x] 19.5 Add full tag acceptance, artifact validation, provenance attestation,
  protected-environment PyPI trusted publishing, and documented release recovery.
- [x] 19.6 Add a public roadmap and professional contribution/security guidance.
- [~] 19.7 Apply remote repository settings, labels, milestones, branch protection,
  and the PyPI environment after the hardening pull request is green and merged.

---

## Running log (newest first)

- 2026-07-23 — Trim-aware waypoint control completed: bounded nonlinear/analytic
  initialization with explicit failure policy, selectable total-energy control,
  tangent line fillets and loiter transitions, bounded command slew, controller-
  facing envelope margins, and a provenance-stamped two-backend 1 m/s-crosswind
  campaign. All 54 reference artifacts reproduce byte-for-byte. **668 tests pass
  with 81.15% branch coverage; Ruff and strict MyPy pass.**

- 2026-07-23 — Truth-isolated estimated waypoint navigation integrated with seeded
  timestamped IMU/GNSS/barometer/airspeed, fixed-lag 15-state ESKF replay, NIS
  health gates, covariance/latency diagnostics, a full coefficient-plant mission,
  and a deterministic 20 s GNSS-outage/recovery campaign. **641 tests pass with
  80.86% branch coverage; Ruff and strict MyPy pass.**

- 2026-07-23 — Repository-hardening tranche implemented: nonduplicating cross-platform
  CI, typed clean-install distributions, immutable Action pins, CodeQL/dependency
  review/audit, Dependabot/CODEOWNERS, exact traceability audits, tag/version gating,
  provenance-attested release automation, roadmap, and release documentation. Local
  wheel/sdist/Twine/clean-install/actionlint/security checks pass; **635 tests pass
  with 80.66% branch coverage under pytest 9.0.3; Ruff and strict MyPy pass.** Remote
  settings and PyPI trusted-publisher registration follow the green pull request.

- 2026-07-21 (session 3) — Phase 11 interactive map planner + white-on-white fix (workbench +
  planner), Phase 16 orbital rendezvous/proximity-ops (CW dynamics, two-impulse + V-bar approach,
  conjunction, orbit-change, `aerognc rpo`), and Phase 17.1 `aerognc.fly_mission` API façade.
  **518 unit+integration tests pass; ruff + mypy --strict clean.** Still deferred: SITL/MAVLink
  backends (need pymavlink/JSBSim), full EKF nav (7.3), TECS (5.3), trim wiring (5.5), and the
  larger Phase 17 web/accel/config-dedup items. These are documented for a future session.

- 2026-07-21 (session 2) — Phases 4-10, 12, 13, 15 implemented and committed per phase on
  `feature/waypoint-gnc`: guidance (4 modes) + navigation state, cascaded autopilot,
  control-surface actuators w/ failures, navigation providers, internal fixed-wing backend,
  mission state machine, safety manager, the integrated `run_waypoint_mission` loop, plots,
  CSV/JSON logging, and `aerognc mission validate` / `aerognc waypoint` CLI. Phase 18 docs
  (user guide, SITL/hardware roadmap, README section, CHANGELOG) added. The demo mission
  flies navigate->loiter->return-home->complete. **491 tests pass; ruff + mypy --strict clean.**
  DEFERRED for future sessions: Phase 11 interactive Tk map planner + white-on-white UI fix
  + live preview; Phase 16 orbital RPO feature; Phase 17 web/API improvements; SITL/MAVLink
  backends (10.3-10.8); full EKF-estimated nav (7.3); TECS (5.3); trim solver wiring (5.5).

- 2026-07-21 — Phase 3 path manager done: line/orbit segments, coordinated-turn radius,
  fillet geometry, half-plane turn anticipation, robust chatter-free waypoint switching,
  18 new tests (415 unit tests pass overall). Committed on `feature/waypoint-gnc`.
  Fillet arc-following + Dubins deferred (3.5). Next: Phase 4 guidance (L1/vector-field).
- 2026-07-21 — Git set up: baseline commit on `main`, work isolated on branch
  `feature/waypoint-gnc`, committing per phase (user request).
- 2026-07-21 — Phase 1 inspection complete; Phase 2 foundations (waypoint/mission models,
  local-frame geometry, mission I/O) implemented with 38 passing unit tests. Boundary on
  weaponization recorded.
