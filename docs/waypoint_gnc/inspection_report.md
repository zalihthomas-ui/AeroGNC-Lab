# Repository Inspection Report — Waypoint Fixed-Wing GNC Integration

_Date: 2026-07-21. Author: Zalih Thomas. Companion: `implementation_plan.md`, `../../TODO.md`._

## Boundary (scope decision)

The requester asked for weapons / targeting / kinetic anti-satellite capability
("navigate a satellite to take down another satellite mid-orbit"). This integration
**implements the legitimate, dual-use-neutral subset only**:

- **Orbital rendezvous & proximity operations (RPO):** navigate a chaser satellite to
  *approach, inspect, or station-keep* near a target. (Phase 16.)
- **Relative-orbit / perturbation studies:** "how altitude and orbit change when a burn,
  drag, or perturbation is introduced" — pure astrodynamics.

It deliberately does **not** implement kinetic intercept-to-destroy, terminal homing or
proportional navigation against a target, engagement/fuzing/kill-assessment logic, or the
removal of the repository's existing public-safety exclusions. This keeps the project
consistent with its own stated public-safety posture (README §Public-safety statement).
The fixed-wing waypoint workflow is implemented with reduced and nonlinear
coefficient-driven internal plants. PX4/ArduPilot SITL adapters remain optional,
scoped future work; real-hardware output is structurally unavailable.

## 1. Repository at a glance

- **Single language:** Python 3.12+ (≈52,000 LOC under `src/aerognc`). No C++, MATLAB
  *source* (only `.m` validation scripts + `.script`/`.xml` interface stubs), ROS, or
  existing MAVLink/PX4/ArduPilot code. So MAVLink/SITL support is **new** and optional.
- **Packaging:** `pyproject.toml` (hatchling). Deps: numpy, scipy, matplotlib, PyYAML,
  Pillow. Dev: mypy (strict), ruff, pytest, pytest-cov. Console script `aerognc`.
- **Entry points:** `python -m aerognc.cli <subcommand>` (41 subcommands) and Windows
  `run_*.bat` launchers. GUI via `aerognc workbench` (Tk) and `aerognc mission-designer`.
- **Tests:** 162 test files in `tests/{unit,integration,validation}`; 668 tests, 81.15%
  branch coverage, threshold 75%. Tk event-loop files omitted from coverage.
- **Quality gates:** ruff (E,F,I,N,UP,B,SIM,RUF), mypy strict, CI in `.github/workflows/ci.yml`.

## 2. Conventions (must be followed by new code)

- **Frames:** NED navigation, FRD body. Hamilton quaternion `quaternion_nb` (scalar-first,
  body→nav). ECEF/geodetic via `mathematics/geodesy.py`. Frames are **never silently mixed**.
- **Units:** strict SI, encoded in identifier suffixes (`_m`, `_mps`, `_rad`, `_radps`,
  `_kg`, `_nm`, `_s`). Angles stored in radians; degrees only at I/O edges.
- **Style:** `@dataclass(frozen=True, slots=True)` for value types; heavy `__post_init__`
  validation raising `ValueError` with explicit messages; `numpy` float64; helper
  `mathematics.vectors.as_vector(x, n, name=...)`. Public API re-exported via `__init__`
  with explicit `__all__`. Type hints everywhere (mypy strict).

## 3. Existing components — reuse / modify / new

### Reuse as-is (do not duplicate)

| Need | Existing asset |
|---|---|
| PID with anti-windup, deriv filter, saturation | `gnc/pid.py` `PIDController`, `PIDGains` |
| Cascaded attitude (angle→rate), quaternion PD | `gnc/control_loops.py` |
| Actuator lag/rate/position limit + delay | `vehicle/actuators.py` `FirstOrderActuator`, `ActuatorAllocator` |
| Geodetic↔ECEF↔NED, ellipsoid, DCMs | `mathematics/geodesy.py` |
| Body↔nav transform, α/β | `mathematics/coordinates.py` |
| Quaternion↔Euler, DCM, normalize | `mathematics/quaternion.py` |
| Integrators (RK4, Dormand–Prince, events) | `mathematics/integrators.py`, `adaptive_integrators.py` |
| Trim, linearization, LQR, modes, SIL timing | `gnc/flight_analysis.py` (`solve_trim`, `continuous_lqr`, …) |
| EKF / error-state EKF / strapdown INS | `gnc/ekf.py`, `error_state_ekf.py`, `delayed_error_state_ekf.py`, `strapdown_ins.py` |
| Fixed-wing 18-state plant, aero coeffs, sensors | `vehicle/fixed_wing.py`, `aerodynamics.py`, `sensors.py`, `sensor_faults.py` |
| Atmosphere, gravity, wind/gust | `environment/*` |
| Plot style, GIF export, 3D playback | `visualisation/style.py`, `playback_3d.py`, `aircraft_live.py` |
| Immutable run manifests, hashing, reports | `project/*` |
| Astrodynamics for RPO (Kepler, elements, perturbations) | `astrodynamics/*` |
| Config loader pattern | `configuration/*_loader.py` |

### Modify / extend

- `vehicle/actuators.py` — add a named fixed-wing channel set (aileron/elevator/rudder/
  throttle/flaps/spoilers) with neutral+trim offset and **failure modes** (wrap, don't rewrite).
- `visualisation/workbench.py` — **fix white-on-white** (plain `tk.*` widgets keep default
  white bg under the dark ttk theme), redesign input section, add live setup preview and a
  draggable/resizable professional layout, add the map-based planner page.
- `cli.py` — add `waypoint`/`mission` subcommands dispatching to the new runner.
- `configuration/` — add loaders for mission-defaults / gains / guidance / safety / backend.
- `README.md`, `CHANGELOG.md`, `docs/` — document the new workflow.

### New modules (added this effort)

- `mission/` — `waypoint.py`, `mission.py`, `mission_io.py`, later `mission_manager.py`,
  `safety.py`, `edits.py` (undo/redo edit commands).
- `mathematics/local_frame.py` — home-referenced local-tangent frame + great-circle,
  bearing, flat-earth, angle-wrap helpers. **(added)**
- `gnc/path_manager.py`, `gnc/waypoint_guidance.py`, `gnc/fixedwing_autopilot.py`.
- `navigation/` — `NavigationState` struct + perfect/estimated providers.
- `simulation/waypoint_backends.py` — `VehicleBackend` ABC plus reduced and
  coefficient-driven internal backends; JSBSim/MAVLink remain planned adapters.
- `simulation/waypoint_mission.py` — the integrated GNC loop.
- `visualisation/mission_planner_map.py` — interactive map planner.

## 4. State & command representation

- Truth state: `dynamics/state.py` `SixDofState` (13-state: NED pos, NED vel,
  `quaternion_nb`, body rates) and `ThreeDofState`. Aircraft plant uses an 18-state vector
  (adds fuel mass, actuator states) — see `vehicle/fixed_wing.py`.
- Commands today are moment/attitude-schedule based (`gnc/guidance.AttitudeReferenceSchedule`
  is time-indexed only — **no waypoint/target logic exists**, so waypoint guidance is new).
- New structured commands to add: `GuidanceCommand`, `ControlCommand` (roll/pitch/throttle),
  `ActuatorCommand` (aileron/elevator/rudder/throttle), all SI, frame-explicit.

## 5. Coordinate frames & SI check

WGS84-scale ellipsoid conversions exist but no **home-referenced** helper (needed for a
map planner working in lat/lon). Added in `mathematics/local_frame.py`. Units are
consistently SI; degrees appear only in YAML/UI and are converted at the boundary. ✔

## 6. Gaps / risks / TODO/dead-code notes

- No MAVLink/JSBSim/pymavlink dependency present → new backends must be **optional imports**
  (feature-flagged) so the core install stays lean and CI stays green without them.
- Tk UI is excluded from coverage; new UI must keep logic in testable services and expose
  widget-construction smoke tests (match existing workbench test pattern).
- Real-hardware output is outside the accepted scope and must remain structurally
  unavailable; the configuration boundary rejects any request for it.
- Numerical safety: enforce NaN/Inf guards in the new loop (repo already validates finiteness
  aggressively — keep that discipline).

## 7. Compatibility risks

- New subpackages are additive; existing subcommands/tests untouched → low regression risk.
- Adding modules to `coverage source=["aerognc"]` means new code counts toward the 75%
  threshold — ship tests with each module.
- Keep mypy strict-clean and ruff-clean or CI fails.

## 8. Proposed integration sequence

Foundations (models/geo/IO) → path manager → guidance → control → actuators → mission
state machine → safety → navigation modes → internal backend loop → UI (planner + fixes) →
plots → logging/replay → tests/scenarios → SITL backends (optional) → RPO feature → docs.
See `../../TODO.md` for the tickable breakdown.
