# Waypoint Fixed-Wing GNC — User & Developer Guide

> **Safety.** This workflow is for internal simulation and future local software-in-
> the-loop validation. It is not flight-certified, commands no real aircraft, and
> contains no physical-output path. Autonomous landing is not enabled.

## What it does

Plan a fixed-wing mission as an ordered list of waypoints, then fly it in the
internal simulator. The GNC chain automatically computes desired course, heading,
altitude, airspeed, roll/pitch, and the aileron/elevator/rudder/throttle commands,
sequences the waypoints (including loiter and return-home), enforces a safety
envelope, and logs everything.

## Quick start

```bash
# Validate a mission (schema + flight-envelope checks)
python -m aerognc.cli mission validate missions/waypoint_demo.mission.yaml

# Fly the fully configured runtime (writes CSV + JSON log and a PNG dashboard)
python -m aerognc.cli waypoint --config configs/waypoint_gnc.yaml

# Fly the same mission on the nonlinear coefficient-driven 18-state plant
python -m aerognc.cli waypoint --config configs/waypoint_gnc_coefficient.yaml

# Reproduce the reduced-versus-coefficient acceptance evidence
python scripts/compare_waypoint_backends.py

# The concise mission-only form remains available
python -m aerognc.cli waypoint --mission missions/waypoint_demo.mission.yaml \
    --guidance vector_field --output results/waypoint_gnc

# Add a steady crosswind and try a different guidance law
python -m aerognc.cli waypoint --mission missions/waypoint_demo.mission.yaml \
    --guidance l1_guidance --wind-east-mps 6
```

The bundled `missions/waypoint_demo.mission.yaml` flies
`navigate → loiter → return-home → complete` on both plants without safety
intervention. The committed comparison records 247.25 s for the reduced plant and
180.30 s for the coefficient plant, a 0.40 m final horizontal separation, and a
1.37 duration ratio under the declared 1.5 limit.

## Runtime configuration (schema version 1)

`configs/waypoint_gnc.yaml` is the reproducible execution boundary. It references
the mission and explicitly records:

- solver step, time limit, initial altitude, and initial airspeed;
- wind and gravity;
- perfect or seeded noisy navigation, including an optional GPS-dropout window;
- guidance mode and gains;
- cascaded-autopilot gains, limits, and trim;
- safety envelope and geofence;
- either reduced internal-vehicle parameters or a strict fictional-aircraft
  configuration reference;
- actuator limits, dynamics, and injected failure modes; and
- the simulation-only hardware gate.

Unknown or missing keys, unsupported schema versions/backends, invalid values, and
`hardware.allow_real_vehicle_output: true` fail before propagation. CLI guidance,
wind, step, time-limit, and output options act as explicit one-run overrides.
Configured-run JSON metadata records SHA-256 digests of the runtime file and mission.
The coefficient backend also records aircraft identity, model type, aerodynamic
backend, steady wind, source filename, and aircraft-configuration SHA-256.

### Built-in backend selection and independent comparison

The default runtime contains:

```yaml
vehicle:
  backend: internal_reduced
  parameters:  # reduced response coefficients
    # ...
```

`configs/waypoint_gnc_coefficient.yaml` instead selects:

```yaml
vehicle:
  backend: internal_coefficient
  aircraft_config: aircraft_waypoint_uav.yaml
```

The coefficient adapter propagates planet-centred inertial position/velocity,
quaternion attitude, body rates, mass, physical surfaces, and throttle. It maps the
mission actuator bank into the plant without applying actuator lag twice, converts
the rotating-planet state back to the initial local NED frame, and fails closed if
the runtime wind or gravity contract is inconsistent. `aircraft_waypoint_uav.yaml`
contains only synthetic data for the fictional 18 kg Sparrow-X2 research UAV.

`scripts/compare_waypoint_backends.py` runs the identical mission through both
models and writes a deterministic JSON record. Acceptance requires both missions to
complete without safety intervention, maximum cross-track below 175 m, duration
ratio below 1.5, terminal horizontal separation below 5 m, altitude difference
below 5 m, and airspeed difference below 1 m/s. The current reference record is
[`results/reference/waypoint_backend_comparison.json`](../../results/reference/waypoint_backend_comparison.json).
Passing these mission-level bounds demonstrates consistent behavior across two
independently structured simulators; it is not aircraft certification or proof that
the models are identical.

## Mission file format (schema version 1)

```yaml
mission_version: 1
mission:
  name: waypoint_demo
  description: Basic autonomous fixed-wing mission.
home:
  latitude_deg: 39.925
  longitude_deg: 32.8369
  altitude_m: 0.0
defaults:            # applied to waypoints that omit these fields
  airspeed_mps: 20.0
  acceptance_radius_m: 30.0
  altitude_tolerance_m: 10.0
limits:              # safe flight envelope used for validation
  min_altitude_m: 0.0
  max_altitude_m: 3000.0
  min_airspeed_mps: 12.0
  max_airspeed_mps: 45.0
  max_bank_deg: 45.0
waypoints:
  - id: 1
    name: WP1
    latitude_deg: 39.927
    longitude_deg: 32.840
    altitude_m: 120.0
    altitude_reference: relative_home   # relative_home | msl | agl
    action: fly_through                 # see actions below
  - id: 2
    name: WP2
    latitude_deg: 39.930
    longitude_deg: 32.847
    altitude_m: 180.0
    airspeed_mps: 22.0
    action: loiter
    loiter_radius_m: 100.0
    loiter_duration_s: 60.0
    loiter_direction: clockwise         # clockwise | counterclockwise
  - id: 3
    name: HOME_RETURN
    latitude_deg: 39.925
    longitude_deg: 32.8369
    altitude_m: 100.0
    action: return_home
```

### Interactive planner playback

In the planner (`aerognc mission-planner` / `run_mission_planner.bat`), **Simulate**
runs the mission on the internal simulator and overlays the flown track; **Play /
Pause / Reset** then animate the aircraft flying it (moving glyph with a heading
tick and a live HUD showing time, altitude, airspeed, groundspeed, active waypoint,
mission state, and cross-track error). Enter a steady **Wind N/E** before Simulate
to see the wind's effect, and **3D plot** saves the 3D dashboard for the last run.

**Actions:** `fly_through`, `turn`, `loiter`, `hold`, `change_altitude`,
`change_airspeed`, `takeoff`, `land`, `return_home`, `mission_end`.
(Takeoff and land are reserved and disabled by default.)

**Validation** rejects: latitude ∉ [-90, 90], longitude ∉ [-180, 180], altitude
outside the configured band, airspeed outside the safe envelope, non-positive
radii, a loiter radius below the minimum coordinated-turn radius
`r = v² / (g·tan φ)`, and duplicate waypoint ids.

## Using it as a library

```python
import aerognc

configured = aerognc.fly_configured_mission("configs/waypoint_gnc.yaml")
print(configured.summary())

from aerognc.mission import load_mission
from aerognc.gnc.waypoint_guidance import GuidanceMode
from aerognc.simulation.waypoint_mission import run_waypoint_mission, WaypointMissionConfig

mission = load_mission("missions/waypoint_demo.mission.yaml").validate()
result = run_waypoint_mission(
    mission,
    WaypointMissionConfig(guidance_mode=GuidanceMode.VECTOR_FIELD, wind_ned_mps=(0, 4, 0)),
)
print(result.summary())            # outcome, cross-track, altitude/airspeed bounds
result.to_csv("results/waypoint_gnc/log.csv")
```

## Architecture (information flow)

```mermaid
flowchart TD
  M[Mission YAML] --> MM[MissionManager - state machine]
  MM --> PM[PathManager - segments + switching]
  PM --> G[Guidance law]
  G --> AP[Cascaded autopilot]
  AP --> CS[Control surfaces + failures]
  CS --> B[VehicleBackend - internal dynamics]
  B --> NP[Navigation provider]
  NP --> G
  NP --> MM
  SAFE[SafetyManager] -. monitors / recommends .- MM
```

| Layer | Module |
|---|---|
| Mission models & I/O | `aerognc.mission` (`waypoint`, `mission`, `mission_io`) |
| Geometry | `aerognc.mathematics.local_frame` |
| Planning | `aerognc.gnc.path_manager` |
| Guidance | `aerognc.gnc.waypoint_guidance` |
| Control | `aerognc.gnc.fixedwing_autopilot` |
| Actuators | `aerognc.vehicle.control_surfaces` |
| Navigation | `aerognc.navigation` (`state`, `providers`) |
| Mission state / safety | `aerognc.mission.mission_manager`, `aerognc.mission.safety` |
| Backend / runner | `aerognc.simulation.waypoint_backends`, `aerognc.simulation.waypoint_mission` |
| Visualization | `aerognc.visualisation.waypoint_mission` |

## Conventions

- **Frames:** NED navigation, FRD body, Hamilton `quaternion_nb`. Ground **course**
  (direction of travel) and body **heading** (nose direction) are kept separate;
  wind correction applies only to the heading command.
- **Units:** strict SI internally; latitude/longitude in degrees only at the YAML
  boundary. Surface commands are normalized `[-1, 1]`, throttle `[0, 1]`, with the
  convention: positive aileron → roll right, positive elevator → pitch up, positive
  rudder → yaw right.

## Testing

```bash
python -m pytest tests/unit tests/integration/test_waypoint_mission.py
python scripts/compare_waypoint_backends.py
```

## Known limitations

- The reduced backend is a **6-DOF-lite** mission/control-design model. The optional
  18-state backend adds coefficient-driven nonlinear dynamics, rotating-planet
  kinematics, propulsion, fuel mass, atmosphere, wind, and stall behavior. Both use
  synthetic fictional inputs and neither is a validated real-aircraft model.
- Takeoff, autonomous landing, TECS, full EKF-estimated navigation, and the
  SITL/MAVLink backends are scoped but deferred — see
  `../../TODO.md` and `sitl_hardware_roadmap.md`.
