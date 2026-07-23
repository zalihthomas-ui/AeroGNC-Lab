# AeroGNC-Lab

**A modular, verification-first flight dynamics, guidance, navigation, and control platform.**

[![CI](https://github.com/zalihthomas-ui/AeroGNC-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/zalihthomas-ui/AeroGNC-Lab/actions/workflows/ci.yml)
[![Security](https://github.com/zalihthomas-ui/AeroGNC-Lab/actions/workflows/security.yml/badge.svg)](https://github.com/zalihthomas-ui/AeroGNC-Lab/actions/workflows/security.yml)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)](#status)

AeroGNC-Lab is a Python-first engineering laboratory for inspectable flight mechanics,
GNC, astrodynamics, simulation, and verification. Core models are implemented directly;
established scientific libraries support independent validation, plotting, and utilities
rather than hiding the dynamics. Assumptions, units, requirements, and evidence remain
traceable from equation to test.

**Explore:** [Quick start](#installation-and-quick-start) ·
[Architecture](docs/architecture.md) · [Validation](docs/validation_report.md) ·
[Requirements](requirements/system_requirements.md) · [Contributing](CONTRIBUTING.md) ·
[Roadmap](ROADMAP.md) · [Citation](CITATION.cff)

![AeroGNC-Lab interplanetary mission-control dashboard](results/reference/interplanetary_mission_control.png)

| At a glance | |
|---|---|
| Core domains | Flight dynamics, GNC, astrodynamics, simulation, and verification |
| Models | 3-DOF and 6-DOF flight, rotating planets, fixed-wing aircraft, and orbital missions |
| Interfaces | Python API, command-line tools, YAML projects, and a desktop engineering workbench |
| Evidence | 668 deterministic tests and 81.15% branch-aware coverage on the current branch |
| Runtime | Python 3.12 or newer |
| License | MIT |

> [!IMPORTANT]
> **Scope and safety:** all executable vehicles, missions, and parameters are fictional and
> synthetic. AeroGNC-Lab is intended for education, research, and engineering portfolios;
> it is not a certification, launch-approval, or operational mission-design tool. The project
> excludes target interception, terminal homing, engagement logic, classified information,
> proprietary vehicle data, and operational weapon-system information.

## What it solves and who it is for

A motion solver answers: **given a starting state, vehicle/environment model, and
commands, how do position, velocity, orientation, rates, loads, and events change
with time?** AeroGNC-Lab exposes four beginner-facing applications of that question:

- The **Rocket Simulator** predicts the 3D translation and rotation of a fictional
  research rocket and shows whether its closed-loop attitude controller remains
  stable.
- The **Planet Trip Planner** estimates a fictional two-leg transfer with orbit
  capture, parking revolutions, powered departure, destination capture, ideal
  delta-v, and propellant accounting.
- The **Satellite Orbit** sandbox places a fictional satellite by altitude and speed,
  distinguishes force-free, two-body, restricted-three-body, full-N-body and
  perturbed-decay physics, and reports a finite-horizon reentry/lifetime result.
- The **Aircraft Flight** sandbox propagates a fictional coefficient-driven 18-state
  aircraft. Users can run a hands-off evidence case or fly the same equations with a
  keyboard/optional XInput controller while CL, CD, Cm, mass, stall and actuators alter
  the path.

The workbench is useful to aerospace students exploring flight mechanics, engineers
reproducing and comparing algorithms, educators demonstrating models, and technical
reviewers inspecting portfolio evidence. It is not an automatic real-mission design,
operational navigation, launch approval, or certification tool. Astronomy catalog
pages provide read-only context and do not calculate routes to real exoplanets.

## Status

Version 0.8.0 is a verified alpha release candidate. It adds near-planet orbit and
live fictional-aircraft sandboxes to the portable engineering
projects and immutable run evidence, adaptive integration/checkpoint/sensitivity
tools, rotating-planet 6-DOF and staged recovery, advanced mission-analysis and
telemetry boundaries, robust experiment design, a project-aware eight-page desktop
workbench, localhost UDP packet verification, and an actionable one-click diagnostic.
The checksummed snapshot contains all 6,324 NASA-confirmed exoplanets available at
retrieval, explicitly as observational context rather than transfer ephemerides.
Automated lint, strict typing, 668 tests, and 81.15% branch-aware coverage pass; two
MATLAB benchmarks were executed independently. No FMU binary was built or imported,
and GMAT, SPICE, Simulink, and physical HIL remain explicitly unexecuted where
unavailable. Build and generation commands remain local-only and never publish or
modify remote repositories during normal use. Publication is isolated to the
tag-triggered release workflow, requires protected-environment approval and a
short-lived trusted publishing identity, and follows the documented
[release process](docs/release_process.md).

## Capabilities

- Directly implemented NED/FRD coordinates, Hamilton quaternions, table
  interpolation, and event-aware fixed-step RK4
- Direct adaptive Dormand--Prince 5(4) with dense directed events, bounded error
  control, state projection, checksummed checkpoint/restart, deterministic multi-rate
  scheduling, and finite-difference state/parameter sensitivities
- Oblate-ellipsoid geodesy, geodetic/ECEF/NED and inertial/fixed state transforms,
  central-plus-J2 gravity, Coriolis/centrifugal terms, and a configured rotating-body
  ascent on the fictional Orbis-A planet
- Lower-atmosphere 1976 ISA, configurable gravity, altitude-profile wind, and seeded
  time-correlated gusts
- Synthetic thrust/depletion, variable mass/CG/inertia, replaceable aerodynamic
  coefficient interface, and bounded delayed actuators
- Complete N-dimensional regular-grid aerodynamic databases with named axes, six
  coefficients, analytic in-cell gradients, explicit boundary policy, domain
  diagnostics, strict long-form CSV validation, and SHA-256 provenance
- Deterministic 3-DOF research-ascent propagation with burnout, apogee, and ground
  impact events
- Interactive, seekable flight playback with live SI telemetry, event/phase status,
  adjustable speed, pause/restart controls, and optional GIF export
- True 3D quaternion 6-DOF playback with East/North/altitude motion, a body-attitude
  glyph, orbit/chase/top/side/free cameras, synchronized telemetry, and GIF export
- Nonlinear quaternion 6-DOF propagation with variable inertia, synthetic
  aerodynamics, wind, bounded actuators, and closed-loop attitude hold
- Planet-centred inertial quaternion 6-DOF composition with explicit ECEF/geodetic/NED
  diagnostics, rotating-atmosphere relative wind, J2 gravity, and local-reference
  attitude control on the fictional Orbis-A planet
- YAML-driven ordered staging and reefed-to-full recovery with exact dry-mass
  jettison checks, opening-load/ground-contact evidence, and strict SHA-256-tagged
  thrust, mass-property, and aerodynamic CSV import boundaries
- Versioned portable engineering projects with validated scenarios, plugin-safe
  workflow discovery, deterministic seeds, immutable unit-aware run manifests,
  comparison, and self-contained offline HTML reports
- Universal-variable Kepler propagation, Cartesian/classical-element conversion,
  elliptic inclined ephemerides, zero-revolution Lambert transfers, porkchop launch
  windows, B-plane flyby compatibility, and multi-leg epoch correction
- Coverage-aware analytical/tabulated/optional-SPICE ephemeris providers,
  endpoint-verified direct and multi-revolution Lambert enumeration, adaptive finite
  burns, line-of-sight/eclipse/ground-access geometry, and strict scoped CCSDS
  AEM/OPM/TDM exchange
- Complete constrained launch-window screens with explicit infeasible cells,
  C3/arrival-speed limits, bounded deterministic refinement, and independent endpoint
  propagation
- A fictional capture–dwell–departure tour with sphere-of-influence branches,
  parking revolutions, conservative plane alignment, sequential ideal burn/mass
  accounting, destination capture, and an explicit orbit-versus-flyby distinction
- Provenance-tagged UTC/TAI/TT/preliminary-TDB conversion, fixed
  HELIOS_ECLIPJ2000/J2000 frames, CCSDS OEM 3.0 exchange, and reviewable unexecuted
  GMAT/SPICE validation interfaces
- Exact-time inertial/RTN impulses, finite burns, ideal propellant accounting and a
  hard dry-mass floor; optional J2, solar-radiation pressure and relativity terms
- Primary-centred restricted N-body astrodynamics plus a separate mutually
  interacting full N-body conservation model and honest optional SPICE interface
- A unified satellite sandbox selecting force-free analytical control, conventional
  two-body orbit, restricted three-body, full mutually interacting N-body, or
  central-plus-J2 rotating-atmosphere decay from one strict YAML configuration
- Finite-horizon satellite survival/reentry reporting with configurable altitude,
  speed rule, inclination, mass, area, Cd, density sensitivity, threshold and
  disabled-by-default dry-mass-bounded ideal correction impulses
- A fictional 18-state planet-centred fixed-wing plant with quaternion attitude,
  nonlinear six-coefficient aerodynamics, post-stall lift loss/drag rise, fuel mass,
  mass-scaled inertia, actuator states, wind, J2 gravity and optional rocket assist
- Live 3D aircraft flight using keyboard or optional Windows XInput, calculated stall
  speed/load factor/actual turn rate, chase/orbit/top/free cameras, and a verified
  public-safe 100 km research-ascent attitude-aid benchmark
- Bounded visual-only UTF-8 OBJ and ASCII/binary STL import with polygon
  triangulation, explicit source-axis conversion, and a bundled low-poly fictional
  aircraft; visual meshes never silently change engineering parameters
- A tuned fictional Asteria–Brontes–Caelus gravity-assist mission with a verified
  heliocentric energy gain and destination-corridor arrival
- Dark interactive 3D mission control with live ephemerides, energy history, six
  cameras, event jumping, timeline/speed controls, snapshot, and GIF export
- Native guided Mission Designer with understandable unit-labelled inputs for route,
  epochs, spacecraft, maneuvers, constraints, uncertainty and immediate 3D playback
- Cascaded PID and manually placed state feedback with quantitative response and
  disturbance-rejection comparison
- Configured trim, nonlinear finite-difference linearisation, modes,
  controllability/observability, Hamiltonian LQR, gain scheduling, frequency margins,
  flight-data system identification and measured software-in-the-loop timing
- A 36-point Mach-altitude-mass trim/control envelope, trilinear scheduled feedback,
  between-grid stability checks, actuator-authority margins, and 120 seeded uncertain
  derivative/control/inertia cases
- Offline deterministic ascent-reference search plus online max-Q, proper-load,
  loaded-angle-of-attack, and desired-apogee governance with throttle-consistent
  propellant depletion
- Seeded multi-rate gyro/accelerometer/barometer/GNSS-like sensors and a scoped
  vertical filter, plus a 15-state quaternion error-state inertial/GNSS/barometric
  navigation filter with covariance bounds
- Rotating-frame strapdown INS with two-sample coning/sculling compensation and a
  delayed 15-state ESKF with fixed-lag replay, NIS gating, sensor-health logic,
  observability rank, and seeded NIS/NEES consistency evidence
- Strict versioned telemetry mappings with unit conversion, quality/missing-value
  policy and SHA-256 provenance; affine clock alignment, gap-aware residual analysis,
  and Rauch--Tung--Striebel smoothing with covariance safeguards
- Reproducible process-parallel coupled Monte Carlo analysis with confidence
  intervals, requirement margins, worst cases, and linear sensitivity screening
- Direct seeded Latin-hypercube, deterministic Sobol and Morris designs, Pearson/rank
  screening, bootstrap intervals, checksummed resumable ensembles, failed-member
  accounting, and explicit non-real-time resource benchmarks
- Measurement-only synthetic flight-test CSV generation, reload, event detection,
  reconstruction, and automatic expected-versus-observed summary
- Asynchronous measurement-log clock alignment, dropout-preserving resampling,
  robust outlier treatment, Huber physical-parameter identification, residual
  diagnostics, confidence intervals, and held-out model prediction
- Executed optional MATLAB constant-force and adaptive two-body cross-checks plus a
  clearly unexecuted Simulink validation interface
- Versioned future HIL packets and seeded software-only latency, jitter, and loss
  and duplication emulation, wrap-aware stale rejection, a command watchdog, and a
  deterministic dual-link software loopback without claiming physical hardware
  testing
- A separately scoped localhost UDP adapter with numeric-source filtering, bounded
  receive timeout, CRC/type/sequence gates, observable counters, and a fail-silent
  command watchdog; it makes no network-security, hard-real-time, or physical-HIL claim
- A reviewable FMI 3.0 Co-Simulation attitude-controller variable contract with an
  explicit non-FMU/non-execution status, ready for a future C wrapper and independent
  import rather than presented as completed FMI validation
- Requirement-linked unit, analytical, convergence, cross-solver, integration, and
  runtime checks
- A SHA-256-protected NASA confirmed-exoplanet snapshot, sourced Milky Way/Solar
  System context, direct ICRS/Galactic transforms, readable catalog filters, and
  selection-bias-aware evidence plots; observational rows remain separate from
  executable synthetic ephemerides
- A one-command/readable environment diagnostic covering Python, package/dependency
  versions, core configurations, catalog hash, writable results, and optional tools,
  with a remediation action beside each issue and no automatic machine changes

## Architecture

```mermaid
flowchart LR
  C[Validated YAML configuration] --> S[Simulation orchestrator]
  E[Environment] --> P[Plant dynamics]
  V[Vehicle models] --> P
  G[Guidance] --> K[Control]
  K --> A[Actuators]
  A --> P
  P --> T[Truth state]
  T --> X[Sensors]
  X --> N[Navigation filter]
  N --> G
  S --> P
  C --> I[Interplanetary ephemerides]
  I --> B[Restricted N-body plant]
  C --> O[Orbit sandbox]
  B --> O
  C --> F[Coefficient-driven aircraft]
  E --> F
  V --> F
  U[Keyboard / XInput] --> F
  M[Visual-only OBJ / STL] --> Q[Live 3D view]
  F --> Q
  B --> L
  O --> L
  F --> L
  T --> L[Deterministic logging]
  L --> R[Verification and visualisation]
```

See [architecture](docs/architecture.md), [coordinate conventions](docs/coordinate_systems.md),
and [mathematical model](docs/mathematical_model.md) for the engineering definition.
Persistent studies, immutable manifests, run comparison, and offline reports are
described in the [engineering project workflow](docs/project_workflow.md).

## Installation and quick start

Python 3.12 or newer is required:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m aerognc.cli run --config configs/three_dof_nominal.yaml
```

To run the checked-in multi-scenario engineering project and retain reproducible
evidence:

```bash
python -m aerognc.cli project validate projects/portfolio_demo.aerognc.yaml
python -m aerognc.cli project run projects/portfolio_demo.aerognc.yaml nominal-3dof
python -m aerognc.cli project list projects/portfolio_demo.aerognc.yaml
```

For the easiest Windows start after installation, double-click
[`run_solver.bat`](run_solver.bat), [`run_aerognc.bat`](run_aerognc.bat), or
[`run_simulation.bat`](run_simulation.bat). It opens a Start page with four one-click
choices: **Play Rocket Example**, **Play Satellite Example**, **Fly Aircraft Example**,
or **Play Planet Trip Example**. No input editing is required for the first run. The
four solver pages show only basic inputs by default, place specialist settings behind an
explicit advanced control, and explain what happened before listing engineering
metrics. Saved-run evidence, read-only astronomy data, and engineering checks are
kept in separately named pages. See the
[workbench guide](docs/simulation_workbench.md).
If startup fails, double-click [`diagnose_aerognc.bat`](diagnose_aerognc.bat); it
checks Python, the editable package, core configurations, catalog integrity, the
result location, and optional tools, then prints an action beside every issue.
The rationale for retaining the lightweight native UI is recorded in the
[UI architecture decision](docs/ui_architecture_decision.md).

`pyproject.toml` carries portable bounded dependencies. For exact reproduction of
the verified Windows/Python 3.13 environment, install with
`-c requirements/lock-py313-windows.txt`. The lock is platform-specific and does not
replace the supported dependency ranges.

Other complete workflows use the same pattern:

```bash
python -m aerognc.cli play --config configs/three_dof_nominal.yaml
python -m aerognc.cli play-3d --config configs/six_dof_nominal.yaml
python -m aerognc.cli six-dof --config configs/six_dof_nominal.yaml
python -m aerognc.cli interplanetary --config configs/interplanetary_gravity_assist.yaml
python -m aerognc.cli orbit-tour --config configs/orbit_assisted_tour.yaml
python -m aerognc.cli orbit-sandbox --config configs/orbit_sandbox.yaml --play
python -m aerognc.cli aircraft --config configs/aircraft_sandbox.yaml
python -m aerognc.cli fly-aircraft --config configs/aircraft_sandbox.yaml
python -m aerognc.cli benchmark --config configs/three_dof_nominal.yaml --repetitions 3 --max-wall-time-s 5
python -m aerognc.cli launch-window --config configs/launch_window_optimization.yaml
python -m aerognc.cli catalog --query TRAPPIST-1
python -m aerognc.cli workbench
python -m aerognc.cli software-loopback --samples 500 --seed 218
python -m aerognc.cli udp-loopback --samples 100
python -m aerognc.cli diagnose
python -m aerognc.cli fmi-interface
python -m aerognc.cli mission-designer
python -m aerognc.cli flight-analysis --config configs/flight_control_analysis.yaml
python -m aerognc.cli rotating-ascent --config configs/rotating_planet_ascent.yaml
python -m aerognc.cli rotating-six-dof --config configs/rotating_six_dof.yaml
python -m aerognc.cli multistage-recovery --config configs/multistage_recovery.yaml
python -m aerognc.cli aero-analysis --config configs/three_dof_aero_database.yaml
python -m aerognc.cli flight-envelope --config configs/flight_envelope.yaml
python -m aerognc.cli constrained-ascent --config configs/constrained_ascent_guidance.yaml
python -m aerognc.cli attitude --config configs/attitude_control.yaml
python -m aerognc.cli navigation --config configs/navigation_demo.yaml
python -m aerognc.cli advanced-navigation --config configs/advanced_navigation.yaml
python -m aerognc.cli flight-data-identification --config configs/flight_data_identification.yaml
python -m aerognc.cli monte-carlo --config configs/monte_carlo.yaml
python -m aerognc.cli flight-test --config configs/navigation_demo.yaml
python -m aerognc.cli waypoint --config configs/waypoint_gnc.yaml
```

The nominal synthetic run currently reaches 1.101 km apogee at 15.569 s. Burnout is
detected at 3.350 s and ground impact at 31.794 s. These values are regression-scale
software evidence for the fictional configuration, not predictions for a real vehicle.

For the interplanetary gravity-assist solver and improved mission-control UI, run:

On Windows, double-click `run_interplanetary.bat` for the detailed unit-labelled
Mission Designer, or open it from the unified workbench. The equivalent PowerShell
command is:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli mission-designer
```

The UI can solve a direct Lambert transfer, screen a preliminary one-assist route,
apply manual direct-transfer corrections, draw a C3/arrival-speed porkchop plot, run
seeded design uncertainty, save readable inputs, or launch the verified reference.
See the [Mission Designer guide](docs/mission_designer.md). To bypass the designer
and run that reference directly, use:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli interplanetary `
  --config configs\interplanetary_gravity_assist.yaml
```

The custom RK4 solver propagates a primary-centred six-state spacecraft under Helios
and all configured planet accelerations, detects the Brontes flyby and Caelus arrival
corridor, writes deterministic CSV/JSON evidence, and opens a dark 3D dashboard. The
baseline gains about 10.1 km/s of heliocentric speed across the flyby boundary and
reaches the destination corridor near day 1843.5. `C` changes camera, `N` jumps to the
next event, and the mouse rotates Free view. This is a hyperbolic unpowered flyby—not
temporary orbit capture—and every body and trajectory is synthetic. See the
[interplanetary mission model and UI guide](docs/interplanetary_mission.md).

![Interplanetary gravity-assist mission control](results/reference/interplanetary_mission_control.png)

For an explicitly captured orbit rather than an unpowered flyby, run:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli orbit-tour `
  --config configs\orbit_assisted_tour.yaml
```

The synthetic spacecraft injects from Asteria, captures at Neria, completes two
parking revolutions, pays a conservative plane-alignment burn, departs from
periapsis, and captures at Caelus. The deliberately difficult geometry totals
31.456 km/s ideal delta-v and ends at 8.976 t, exposing why capture/orientation is not
a free gravity assist. See the [orbit-assisted tour model](docs/orbit_assisted_tour.md).

![Fictional capture-dwell-departure tour](results/reference/orbit_assisted_tour.png)

For a near-planet satellite rather than an interplanetary transfer, run:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli orbit-sandbox `
  --config configs\orbit_sandbox.yaml --play
```

The prepared 200 km case applies central gravity, $J_2$, a rotating reference
atmosphere and drag to the fictional Meridian-1 satellite. The same page can switch to
force-free, two-body, restricted-three-body, or full-N-body propagation. A run that
does not cross the 120 km threshold reports only a lower bound on finite-horizon
lifetime. See the [Satellite Orbit Sandbox](docs/orbit_sandbox.md).

![Near-planet satellite orbit](results/reference/orbit_trajectory_3d.png)

![Satellite altitude, osculating orbit and drag](results/reference/orbit_decay_diagnostics.png)

For the fictional fixed-wing plant, run a hands-off evidence case or live flight:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli aircraft `
  --config configs\aircraft_sandbox.yaml
.\.venv\Scripts\python.exe -m aerognc.cli fly-aircraft `
  --config configs\aircraft_sandbox.yaml
```

The Aquila-X1 uses calculated nonlinear aerodynamic forces and moments, quaternion
attitude, fuel/mass, rate-limited surfaces and rotating-planet gravity. The HUD shows
synthetic stall onset, changing stall speed, load factor and actual heading turn rate.
Press `T` in live flight for the separately verified civilian 100 km research-ascent
aid; crossing that boundary is not orbital insertion or real-design evidence. Importing
an OBJ/STL changes only the visible geometry. See
[Fictional Aircraft Simulation and Live Flight](docs/aircraft_simulation.md).

![Fictional coefficient-driven aircraft path](results/reference/aircraft_trajectory_3d.png)

![Aircraft coefficient and control diagnostics](results/reference/aircraft_flight_diagnostics.png)

The constrained direct-transfer opportunity search is reproducible with:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli launch-window `
  --config configs\launch_window_optimization.yaml
```

It screens the complete fictional epoch grid, hatches infeasible opportunities,
performs 210 deterministic evaluations, and refines the ideal injection-plus-capture
cost to 7.312 km/s with 0.00413 m independent endpoint error. See the
[launch-window method](docs/launch_window_optimization.md).

![Constrained synthetic launch window](results/reference/launch_window_optimization.png)

To browse the bundled real observational context without mixing it into the
fictional dynamics, run `python -m aerognc.cli catalog`, optionally adding
`--query`, `--method`, `--min-year`, or `--max-distance-pc`. The snapshot contains
6,324 NASA-confirmed planets in 4,738 host systems as retrieved on 2026-07-19. It is
not a census of every Milky Way planet and is not an interstellar ephemeris. See the
[catalog provenance and limitations](docs/galaxy_catalog.md).

![NASA confirmed-exoplanet catalog context](results/reference/milky_way_confirmed_exoplanet_catalog.png)

For the interactive 3D quaternion simulation, run:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli play-3d `
  --config configs\six_dof_nominal.yaml
```

This opens a true 3D East/North/altitude scene. The vehicle centreline and coloured
body axes rotate from the simulated quaternion; `C` cycles orbit, chase, top, side,
and free cameras, and mouse dragging rotates the free camera. The dashboard includes
play/pause, restart, timeline seeking, speed controls, live SI telemetry, body rates,
aerodynamic angles, and attitude error. See the [3D player guide](docs/playback_3d.md).

![Interactive quaternion 6-DOF 3D playback](results/reference/six_dof_playback_3d.png)

For the longer 3-DOF launch-to-impact animated view, run:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli play `
  --config configs\three_dof_nominal.yaml
```

The window provides play/pause, restart, timeline seeking, speed adjustment, a moving
vehicle marker, flight phase/events, and live telemetry. See the
[playback guide](docs/playback.md) for controls and headless GIF export.

![Interactive 3-DOF flight playback](results/reference/three_dof_playback.png)

![Nominal 3-DOF trajectory](results/reference/three_dof_trajectory.png)

![Nominal 3-DOF loads](results/reference/three_dof_loads.png)

![Closed-loop quaternion 6-DOF ascent](results/reference/six_dof_ascent.png)

![Attitude-controller comparison](results/reference/attitude_control_comparison.png)

![Flight-control modes, frequency response, and LQR gain schedule](results/reference/flight_control_analysis.png)

![Rotating-oblate-planet research ascent](results/reference/rotating_planet_ascent.png)

![Rotating-planet quaternion 6-DOF ascent](results/reference/rotating_six_dof_ascent.png)

![Fictional two-stage recovery demonstration](results/reference/multistage_recovery.png)

![Synthetic multidimensional aerodynamic database](results/reference/aerodynamic_database_analysis.png)

![Flight-envelope trim and scheduled control](results/reference/flight_envelope.png)

![Constrained research-rocket ascent guidance](results/reference/constrained_ascent_guidance.png)

![Synthetic vertical navigation](results/reference/navigation_filter.png)

![Rotating strapdown INS and delayed ESKF](results/reference/advanced_navigation.png)

![Synthetic flight-data alignment and identification](results/reference/flight_data_identification.png)

![Coupled Monte Carlo requirement dashboard](results/reference/monte_carlo_summary.png)

![Monte Carlo sensitivity screening](results/reference/monte_carlo_sensitivity.png)

![Synthetic flight-test evaluation](results/reference/flight_test_summary.png)

## Waypoint fixed-wing autonomous GNC

A waypoint-based autonomous fixed-wing workflow demonstrates the full chain from
mission planning to actuator commands. Define a mission as an ordered list of
waypoints (home, fly-through, loiter, return-home), then pair it with a versioned
runtime configuration covering navigation, guidance, control, safety, vehicle,
actuators, wind, and solver settings. The GNC system computes desired course,
heading, altitude, airspeed, roll/pitch, and aileron/elevator/rudder/throttle
commands, sequences the waypoints, holds loiter patterns, returns home, and enforces
a safety envelope (airspeed/bank/pitch/altitude/geofence/GPS-loss).

```bash
python -m aerognc.cli mission validate missions/waypoint_demo.mission.yaml
python -m aerognc.cli waypoint --config configs/waypoint_gnc.yaml
python -m aerognc.cli waypoint --config configs/waypoint_gnc_coefficient.yaml
python -m aerognc.cli waypoint --config configs/waypoint_gnc_estimated.yaml
python -m aerognc.cli waypoint --config configs/waypoint_gnc_tecs.yaml
python scripts/compare_waypoint_backends.py
python scripts/verify_waypoint_navigation.py
python scripts/verify_waypoint_control.py
```

The concise form remains available with `--mission`; guidance, wind, step, time-limit,
and output flags can override either form explicitly. Library users can call
`aerognc.fly_configured_mission("configs/waypoint_gnc.yaml")`. The runtime schema
selects either the reduced mission-level plant or the nonlinear coefficient-driven
18-state fictional UAV plant, rejects external/real-vehicle output, and records
runtime, mission, backend, aircraft, and aircraft-configuration provenance in the
structured result.

The estimated-navigation runtime feeds the coefficient-driven plant through seeded
timestamped IMU, GNSS, barometric, and airspeed sensors and a fixed-lag 15-state
ESKF. A committed 20-second GNSS-outage campaign bounds drift and recovery while
keeping truth confined to a separate scoring channel: 9.109 m maximum outage error,
0.355 m recovery position RMS, and 0.061 m/s recovery velocity RMS. Runtime
diagnostics expose covariance, latency, NIS gates, and sensor health without truth
scores. See the [estimated-navigation design](docs/waypoint_gnc/estimated_navigation.md).

The trim/total-energy runtime solves a bounded straight-flight equilibrium, starts
the controller and actuators bumplessly at that solution, coordinates altitude and
airspeed through energy sum/balance, and replaces abrupt corners with tangent
fillets and direction-consistent loiter entry/exit. Its 1 m/s-crosswind campaign
completes on both internal plants with zero safety events or actuator saturation,
at most 20.831 m cross-track, at least 8.000 m/s stall margin, at least 70.14%
remaining surface authority, and 1.346 m terminal separation. See the
[trim/TECS/path design and evidence](docs/waypoint_gnc/trim_tecs_path_control.md).

Selectable guidance (`direct_bearing`, `line_of_sight`, `l1_guidance`,
`vector_field`) feeds a cascaded autopilot; a `VehicleBackend` interface and a
`NavigationProvider` abstraction keep both internal plants interchangeable and form
the boundary for future local JSBSim or ArduPilot/PX4 SITL adapters. Real-aircraft
output is structurally unavailable, the models are not flight-certified, and
autonomous landing is disabled. The committed matched-mission evidence reports both
models completing without safety intervention and meeting declared cross-model
bounds. See the [waypoint GNC user & developer guide](docs/waypoint_gnc/user_guide.md),
the [SITL/hardware roadmap](docs/waypoint_gnc/sitl_hardware_roadmap.md), and the
task tracker in [`TODO.md`](TODO.md).

## Mathematical model summary

The baseline point-mass equations propagate NED position/velocity with explicit
thrust, air-relative drag, gravity, wind, and scheduled mass. An optional model
propagates ECEF position/velocity over a rotating oblate ellipsoid with J2, Coriolis,
and centrifugal acceleration. The rigid-body increment uses a 13-state quaternion
model. A separate six-state ecliptic-frame model applies primary gravity plus direct
and indirect planetary accelerations using analytical ephemerides.
The interplanetary layer adds universal conic propagation, Lambert boundary-value
solutions, B-plane geometry, maneuver mass flow, optional perturbations, and
restricted/full N-body alternatives. Governing equations and assumptions are documented in
[`docs/mathematical_model.md`](docs/mathematical_model.md); frame and sign conventions
are normative in [`docs/coordinate_systems.md`](docs/coordinate_systems.md). Advanced
mission mathematics and fidelity boundaries are in
[`docs/advanced_astrodynamics.md`](docs/advanced_astrodynamics.md).
The selectable near-planet force models and honest finite-horizon decay semantics are
specified in the [satellite orbit model](docs/orbit_sandbox.md). The 18-state
coefficient-driven aircraft equations, stall model, live controls, visual-mesh boundary
and 100 km scope are specified in the
[fictional aircraft model](docs/aircraft_simulation.md).
The new interfaces are detailed in [geodesy/rotating flight](docs/geodesy_rotating_planet.md),
[aerodynamic databases](docs/aerodynamic_database.md),
[flight-envelope analysis](docs/flight_envelope.md), and
[constrained ascent guidance](docs/constrained_ascent_guidance.md).
The higher-order estimator and measurement-analysis workflows are documented in
[advanced navigation](docs/advanced_navigation.md) and
[flight-data identification](docs/flight_data_identification.md). General
measurement-file import, provenance, asynchronous clock correction, gap/residual
evidence, and offline RTS smoothing are specified in
[telemetry reconstruction](docs/telemetry_analysis.md).
Reproducible Latin-hypercube/Sobol/Morris studies, interruption-safe ensembles, and
local resource budgets are documented in
[robust experiments](docs/robust_experiments.md).
The new astrodynamics boundaries are detailed in the
[orbit-assisted tour](docs/orbit_assisted_tour.md),
[launch-window optimizer](docs/launch_window_optimization.md), and
[time/frame/interoperability record](docs/astrodynamics_interoperability.md). The
[verified mission-analysis boundary](docs/mission_analysis.md) documents
coverage-aware ephemerides, direct multi-revolution Lambert enumeration, finite-burn
execution, visibility/access geometry, and scoped AEM/OPM/TDM exchange.

## Verification philosophy

Plots communicate behaviour but are not treated as proof. Verification combines
analytical benchmarks, unit tests, convergence studies, independent SciPy solver
comparisons, cross-model consistency checks, regression cases, and requirement-based
acceptance. Evidence is mapped in
[`requirements/traceability_matrix.csv`](requirements/traceability_matrix.csv).

## Verification summary

| Evidence | Executed result |
|---|---|
| Python suite | 668 deterministic unit, integration, and validation tests pass |
| Coverage | 81.15% branch-aware core/package coverage; enforced threshold 75% |
| RK4 | Fourth-order convergence and independent SciPy agreement below \(10^{-6}\) |
| Adaptive numerics | Dormand--Prince convergence/reference agreement, dense events, checkpoints, scheduler order, and variational derivatives pass |
| Nominal 3-DOF | 1101.49 m apogee; burnout/apogee/impact events ordered and bounded |
| Closed-loop control | PID settling 2.495 s, overshoot 7.86%, recovery 0.94 s |
| Linear flight control | Stable LQR modes, Riccati residual \(2.16\times10^{-14}\), phase margin 66.9 deg |
| Rotating flight mechanics | Geodetic/state round trips and frame-term signs pass; inertial 6-DOF quaternion error is below \(10^{-9}\) |
| Staging and recovery | Ordered ignition/burnout/separation, dry-mass jettison, continuous inflation, opening load, and ground contact pass |
| Engineering projects | Five bundled workflows validate; immutable manifests, hashes, result comparison, reports, cancellation, and registry isolation pass |
| Aerodynamic database | Exact multilinear fields/gradients, all boundary policies, malformed grids, provenance, and configured flight pass |
| Flight envelope | 36/36 trims; 12/12 interpolated points and 120/120 seeded uncertain models stable; minimum unused authority 99.23% |
| Constrained ascent | 785.3 m vs 800 +/- 18 m; 8.84 kPa max-Q; 6.00 g0 proper load; 4.10 deg loaded angle; all pass |
| Error-state navigation | Stationary/rotating propagation, covariance PSD, and GNSS/barometer correction pass |
| Advanced navigation | 2.490 m position, 0.290 m/s velocity, 1.811 deg attitude RMS; rank 15; 18-step delayed replay; consistency/integrity checks pass |
| Flight-data identification | Clock offset/drift recovered; all physical parameters within 0.4%; R-squared 0.9983; held-out pitch RMS 0.076 deg |
| Gravity assist | 371,000 km centre distance; +10.1 km/s boundary speed change; Caelus corridor reached |
| Orbit-assisted tour | Two Neria parking revolutions; 31.456 km/s accounted ideal burns; 8,975.65 kg final mass above dry limit; all assertions pass |
| Launch-window search | 210 evaluations; 7.312386 km/s selected cost; 0.00413 m propagated endpoint error; all constraints pass |
| Time/frame/OEM | Leap/time-offset, orthonormal frame, and CCSDS SI/exchange round trips pass; GMAT/SPICE execution remains false |
| Mission-analysis boundary | Multi-revolution endpoints, ephemeris coverage, finite-burn mass balance, visibility/access, and AEM/OPM/TDM strict round trips pass |
| Astronomy catalog | SHA/count/order/uniqueness verified for 6,324 confirmed planets; ICRS/Galactic known-direction and round-trip tests pass |
| Orbit sandbox | Force-free analytical match; one-period two-body closure; deterministic restricted/full N-body; drag-energy loss and lifetime-scope tests pass |
| Fictional aircraft | CL/CD/Cm/mass derivative sensitivity, post-stall lift/drag, trim, turn, mesh/controller and 100 km boundary tests pass |
| Simulation Workbench | Eight-page native widget smoke, project/run operations, editable rocket/orbit/aircraft/tour inputs, validation, catalog filtering, and 3D host grouping pass |
| Software loopback | 500 samples; 499 state and 498 command packets accepted; zero deadline misses or transport drops; two safe startup-watchdog activations |
| Localhost UDP | 100/100 state and command packets accepted; source/CRC/type/sequence/timeout/watchdog tests pass; no real-time or physical-HIL claim |
| Telemetry and smoothing | Mapping/unit/quality/provenance validation, clock alignment, gap-aware residuals, covariance PSD, and RTS error reduction pass |
| Robust studies | LHS/Sobol/Morris designs, correlations/bootstrap, worker-independent resumable ensembles, and scoped benchmark budgets pass |
| Environment diagnostic | Python/package/dependencies, 6,324-row catalog hash, writable results, and optional tools report READY with explicit remediation |
| FMI preparation | Deterministic 24-variable FMI 3.0 Co-Simulation XML contract passes project validation; FMU build/import and official XSD validation remain false |
| Navigation | Altitude RMS reduced from 3.018 m raw to about 0.453 m estimated |
| Waypoint estimated navigation | 20 s GNSS outage: 9.109 m maximum drift, 0.355 m recovery RMS, valid state throughout |
| Waypoint trim/TECS/path control | Both internal plants complete in 1 m/s crosswind; <=20.831 m cross-track, >=8.000 m/s stall margin, zero saturation/safety events, 1.346 m terminal separation |
| MATLAB constant force | Python/MATLAB maximum state difference \(5.12\times10^{-13}\), pass |
| MATLAB two body | Universal Kepler vs adaptive `ode113`: 77.9 nm position and \(1.11\times10^{-10}\) m/s velocity error, pass |

Executed evidence, assumptions, and gaps are recorded in the
[validation report](docs/validation_report.md). MATLAB details are in the
[cross-validation record](docs/matlab_validation.md); the separate
[Simulink status](docs/simulink_validation.md) states why no result is claimed.

## Repository map

- `src/aerognc/`: directly implemented atmospheric, rigid-body, GNC, and astrodynamics models
- `configs/`: readable synthetic vehicle and scenario definitions
- `projects/`: portable multi-scenario workspaces for reproducible local studies
- `data/catalogs/`: dated, checksummed observational context and its provenance
- `tests/`: unit, integration, and independent validation evidence
- `requirements/`: measurable specification, plan, and traceability
- `docs/`: conventions, equations, subsystem designs, and reports
- `examples/` and `scripts/`: reproducible engineering workflows
- `matlab_validation/`, `simulink_validation/`, `gmat_validation/`, and `fmi_validation/`: optional independent checks and truthful execution status
- `results/reference/`: 54 compact, reproducible representative PNG/JSON artifacts

The branch-coverage metric excludes the native Tk Mission Designer and Simulation
Workbench event-loop adapters, which are exercised with CLI dispatch tests and
live-window/widget-construction smoke checks. Their input, mission-planning,
propagation, catalog, uncertainty, and plotting backends remain in the automated
coverage scope. All 54 reference PNG/JSON artifacts were SHA-256 identical across two
consecutive complete generations; deterministic numerical records omit runtime and
workspace-dependent paths. The current compact set can be audited with
`tests/integration/test_reference_generation.py`.

## Roadmap and limitations

The project deliberately favors inspectable models and verification depth over
high-fidelity breadth. The baseline atmospheric 3-DOF/6-DOF plant still uses a flat,
nonrotating local NED frame; ECEF translation and planet-centred inertial 6-DOF are
explicit alternatives, not silently mixed into that baseline. It uses a 47 km ISA
limit, a sparse synthetic
demonstration aerodynamic grid, simplified rail-exit/actuator behavior, a reduced
two-state envelope model, a pitch-plane ascent optimizer, and a three-state vertical
demo filter. It excludes structural flexibility, slosh,
  aeroelasticity, high-fidelity coupled canopy dynamics, combustion/internal-ballistics transients,
  production inertial navigation, real-time operating-system guarantees, physical flight data,
and hardware tests. The 3-DOF model is not an attitude model. Monte Carlo correlation
is screening evidence, not global sensitivity proof.

The reference interplanetary model uses prescribed synthetic elliptical ephemerides
and a massless spacecraft. Full mutual N-body gravity, J2, radiation pressure,
relativity and SPICE are available as separate opt-in studies, not silently enabled
in the regression case. There is no bundled operational ephemeris, low-thrust/global
trajectory optimiser, certified navigation covariance, atmospheric entry,
finite-burn orbit-capture execution, or landing. Destination arrival means corridor
entry unless the separate ideal patched-conic capture budget is explicitly reported.
The real exoplanet catalog is sparse, dated observational context: it neither supplies
complete planetary phases nor enables physically meaningful interstellar transfers.

Future work may add flexible-body/canopy coupling, calibrated/lever-arm navigation
states, variance-based global sensitivity indices, and an executed Simulink
comparison. Hardware selection remains deferred until controller timing and target
I/O requirements are measured; see the [project roadmap](ROADMAP.md) and
[future HIL plan](docs/future_hil.md).

## Author, licence and citation

AeroGNC-Lab is created and maintained by
[Zalih Thomas](https://github.com/zalihthomas-ui). Released under the
[MIT License](LICENSE). Citation metadata is provided in
[`CITATION.cff`](CITATION.cff). The canonical source repository is
[github.com/zalihthomas-ui/AeroGNC-Lab](https://github.com/zalihthomas-ui/AeroGNC-Lab).
