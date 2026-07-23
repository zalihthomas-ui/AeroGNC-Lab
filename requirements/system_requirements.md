# AeroGNC-Lab System Requirements Specification

**Baseline:** 0.8  
**Date:** 2026-07-23
**Scope:** fictional civilian research/sounding-rocket simulation and verification

`Shall` denotes a binding first-release requirement. Verification method codes are
defined in `verification_plan.md`.

## Dynamics

- **SYS-DYN-001 (T,C):** The point-mass model shall propagate NED position and
  velocity with variable mass, thrust, gravity, aerodynamic drag, and wind at a
  configurable fixed step no greater than 0.1 s.
- **SYS-DYN-002 (T,C):** The rigid-body model shall propagate the documented
  13-element NED/FRD quaternion state and agree with constant-force analytical
  translation to relative error below \(10^{-5}\) at a 0.01 s step over 5 s.
- **SYS-DYN-003 (T):** Every accepted rigid-body step shall maintain quaternion norm
  within \(10^{-10}\) of unity after renormalisation.
- **SYS-DYN-004 (T,C):** Principal-axis constant-torque and torque-free symmetric-body
  cases shall meet benchmark tolerances stated in their tests.
- **SYS-DYN-005 (T,C):** Simplified 3-DOF and 6-DOF translations supplied identical
  forces and mass shall agree within \(10^{-6}\) m over the benchmark interval.
- **SYS-DYN-006 (T,C):** The optional rotating-frame point-mass model shall propagate
  ECEF position/velocity with central-plus-\(J_2\) gravity, Coriolis, centrifugal,
  thrust, drag, and wind; inertial/ECEF state transforms shall round-trip within
  \(10^{-8}\) m and \(10^{-10}\) m/s in the defined cases.

## Environment

- **SYS-ENV-001 (T,C):** The ISA model shall return temperature, pressure, density,
  and speed of sound from -500 m through 47 km and match standard sea-level values
  within 0.1%.
- **SYS-ENV-002 (T):** Atmosphere queries outside the configured range shall fail
  clearly rather than silently extrapolate.
- **SYS-ENV-003 (T):** Gravity shall support constant and inverse-square modes and
  produce positive NED-down acceleration for nonnegative altitude.
- **SYS-ENV-004 (T):** Wind shall support deterministic profiles and seeded stochastic
  perturbations; equal seeds and inputs shall produce equal sequences.
- **SYS-ENV-005 (T):** Mach and dynamic pressure utilities shall reject invalid
  density/speed-of-sound inputs and return SI-consistent values.
- **SYS-ENV-006 (T,C):** Oblate geodetic/ECEF conversion shall round-trip equatorial,
  mid-latitude, and near-polar cases within declared numerical tolerances, and the
  synthetic rotating-planet model shall expose separately testable gravity,
  Coriolis, centrifugal, surface-down gravity, and local transport-rate terms.

## Vehicle

- **SYS-VEH-001 (T):** Propulsion shall interpolate a synthetic thrust-time curve,
  return zero thrust outside its burn interval, and never reduce mass below dry mass.
- **SYS-VEH-002 (T):** Propellant depletion shall equal configured propellant mass at
  burnout within \(10^{-9}\) kg and be monotonic during burn.
- **SYS-VEH-003 (T):** Centre of gravity and a positive-definite inertia tensor shall
  vary continuously with remaining propellant fraction.
- **SYS-VEH-004 (T):** Aerodynamic coefficient tables shall define error, clamp, or
  linear-extrapolation behaviour explicitly and support Mach/angle-of-attack inputs.
- **SYS-VEH-005 (T,A):** Aerodynamic force shall oppose air-relative velocity in the
  drag-only case, and documented moment signs shall pass principal-axis sign tests.
- **SYS-VEH-006 (T):** Each actuator shall enforce time constant, position limit, rate
  limit, command delay, and saturation at every accepted update.
- **SYS-VEH-007 (A):** All baseline vehicle values shall be labelled fictional and
  synthetic in configuration and documentation.
- **SYS-VEH-008 (T,D):** A long-form regular-grid aerodynamic database shall support
  one or more named axes, all six force/moment coefficients, analytic in-cell
  coefficient gradients, explicit boundary policy, grid-completeness checks, and
  SHA-256 source provenance; the configured database shall run in the 3-DOF plant.
- **SYS-VEH-009 (T):** Reusable synthetic sensor-fault functions shall implement
  deterministic bias steps, one-shot spikes, stuck output, and interval dropout with
  strict channel/time validation.

## Guidance

- **SYS-GUI-001 (T):** Guidance shall interpolate a configurable time-indexed
  research-ascent attitude schedule and clamp outside its defined time range.
- **SYS-GUI-002 (A):** No public API or example shall implement target interception,
  pursuit, proportional navigation against a target, terminal homing, or engagement.
- **SYS-GUI-003 (T,D):** Public-safe constrained ascent guidance shall separate an
  offline reference search from online angle-of-attack, dynamic-pressure, proper-load,
  and desired-apogee governance; the synthetic benchmark shall satisfy its configured
  powered-ascent constraints and apogee tolerance.

## Navigation

- **SYS-NAV-001 (T):** Gyroscope, accelerometer, barometer, and civilian-GNSS-like
  models shall support configured sample rate, white noise, bias, bias drift,
  quantisation, delay, and seeded dropout where applicable.
- **SYS-NAV-002 (T):** Equal sensor configuration, truth sequence, and random seed
  shall yield identical measurement sequences.
- **SYS-NAV-003 (T,C):** The scoped navigation filter shall expose state and covariance,
  keep covariance symmetric, and reduce RMS altitude error relative to raw noisy
  barometer data in the defined validation case.
- **SYS-NAV-004 (A):** Filter state, process/measurement models, covariances,
  assumptions, and limitations shall be documented.
- **SYS-NAV-005 (T,A):** A 15-state NED/FRD quaternion error-state filter shall
  propagate position, velocity, attitude error, gyro bias, and accelerometer bias;
  fuse GNSS-like position/velocity and barometric altitude; preserve quaternion norm
  within \(10^{-12}\); and keep covariance positive semidefinite in its benchmarks.
- **SYS-NAV-006 (T,C):** A rotating-oblate-planet strapdown mechanisation shall apply
  two-sample coning/sculling compensation, body/navigation rotation, J2 gravity,
  Coriolis and transport terms while preserving a unit Hamilton quaternion.
- **SYS-NAV-007 (T,D):** The delayed 15-state ESKF shall apply timestamped GNSS-like
  and barometric measurements at fixed-lag snapshots, replay later accepted records,
  reject stale/outlying innovations, and expose deterministic sensor-health
  degradation and recovery.

## Control

- **SYS-CTL-001 (T):** PID control shall implement derivative action, output limiting,
  and anti-windup that prevents integrator growth while saturated in the benchmark.
- **SYS-CTL-002 (T,D):** The cascaded attitude/rate loop shall settle the single-axis
  step case within 3 s, with less than 15% overshoot and finite bounded command.
- **SYS-CTL-003 (T):** Manually implemented state-feedback gain calculation shall
  place the controllable SISO benchmark poles within \(10^{-8}\) of their request.
- **SYS-CTL-004 (T,D):** Controller comparison shall report rise time, settling time,
  overshoot, RMS/max error, control effort, saturation duration, disturbance recovery,
  and measured execution time.
- **SYS-CTL-005 (T):** Actuator allocation shall map bounded fictional vehicle torque
  commands to bounded channel commands with documented signs.
- **SYS-CTL-006 (T,C):** Bounded nonlinear trim and central-difference linearisation
  shall reproduce the defined analytical trim and pendulum Jacobians within
  \(10^{-8}\), and expose controllability/observability matrices.
- **SYS-CTL-007 (T,C):** The manually implemented Hamiltonian continuous LQR shall
  agree with an independent SciPy Riccati solution within \(10^{-9}\), report stable
  closed-loop modes and Riccati residual below \(10^{-10}\), and support a configured
  gain schedule.
- **SYS-CTL-008 (T,D):** Configured flight analysis shall report frequency response,
  gain/phase margins, system-identification residual, and measured SIL mean, 95th
  percentile, maximum, checksum, and deadline misses without claiming physical HIL.
- **SYS-CTL-009 (T,C,D):** Envelope analysis shall converge bounded trim, derive
  finite local \(A/B\) models, confirm full controllability/observability, synthesize
  stable LQR gains, quantify actuator authority, and verify trilinearly interpolated
  gains at every configured between-grid point.

## Simulation

- **SYS-SIM-001 (T,D):** The CLI shall run the nominal 3-DOF YAML scenario and write a
  deterministic CSV, event summary, maximum summary, and labelled figures.
- **SYS-SIM-002 (T):** Burnout, apogee, and descending ground impact shall be detected
  with event-time resolution no worse than one configured integration step.
- **SYS-SIM-003 (T):** Configuration shall reject missing keys, unknown model choices,
  nonpositive step/reference values, and inconsistent mass/thrust values with clear
  contextual exceptions.
- **SYS-SIM-004 (T):** Result logs shall have monotonic time, stable column names with
  units, and no hidden random state.
- **SYS-SIM-005 (T,D):** A configured 6-DOF scenario and a closed-loop attitude example
  shall complete without non-finite state values.
- **SYS-SIM-006 (T):** Event-aware integration shall support direction filters,
  terminal/nonterminal events, and bracketed crossing-time estimation.
- **SYS-SIM-007 (T,A):** The future HIL boundary shall define versioned, checksummed
  plant/controller packets and provide deterministic seeded latency, jitter, and
  loss emulation without claiming that physical HIL has been performed.
- **SYS-SIM-008 (T,D):** The CLI shall provide a 3-DOF flight player with pause,
  restart, seeking, bounded speed adjustment, live SI telemetry, flight-event status,
  and optional headless GIF export without modifying the source simulation result.
- **SYS-SIM-009 (T,D):** The CLI shall provide a quaternion 6-DOF 3D player showing
  East/North/altitude trajectory and body attitude, five selectable camera modes,
  pause/restart/seek/speed controls, live SI telemetry, and optional deterministic
  headless GIF export without modifying the source simulation result.
- **SYS-SIM-010 (T,D):** A Windows batch file and CLI command shall open a responsive
  guided Mission Designer with unit-labelled route, epoch, spacecraft, maneuver,
  constraint, uncertainty and result inputs; numerical errors shall be shown without
  closing the application.
- **SYS-SIM-011 (T,D):** CLI workflows shall run the rotating-planet ascent,
  aerodynamic-database audit, flight-envelope analysis, and constrained-ascent
  optimization from strict YAML and write deterministic machine-readable reports and
  labelled figures.
- **SYS-SIM-012 (T,D):** Strict configured CLI workflows shall execute advanced
  navigation and asynchronous flight-data identification, return nonzero when their
  numerical acceptance fails, and write deterministic JSON/CSV evidence and labelled
  figures.
- **SYS-SIM-013 (T,D,A):** A unified local desktop workbench and one-click Windows
  launcher shall explain the input-to-equations-to-results workflow, run the verified
  6-DOF rocket and fictional capture-orbit-departure tour from one start-page action,
  expose only basic unit-labelled inputs by default, place specialist inputs behind
  an explicit advanced disclosure, explain numerical outputs in plain language, and
  launch both 3D players. It shall also search the checksummed confirmed-exoplanet
  snapshot and display the sourced eight-planet Solar System table while identifying
  both as read-only reference data rather than executable ephemerides.
- **SYS-SIM-014 (T,D):** The software-only future-HIL loopback shall exchange typed
  plant and controller packets across independently seeded latency/jitter/loss/
  duplication links, reject stale and duplicate sequences across unsigned 32-bit
  wrap, apply a zero command after the configured receive timeout, and reproduce
  identical counters, logical latency statistics, and command checksum for identical
  inputs without claiming operating-system or hardware timing.
- **SYS-SIM-015 (T,D,A):** The unified desktop workbench shall open/save/validate a
  project, display scenario and run history, execute and cancel a selected scenario
  in the background, compare compatible runs, and open the generated engineering
  report while preserving the existing beginner rocket/tour/catalog workflows.
- **SYS-SIM-016 (T,D):** A software-only UDP transport shall use the existing
  versioned packet codec, bounded receive timeout, source filtering, sequence
  acceptance and watchdog behavior; loopback tests shall use localhost only and no
  physical-HIL result shall be claimed.

## Astrodynamics

- **SYS-AST-001 (T,C):** Configured circular planetary ephemerides shall preserve
  semi-major-axis radius, vis-viva circular speed, and Keplerian period to numerical
  tolerance, including configured inclination and ascending-node transformations.
- **SYS-AST-002 (T,A):** The interplanetary plant shall propagate the six-element
  primary-centred SI state under primary gravity and each prescribed planet's direct
  and indirect acceleration using deterministic custom RK4 with a configured maximum
  step no greater than six hours and encounter step reduction.
- **SYS-AST-003 (T,C):** Directly implemented Hohmann and hyperbolic-flyby utilities
  shall agree with their documented vis-viva, Kepler-period, eccentricity, turn-angle,
  impact-parameter, and periapsis-speed equations.
- **SYS-AST-004 (T,D):** The synthetic gravity-assist scenario shall clear the assist
  body, cross its configured encounter boundary in order, gain at least 9 km/s of
  primary-relative speed, preserve encounter-boundary planet-relative speed within
  2 m/s, and enter the configured destination corridor.
- **SYS-AST-005 (T,D):** The interplanetary CLI shall write deterministic SI trajectory
  and event/summary records and provide an immutable 3D mission-control player with
  six camera modes, event seeking, speed/timeline controls, energy telemetry, PNG
  snapshot, and headless GIF export.
- **SYS-AST-006 (T,A):** Interplanetary configuration and documentation shall label
  every baseline body, vehicle, phase, and trajectory fictional, civilian, and
  synthetic and distinguish hyperbolic flyby, corridor arrival, orbit capture, and
  operational ephemeris claims.
- **SYS-AST-007 (T,C):** Universal-variable conic propagation and classical-element
  conversion shall pass circular quarter-orbit, inclined round-trip, and independent
  MATLAB `ode113` endpoint comparisons within declared tolerances.
- **SYS-AST-008 (T,C):** The zero-revolution Lambert solver shall reproduce its
  endpoint under independent universal propagation within 0.01 m and provide
  deterministic C3/arrival-speed porkchop grids with explicit infeasible cells.
- **SYS-AST-009 (T,A):** Single-assist design shall report incoming/outgoing excess
  velocities, B-plane vector/components, turn angle, periapsis altitude, powered
  mismatch, altitude/speed feasibility, and a bounded differential-correction record.
- **SYS-AST-010 (T,D):** Inertial/RTN impulses and finite burns shall apply at exact
  event boundaries, obey the ideal rocket equation/mass-flow relation, log mass and
  maneuver events, and never cross configured dry mass.
- **SYS-AST-011 (T,A):** Optional J2, solar-radiation pressure and relativistic force
  terms shall be disabled by default, dimensionally SI, explicitly configured, and
  pass direction/scale tests.
- **SYS-AST-012 (T,C):** The separate full mutually interacting N-body model shall
  conserve total linear momentum and bound relative energy error below \(10^{-8}\)
  over the defined two-body orbital benchmark.
- **SYS-AST-013 (T,A):** Analytical and tabulated ephemerides shall implement a common
  SI interface; optional SPICE shall require `spiceypy` and existing user kernels and
  fail explicitly rather than fabricate external states.
- **SYS-AST-014 (T,D):** A planned direct trajectory shall reach its Lambert endpoint,
  and a user-entered correction shall measurably change propagated path, mass,
  destination miss, event timeline, and feasibility displayed by the 3D UI.
- **SYS-AST-015 (T,C,D):** The fictional orbit-assisted tour shall model departure
  injection, assist-body capture, an integer configured parking-orbit dwell,
  conservative asymptote-plane alignment, departure, and destination capture; apply
  the ideal rocket equation sequentially; remain above dry mass; and close both
  Lambert endpoints within 0.1 m.
- **SYS-AST-016 (T,C,D):** Launch-window analysis shall evaluate a complete configured
  departure/arrival grid, preserve invalid cells explicitly, enforce C3 and arrival
  excess-speed feasibility, refine deterministically to the configured epoch
  tolerance, never worsen the best feasible grid cost, and close the selected
  Lambert endpoint within 0.1 m.
- **SYS-AST-017 (T,C,A):** UTC, TAI, TT, Julian-date, and preliminary TDB conversion
  shall use a provenance-tagged leap-second table with an explicit validity date;
  fixed HELIOS_ECLIPJ2000/J2000 state rotations shall be orthonormal and round-trip
  position and velocity within (10^{-9}) relative tolerance.
- **SYS-AST-018 (T,D):** CCSDS OEM 3.0/KVN export shall use explicit fictional object
  identifiers, frame/time metadata, monotonic epochs, and a single documented SI to
  km/km-s exchange conversion; the local parser shall recover states within
  (10^{-6}) m and (10^{-9}) m/s.
- **SYS-AST-019 (T,A):** GMAT and SPICE interfaces shall detect availability without
  implicit execution, generate a reviewable independent point-mass case, validate
  external report shape/epochs before comparison, and record `executed=false` when
  no external run or kernel set exists.

## Astronomy data

- **SYS-DAT-001 (T,A):** Bundled Milky Way context and the eight Solar System planets
  shall carry authoritative public source URLs, explicit units, and a statement that
  descriptive values are neither a complete Galactic model nor an ephemeris.
- **SYS-DAT-002 (T,D):** The confirmed-exoplanet snapshot shall record its NASA
  Exoplanet Archive TAP query, retrieval UTC, table, field list, row count, and
  SHA-256; loading shall reject checksum, row-count, ordering, uniqueness, or schema
  violations while preserving unavailable reported values as missing.
- **SYS-DAT-003 (T,C):** Catalog search shall support deterministic name/host,
  discovery-method/year, distance, and result-limit filters; direct ICRS/Galactic
  transforms shall reproduce the known Galactic-centre direction within 0.001 deg
  and round-trip nonsingular directions within (10^{-10}) deg.
- **SYS-DAT-004 (T,D,A):** Catalog CLI evidence shall report snapshot and selection
  completeness, export the complete filtered rows, and label the map as a
  selection-biased heliocentric detection view rather than a census of Milky Way
  planets or an executable interstellar ephemeris.

## Engineering project workflow

- **SYS-PRJ-001 (T,D):** A versioned project file shall save project metadata,
  relative scenario references, workflow type, default seed, result directory, and
  tags; loading shall reject unknown keys, unsafe paths, duplicate scenario names,
  unsupported schema versions, and missing configurations with contextual errors.
- **SYS-PRJ-002 (T,A):** Every executed project scenario shall write a run manifest
  containing a stable input fingerprint, software/Python/platform versions, UTC run
  identifier, configuration SHA-256, seed, solver settings, status, warnings,
  artefacts, requirement outcomes, and explicit safety scope.
- **SYS-PRJ-003 (T,D):** A result store shall commit a completed run atomically,
  maintain a queryable local index, preserve immutable manifests and channel units,
  and reload a trajectory without access to the original in-memory result.
- **SYS-PRJ-004 (T,D):** Run comparison shall align selected same-unit channels over
  their common time domain and report sample count, bias, RMS difference, maximum
  absolute difference, and final difference; missing channels or unit mismatches
  shall fail clearly.
- **SYS-PRJ-005 (T,D):** The project layer shall generate a self-contained,
  print-ready HTML engineering report containing provenance, events, maxima,
  requirement margins, warnings, and artefact links with escaped user content.
- **SYS-PRJ-006 (T,A):** A typed workflow registry shall reject duplicate names,
  expose deterministic discovery metadata, and allow optional entry-point plugins
  without making third-party plugins a dependency of the built-in workflows.
- **SYS-PRJ-007 (T,D):** CLI commands shall create, inspect, validate, execute, list,
  compare, and report project runs; every desktop-triggered project execution shall
  use the same public service rather than a UI-only simulation path.

## Numerical methods

- **SYS-NUM-001 (T,C):** A directly implemented adaptive Dormand-Prince 5(4)
  integrator shall enforce finite positive tolerances and step bounds, report
  accepted/rejected steps and derivative evaluations, and meet configured error
  tolerances on analytical and independent SciPy benchmarks.
- **SYS-NUM-002 (T,C):** Adaptive event detection shall honor crossing direction and
  terminal behavior and locate event time by safeguarded dense-output bisection to a
  configurable time tolerance no larger than (10^{-8}) s in the benchmark case.
- **SYS-NUM-003 (T,C):** Versioned JSON/NPZ checkpoints shall preserve epoch, state,
  next step, and metadata checksum; split-and-resumed propagation shall agree with a
  continuous run within the declared solver tolerance.
- **SYS-NUM-004 (T):** A deterministic logical-time scheduler shall execute named
  plant, controller, and sensor tasks at distinct periods with stable tie ordering,
  deadline statistics, cancellation, and no dependence on wall-clock scheduling.
- **SYS-NUM-005 (T,C):** A finite-difference variational propagator shall integrate a
  state-transition matrix and selected parameter sensitivities, validate shapes and
  perturbations, and reproduce the matrix exponential of a linear benchmark within
  (10^{-5}).

## Extended engineering models

- **SYS-DYN-007 (T,C,D):** A rotating-oblate-planet quaternion 6-DOF composition
  shall propagate geodetic position, inertial velocity, attitude, angular rate,
  variable mass and inertia while keeping force/moment frames explicit and
  quaternion norm error below (10^{-9}) in the verified scenario.
- **SYS-VEH-010 (T,D):** A generic fictional multistage vehicle shall support ordered
  ignition, burnout, separation, dry-mass jettison, thrust/mass continuity checks,
  and deterministic stage-event reporting without reducing mass below the active
  dry-mass floor.
- **SYS-VEH-011 (T,D):** A public-safe recovery model shall support deployment delay,
  reefing/inflation time, drag-area evolution, opening-load reporting, and descending
  ground-contact detection for a fictional research vehicle.
- **SYS-VEH-012 (T,A):** Propulsion, mass-property, and aerodynamic CSV importers shall
  require unit-bearing schemas, finite monotonic axes, source SHA-256, explicit
  interpolation/extrapolation policy, and actionable rejection of malformed data.
- **SYS-AST-020 (T,A):** Analytical, tabulated, and optional SPICE ephemeris providers
  shall share a coverage-aware state interface; unavailable kernels or out-of-range
  epochs shall fail without silently substituting analytical states.
- **SYS-AST-021 (T,C,D):** Transfer search shall evaluate both transfer directions and
  a configurable integer-revolution set, independently propagate every accepted
  candidate to its endpoint, and rank feasible candidates by declared objective and
  deterministic tie breaking.
- **SYS-AST-022 (T,C):** Direct geometry utilities shall detect line-of-sight,
  spherical occultation, eclipse state, and ground-station elevation crossings with
  analytical benchmark comparisons and explicit body/frame definitions.
- **SYS-AST-023 (T,D):** CCSDS AEM, OPM, and TDM KVN boundaries shall validate
  mandatory metadata, time systems, units, monotonic epochs, and round-trip supported
  fictional records without claiming full standard conformance.
- **SYS-NAV-008 (T,C,D):** A batch Rauch-Tung-Striebel smoother shall consume stored
  forward-filter states/covariances and transition matrices, preserve symmetric
  positive-semidefinite covariance, and improve or equal the configured trajectory
  RMS error relative to the forward estimate.
- **SYS-DAT-005 (T,D):** A versioned telemetry mapping shall import CSV channels with
  declared source names, destination names, units, scale/offset, timestamps, quality
  flags and missing-value policy, then export the normalized record and provenance.
- **SYS-DAT-006 (T,C,D):** Telemetry analysis shall align asynchronous clocks,
  calculate gaps, residual bias/RMS/whiteness statistics, and create a deterministic
  comparison report without requiring access to synthetic truth objects.

## Performance

- **SYS-PER-001 (T,D):** Monte Carlo execution shall accept sample count, worker count,
  and master seed, and reproduce identical ordered metrics for identical inputs.
- **SYS-PER-002 (T):** Failed ensemble members shall be recorded with reason while
  successful members continue and remain summarised.
- **SYS-PER-003 (T,D):** Ensemble output shall include mean, standard deviation,
  percentiles, 95% mean confidence interval, correlations, requirement margins, and
  automatic worst-case identification.
- **SYS-PER-004 (D):** The nominal 3-DOF case with 0.02 s step shall complete in less
  than 5 s on the documented development machine, excluding plotting/startup.
- **SYS-PER-005 (T,D):** Mission-design uncertainty shall accept a fixed seed and
  worker count, preserve ordered repeatability, continue after failed solves, and
  report percentiles, central 95% bounds, correlations, and worst-case run indices.
- **SYS-PER-006 (T,D):** Flight-envelope robustness screening shall use a configured
  fixed seed, sample at least ten uncertain derivative/control/inertia cases, and
  report stable fraction, minimum damping, and worst real pole; repeated runs with
  identical inputs shall return identical numerical evidence.
- **SYS-PER-007 (T,C,D):** Design-of-experiments utilities shall provide seeded Latin
  hypercube sampling, deterministic Sobol low-discrepancy points, Morris elementary
  effects, rank/linear correlations, bootstrap confidence intervals, and input-domain
  validation against independently checkable small cases.
- **SYS-PER-008 (T,D):** Ensemble execution shall persist completed members, resume
  after interruption without rerunning valid members, reject incompatible manifests,
  and preserve ordered identical summaries across worker counts.
- **SYS-PER-009 (T,D):** A benchmark command shall record wall time, CPU time, peak
  traced memory, sample/step throughput, environment metadata, and configurable
  pass/fail budgets without treating one development-machine measurement as a
  real-time guarantee.

## Verification

- **SYS-VER-001 (T,C):** Custom RK4 shall demonstrate approximately fourth-order
  convergence (successive error ratio at least 12 when halving step) on the defined
  exponential benchmark.
- **SYS-VER-002 (T,C):** Custom RK4 shall agree with tight-tolerance SciPy `solve_ivp`
  within \(10^{-6}\) on the independent nonlinear benchmark.
- **SYS-VER-003 (T,C):** Quaternion identity, 90-degree rotations, inverse transform,
  norm, and nonsingular Euler round trips shall pass at \(10^{-12}\) scale.
- **SYS-VER-004 (T,D):** A synthetic truth flight shall produce delayed/noisy/missing
  CSV measurements that can be reloaded, event-processed, and summarised without
  access to the original in-memory objects.
- **SYS-VER-005 (C):** Optional MATLAB results shall only be reported when the script
  was executed; otherwise documentation shall state that validation is pending.
- **SYS-VER-006 (A):** Every requirement identifier shall map to implementation,
  evidence, method, and current status in the traceability matrix.
- **SYS-VER-007 (C):** MATLAB R2024a `ode113` two-body propagation shall be reported
  only after execution and agree with Python universal propagation within 0.1 m and
  \(10^{-4}\) m/s; Simulink shall remain explicitly unexecuted when product files are absent.
- **SYS-VER-008 (T,A):** Rotating-frame, aerodynamic-database, envelope, and
  constrained-ascent reports shall state their requirement domains and fidelity
  limitations; figures shall support but not replace numerical pass/fail assertions.
- **SYS-VER-009 (T,C):** The advanced-navigation workflow shall report 15-state
  observability rank, NIS/NEES chi-square consistency fractions, quaternion norm,
  covariance positive-semidefiniteness, and coning/sculling improvement for fixed-seed
  synthetic trials.
- **SYS-VER-010 (T,C,D):** Measurement-only asynchronous logs shall be clock-aligned,
  gap-preserving resampled, robustly cleaned, physically identified, residual-tested,
  and assessed on a held-out interval against measurable configured tolerances.

## Software quality

- **SYS-SWQ-001 (T):** The package shall support clean installation on Python 3.12,
  3.13, and 3.14 with declared bounded dependencies and distributed typing metadata.
- **SYS-SWQ-002 (T):** Ruff checks, strict practical mypy checks, and pytest with at
  least 75% branch-aware coverage shall pass in CI.
- **SYS-SWQ-003 (A,T):** Public numerical APIs shall use type hints, unit-bearing
  names/docstrings, focused modules, and deterministic tests.
- **SYS-SWQ-004 (T):** The GitHub Actions workflow shall install the package and run
  lint, type, test, coverage, and one smoke simulation on supported Python versions.
- **SYS-SWQ-005 (A):** Repository documentation shall include architecture, equations,
  vehicle, control, navigation, Monte Carlo, synthetic flight-test, MATLAB/Simulink,
  future HIL, limitations, licence, citation, and safety scope.
- **SYS-SWQ-006 (T,A):** Generated reference artefacts shall be reproducible and the
  repository shall exclude large transient datasets by default.
- **SYS-SWQ-007 (T,A):** The FMI 3.0 controller interface contract shall declare
  unique stable variable names/value references, causality, SI units and output
  dependencies; its adjacent status shall state `fmu_built=false`,
  `fmu_executed=false`, and official schema validation unexecuted until mandatory C
  implementation, packaging, official checking, and independent import are complete.
- **SYS-SWQ-008 (T,A):** Stable public workflow/model protocols shall carry API
  version metadata, validate plugin compatibility, and isolate plugin discovery
  failures so built-in simulations remain usable.
- **SYS-SWQ-009 (T,D):** A diagnostic command and portable launcher shall report
  Python/package/data availability, writable result location, optional external-tool
  status, and actionable remediation without changing the machine.
- **SYS-SWQ-010 (T,A):** The desktop architecture decision shall compare the retained
  Tk implementation with a PySide6/Qt or local-web prototype for accessibility,
  deployment, 3D capability, licensing, startup, memory, and maintenance before any
  framework replacement.

## Release 0.8 orbit and aircraft additions

### Dynamics

- **SYS-DYN-008 (T,C,D):** The fictional fixed-wing model shall propagate an 18-state
  planet-centred inertial rigid body comprising position, velocity, body-to-inertial
  quaternion, body rates, mass, three actuator positions, and throttle; its trimmed
  benchmark shall remain finite and airborne for eight seconds and retain a unit
  quaternion after every accepted step.

### Environment

- **SYS-ENV-007 (T,C,A):** A clearly scoped orbit/reference atmosphere shall join the
  lower ISA to a log-interpolated synthetic density table through 1000 km, provide a
  bounded tail through 1500 km, match the declared 200 km reference density, remain
  strictly decreasing at the tested thermospheric points, and expose a global density
  sensitivity multiplier without claiming space-weather prediction.

### Vehicle

- **SYS-VEH-013 (T,C,D):** The fictional fixed-wing aerodynamic model shall calculate
  (C_L,C_D,C_Y,C_l,C_m,C_n) from air-relative velocity, body rates and rate-limited
  controls; changing (C_L) slope, (C_D{}_0), (C_m{}_{\alpha}), or mass shall
  measurably change the corresponding translational/rotational derivative, and the
  configured post-stall case shall reduce lift while increasing drag.
- **SYS-VEH-014 (T,A):** The 3D player shall import bounded UTF-8 OBJ plus ASCII/binary
  STL meshes, triangulate supported polygon faces, convert a declared source-axis
  convention to FRD, and state that imported geometry is visual only and cannot
  silently change mass or aerodynamic properties.

### Control

- **SYS-CTL-010 (T,D):** Manual keyboard and optional Windows XInput commands shall
  map to normalized roll, pitch, yaw, throttle, and rocket-assist requests; physical
  surface states shall retain configured position, rate, and first-order response
  limits, and unavailable/disconnected gamepads shall not block keyboard flight.

### Simulation

- **SYS-SIM-017 (T,C,D):** A near-planet satellite sandbox shall select force-free,
  two-body, restricted three-body, full mutually interacting N-body, or J2/drag-decay
  propagation from validated configuration; the force-free case shall match its
  analytical straight line and the two-body circular case shall close one revolution
  within 100 m at the declared step.
- **SYS-SIM-018 (T,D,A):** Satellite decay shall accept altitude, speed rule, duration,
  mass, area, drag coefficient, density scale, reentry threshold, and disabled-by-
  default ideal correction inputs; it shall report reentry/escape events and shall
  describe no-event output only as survival beyond the finite modeled horizon.
- **SYS-SIM-019 (T,D,A):** The desktop workbench shall provide separate beginner-
  labelled Satellite Orbit and Aircraft Flight pages, hide specialist inputs by
  default, import a visual OBJ/STL through a file chooser, and open a live aircraft
  player accepting keyboard controls and optional non-blocking Windows XInput.
- **SYS-SIM-020 (T,C,A):** The optional fictional civilian research-ascent attitude aid
  shall remain separate from manual control, contain no target/terminal guidance, and
  cross the defined 100 km altitude boundary in the configured synthetic benchmark
  while reporting that the crossing is neither orbital insertion nor design proof.
- **SYS-SIM-021 (T,A):** The integrated fixed-wing waypoint workflow shall load a
  versioned runtime configuration covering the mission reference, solver,
  environment, navigation, guidance, autopilot, safety, reduced internal vehicle,
  actuator dynamics/failures, and output directory; it shall reject missing or
  unknown keys, unsupported schema versions or vehicle backends, invalid values, and
  any request for real-vehicle output before propagation begins.
- **SYS-SIM-022 (T,D):** The waypoint CLI and public API shall execute a validated
  runtime configuration with a fresh navigation provider for each run, preserve the
  explicit mission-only CLI form, and record both runtime-configuration and mission
  SHA-256 provenance in configured-run output.

### Astrodynamics

- **SYS-AST-024 (T,C):** The orbit sandbox shall expose conventional terminology:
  one moving body with no force is a straight-line control case, while the two-body
  problem includes a primary and satellite; model descriptions and UI text shall not
  label force-free motion as an orbit.
- **SYS-AST-025 (T,C):** The restricted three-body benchmark and full N-body benchmark
  shall produce finite deterministic states for the configured primary, synthetic
  moons and civilian satellite, with the full model applying pairwise Newtonian
  gravity to every finite-mass body.
- **SYS-AST-026 (T,D,A):** The perturbed orbit model shall combine central gravity,
  J2, rotating-atmosphere drag, satellite mass/area/Cd, dry-mass-bounded ideal
  recircularization, osculating perigee/apogee diagnostics, and modeled revolutions,
  while documenting omitted heating, breakup, lift, attitude, and space-weather
  forecasting.

### Software quality

- **SYS-SWQ-011 (T,D):** Double-clickable `run_solver.bat` and `run_aerognc.bat`
  launchers shall resolve the repository-local virtual-environment interpreter,
  preserve a readable setup failure, and open the unified workbench without requiring
  the user to type a command.
- **SYS-SWQ-012 (T,A):** Continuous integration shall avoid duplicate feature-branch
  push and pull-request runs, enforce lint/format/typing once, execute one canonical
  branch-aware coverage run, exercise supported Python versions plus Windows, and
  clean-install the built typed wheel before it can be released.
- **SYS-SWQ-013 (T):** Every baselined system requirement identifier shall be unique
  and have exactly one traceability row whose implementation and verification paths
  exist and whose declared verification methods are recognized.
- **SYS-SWQ-014 (T,A):** Repository automation shall pin external GitHub Actions to
  immutable commits, monitor Python and workflow dependencies, scan Python code and
  installed dependencies, review pull-request dependency changes, and create release
  distributions through provenance-attested, short-lived-identity workflows.
- **SYS-SWQ-015 (T):** A release tag shall use semantic `vX.Y.Z` form, exactly match
  the package and citation versions, and pass the complete quality and branch-aware
  acceptance suite before distribution artifacts can be published.
