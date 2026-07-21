# Changelog

All notable changes are documented here. The format follows Keep a Changelog,
and the project uses Semantic Versioning.

## [Unreleased]

## [0.8.0] - 2026-07-20

### Added

- A validated Satellite Orbit sandbox with force-free, two-body, restricted
  three-body, full mutually interacting N-body, and central-plus-J2/drag-decay
  modes; it reports correction impulses, revolutions, osculating diagnostics, and
  finite-horizon re-entry or survival without claiming an infinite lifetime.
- A fictional 18-state civilian research-aircraft plant with quaternion attitude,
  nonlinear six-coefficient aerodynamics, post-stall lift loss and drag rise,
  atmosphere-relative wind, changing fuel mass/inertia, bounded actuators, and an
  optional public-safe rocket-assist research-ascent case.
- Live 3D aircraft flight with keyboard and optional Windows XInput commands,
  calculated stall speed/load factor/turn rate, four cameras, and a separate
  no-target 100 km attitude aid; plus bounded visual-only OBJ and ASCII/binary STL
  import and a bundled low-poly Aquila-X1 example model.
- Dedicated beginner-facing Satellite Orbit and Aircraft Flight workbench pages,
  four one-click Start workflows, advanced-input disclosures, plain-language result
  interpretations, deterministic CLI entry points, and a `run_solver.bat` alias.
- Requirements, trace links, governing-equation documents, numerical regression
  cases, and six new compact reference artifacts for the two new workflows.

### Changed

- Reframed the desktop UI around four questions it can answer: rocket motion and
  stability, satellite orbit/lifetime, coefficient-driven aircraft response, or
  preliminary completion of a fictional three-world trip.
- Added one-click prepared simulations on the Start page, hid specialist inputs by
  default, renamed every page by user purpose, separated read-only astronomy data
  from solvers, and replaced metric-first output blocks with plain-language result
  explanations. The guided pages now scroll at the minimum supported window size and
  move directly to the explanation when a calculation completes.
- Advanced package, CLI, citation, FMI interface, requirements, architecture, README,
  validation, and UI metadata to version 0.8.0 and an eight-page workbench.

### Verification

- Force-free analytical motion, one-period two-body closure, restricted/full N-body
  finiteness, drag-energy loss, exact re-entry threshold, and finite-horizon wording
  pass automated orbit tests.
- Aircraft trim, coefficient/mass derivative sensitivity, post-stall behavior,
  actuator/controller mapping, mesh parsing, live-input neutrality, batch/CLI output,
  and the calculated 100 km research-ascent boundary pass automated tests.
- The full hidden eight-page workbench constructs with specialist inputs initially
  hidden. The canonical reference generator completes and produces 51 compact
  PNG/JSON artifacts; the four new figures were inspected locally.
- The complete suite passes 402 deterministic tests with 80.23% branch-aware package
  coverage against the enforced 75% floor; Ruff check/format, strict source typing,
  editable installation, and dependency integrity pass.
- Two consecutive canonical generations retain identical SHA-256 hashes for all 51
  compact reference artifacts.

## [0.7.0] - 2026-07-20

### Added

- Versioned engineering projects with strict portable paths, immutable provenance,
  atomic unit-aware result storage, run comparison, self-contained HTML reports, and
  a shared workflow service for command-line and future desktop clients.
- A directly implemented adaptive Dormand--Prince 5(4) solver with error statistics,
  dense directed-event bisection, quaternion-compatible projection, checksummed
  checkpoint/restart, deterministic multi-rate scheduling, and finite-difference
  state-transition and selected-parameter sensitivity propagation.
- A planet-centred inertial quaternion 6-DOF composition on the fictional Orbis-A
  rotating oblate planet, plus YAML-driven ordered staging, continuous reefed/full
  recovery deployment, opening-load/ground-contact evidence, and strict unit-bearing
  propulsion, mass-property, and aerodynamic CSV importers with SHA-256 provenance.
- Coverage-aware analytical/tabulated/SPICE ephemeris providers, independently
  endpoint-verified zero/multi-revolution Lambert search, exact-boundary adaptive
  finite burns, line-of-sight/eclipse/ground-access geometry, and strict scoped
  CCSDS AEM/OPM/TDM KVN exchange for fictional mission records.
- A strict versioned telemetry CSV mapping with unit conversion, quality/missing-data
  policies and SHA-256 provenance; affine clock alignment, gap-aware residual reports,
  and a stored-history Rauch--Tung--Striebel smoother with covariance PSD safeguards.
- Direct seeded Latin-hypercube, deterministic Sobol, and Morris experiment designs;
  linear/rank sensitivity and bootstrap intervals; checksummed resumable ensembles;
  and a CLI resource benchmark that explicitly makes no hard real-time claim.
- A project-aware desktop tab for open/save/validation, approachable scenario details,
  background execution and cooperative cancellation, immutable run history,
  compatible-run comparison, and self-contained report opening while retaining all
  existing rocket, planetary-tour, and astronomy workflows.
- A bounded localhost-only UDP adapter for the existing versioned future-HIL packet
  codec, with exact source filtering, CRC/type/sequence gates, fail-silent command
  watchdog behavior, observable counters, and explicit non-real-time/non-HIL scope.
- A portable environment diagnostic with catalog-integrity and writable-location
  checks, optional-tool status, remediation beside every issue, and a double-click
  launcher; plus a measured Tk/PySide6/local-web architecture decision and offline,
  accessibility-oriented web comparison prototype.

### Changed

- The bundled portfolio project now composes five verified atmospheric, rotating,
  staging/recovery, and orbit-tour workflows behind the same immutable result store.
- The compact reference suite now includes rotating-planet 6-DOF, multistage
  recovery, and localhost UDP evidence, for 45 deterministic PNG/JSON artifacts.
- SQLite result-index connections now close explicitly after every transaction or
  query; warning-as-error project and ensemble tests verify clean resource ownership.
- Package, CLI, citation, FMI contract, README, validation record, architecture, and
  traceability metadata advanced to version 0.7.0.

### Verification

- 368 deterministic tests pass with 81.40% branch-aware package coverage against the
  enforced 75% threshold; Ruff, strict mypy, editable installation, and dependency
  checks pass.
- The project-aware six-page workbench services, cancellation, immutable run store,
  comparison, and self-contained reports pass focused unit and integration tests.
- The 100-sample localhost UDP run accepts all 100 state and command packets with no
  CRC, source, type, sequence, timeout, or watchdog rejection; this remains neither
  hard-real-time nor physical-HIL evidence.
- The full hidden production workbench (six pages, five scenarios, local catalog), a
  Tk skeleton, and the offline localhost-web comparison were constructed and
  measured; PySide6 was unavailable and no Qt result is claimed.
- All 45 compact reference PNG/JSON artifacts have identical SHA-256 hashes across
  two consecutive complete generations. No proprietary data or external-validation
  result is fabricated.
- MATLAB R2024a re-execution passes the constant-force and independent `ode113`
  two-body tolerances, and all four MATLAB/Simulink-interface scripts have zero code-
  analyzer findings. A Simulink licence response exists but product files remain
  absent, so no Simulink execution is claimed.

## [0.6.0] - 2026-07-19

### Added

- A unified native Simulation Workbench with beginner-readable SI-unit inputs,
  verified/resettable rocket and capture-orbit-departure presets, background solver
  execution, numerical summaries, direct 3D playback, Milky Way catalog filters,
  sourced Solar System data, and a pickable 3D exoplanet-host explorer.
- A dedicated `workbench` CLI command and `run_simulation.bat`; `run_aerognc.bat` now
  opens the unified interface while `run_interplanetary.bat` retains direct access to
  the advanced Mission Designer.
- Wrap-aware packet acceptance, seeded duplicate injection, fail-silent timeout, and
  a dual-link logical-time `software-loopback` workflow with deterministic transport,
  deadline, watchdog, and command-checksum evidence for future HIL preparation.
- A deterministic FMI 3.0 Co-Simulation controller interface contract and generator,
  paired with an explicit record that no FMU binary, official schema validation, or
  runtime import has been completed.

### Changed

- Package, CLI, citation, README, architecture, verification plan, validation record,
  and requirements traceability advanced to version 0.6.0.
- The easy Windows launch path now opens the unified workbench; the advanced Mission
  Designer remains directly available through `run_interplanetary.bat`.
- The compact reference suite now includes deterministic software-loopback evidence.

### Verification

- 256 deterministic tests pass with 79.95% branch-aware package coverage against the
  enforced 75% threshold; Ruff, strict mypy, editable installation, and dependency
  checks pass.
- Native construction covers all five workbench tabs; visual inspection covers the
  start, rocket, planetary-tour, and astronomy-data flows. Numerical rocket, tour,
  catalog, and 3D-explorer services are separately unit-tested.
- The 500-sample seeded loopback accepts 499 state and 498 command packets with zero
  transport drops and zero deadline misses; two startup watchdog activations safely
  command zero output.
- The generated FMI contract declares 24 uniquely referenced scalar variables and
  retains `fmu_built=false`, `fmu_executed=false`, and
  `official_xsd_validation_executed=false`.
- All 40 compact reference PNG/JSON artifacts have identical SHA-256 hashes across
  two consecutive full generations. Physical HIL and FMI runtime execution remain
  explicitly unclaimed.

## [0.5.0] - 2026-07-19

### Added

- A preliminary two-body patched-conic orbit-assisted civilian tour with analytical
  sphere-of-influence branches, departure injection, assist capture, integer parking
  revolutions, conservative asymptote-plane alignment, periapsis departure,
  destination capture, sequential rocket-equation accounting, and Oberth-energy
  diagnostics.
- A deterministic constrained launch-window workflow combining a complete
  departure/arrival Lambert grid with bounded manual coordinate refinement,
  explicit infeasible cells, C3/arrival-speed constraints, and an independent
  universal-propagation endpoint check.
- Explicit HELIOS_ECLIPJ2000/J2000 state transforms and provenance-tagged UTC, TAI,
  TT, Julian-date, and preliminary TDB conversions using IERS Bulletin C 72.
- CCSDS 502.0-B-3 OEM 3.0/KVN write/read support with an explicit SI-to-km exchange
  boundary, plus honest GMAT script/report and optional SPICE detection interfaces.
- A checksummed 2026-07-19 NASA Exoplanet Archive snapshot containing 6,324 confirmed
  planets, approximate sourced Milky Way context, the eight Solar System planets,
  directly implemented ICRS/Galactic transforms, deterministic filters, CLI reports,
  and a selection-bias-aware catalog figure.
- Dedicated orbit-tour, launch-window, interoperability, and galaxy-catalog
  engineering documents and requirement traces.

### Changed

- Package, CLI, requirements baseline, citation, README, architecture, verification
  plan, and validation record advanced to version 0.5.0.
- The compact reference suite now includes the orbit tour, launch-window screen, and
  confirmed-exoplanet context without committing transient trajectory datasets.

### Verification

- 222 deterministic tests pass with 79.32% branch-aware package coverage against the
  enforced 75% threshold; Ruff, strict mypy, and dependency checks pass.
- The orbit tour closes Lambert endpoints below 0.1 m, preserves ordered events and
  two parking revolutions, totals 31.456 km/s ideal delta-v, and ends at 8,975.65 kg
  above its 8,000 kg synthetic dry mass.
- The launch-window workflow performs 210 deterministic evaluations, improves the
  best feasible grid point to 7.312386 km/s, converges within 600 s epoch tolerance,
  and reproduces the selected endpoint to 0.00413 m.
- The bundled exoplanet CSV matches its recorded SHA-256 and row count; 6,297 of
  6,324 rows have complete distance/ICRS inputs for the heliocentric view.
- All 39 compact reference PNG/JSON artifacts have identical SHA-256 hashes across
  two consecutive full generations. GMAT, SPICE, Simulink, physical-flight, and
  physical-HIL results remain explicitly unexecuted where unavailable.

## [0.4.0] - 2026-07-19

### Added

- Rotating-oblate-planet strapdown inertial mechanisation with J2/local gravity,
  Coriolis and transport terms, and directly implemented two-sample coning/sculling
  compensation.
- A delayed 15-state quaternion error-state EKF with fixed-lag snapshots,
  out-of-sequence correction and deterministic replay, NIS gates, stale-data
  rejection, Joseph covariance update, and sensor-health degradation/recovery.
- Reusable deterministic sensor bias-step, spike, stuck, and dropout injection plus a
  22 s configured navigation case with observability rank and seeded NIS/NEES
  consistency evidence.
- A measurement-only asynchronous flight-data workflow with affine clock recovery,
  gap-preserving resampling, detrended Hampel cleaning, local-polynomial derivatives,
  manual Huber IRLS, physical parameter/covariance mapping, residual tests, and a
  truth-independent held-out forward prediction.
- Strict YAML, CLI commands, unit/integration tests, machine-readable reports,
  engineering documentation, and two reviewed reference figures for these workflows.

### Changed

- Package, CLI, requirements baseline, citation, README, architecture, verification
  plan, and validation record advanced to version 0.4.0.
- The reference set now includes advanced-navigation and flight-data-identification
  reports/figures while still excluding large transient CSV records.

### Verification

- 193 deterministic tests pass with 78.71% branch-aware package coverage against the
  enforced 75% threshold; Ruff, strict mypy, and dependency checks pass.
- The configured advanced filter achieves 2.490 m position RMS, 0.290 m/s velocity
  RMS, and 1.811 deg attitude RMS; exercises 18-step replay; and reports full local
  15-state observability with all declared consistency/integrity checks passing.
- The flight-data case recovers the 0.370 s/85 ppm sensor clock, estimates every
  synthetic physical parameter within 0.4%, obtains R-squared 0.9983, and predicts
  held-out pitch/rate within 0.077 deg and 0.072 deg/s RMS.
- All 33 compact reference PNG/JSON artifacts have identical SHA-256 hashes across
  two consecutive full generations. No new MATLAB, Simulink, operational-ephemeris,
  physical-flight, or physical-HIL result is claimed.

## [0.3.0] - 2026-07-19

### Added

- Oblate reference-ellipsoid geodesy with robust geodetic/ECEF conversion,
  ECEF/local-NED position maps, inertial/fixed position-velocity transforms, local
  rotation/transport rates, and near-pole tests.
- A fictional rotating Orbis-A planet with central-plus-J2 gravity, separately
  testable Coriolis/centrifugal terms, and a strict configured ECEF ascent workflow
  with geodetic/NED logging, events, CLI, and reviewed figure.
- General regular-grid N-dimensional multilinear interpolation with analytic in-cell
  gradients and explicit error/clamp/extrapolation behavior.
- A strict long-form aerodynamic CSV database supporting named Mach, angle, rate,
  control, and Reynolds axes; six common coefficients; tensor-grid completeness;
  query diagnostics; and SHA-256 provenance.
- A configured aerodynamic audit/flight case with coefficient slices, table-derived
  Jacobian, machine report, and provider-compatible Monte Carlo drag dispersion.
- A 36-point Mach-altitude-mass nonlinear trim and linearisation workflow with modes,
  rank checks, Hamiltonian LQR, actuator authority, trilinear gain scheduling, 12
  between-grid checks, and 120 seeded uncertain-model checks.
- A separated offline/online constrained research-ascent workflow with deterministic
  coordinate search, max-Q/load/loaded-angle governance, desired-apogee requirement,
  pitch lag/rate limit, throttle-consistent propellant flow, event logs, reports, and
  reviewed comparison figure.
- Dedicated geodesy, aerodynamic-database, flight-envelope, and constrained-ascent
  engineering documentation plus four new compact README figures and reports.

### Changed

- Package, CLI, requirements baseline, citation, and validation record advanced to
  version 0.3.0.
- Architecture and normative coordinate/equation documents now distinguish baseline
  local NED, optional rotating ECEF, and primary-centred inertial dynamics.
- Aerodynamic consumers now accept either the transparent legacy model or a common
  multidimensional provider without hidden library dynamics.

### Verification

- 168 deterministic tests pass with 77.01% branch-aware package coverage against the
  enforced 75% threshold; Ruff and strict mypy pass.
- All 36 envelope trims converge; every design/interpolated/seeded-uncertain closed
  loop is stable and minimum unused actuator position is about 99.23%.
- The constrained case meets (800\pm18\) m desired apogee, 12 kPa max-Q, 6.0 g0
  proper-load, and 5 deg loaded-angle requirements while preserving dry-mass and
  throttle/depletion consistency.
- MATLAB results remain the two previously executed independent benchmarks; no new
  MATLAB, Simulink, or physical-HIL result is claimed.

## [0.2.0] - 2026-07-19

### Added

- A native guided Mission Designer with unit-labelled route, epoch, spacecraft,
  maneuver, constraint, launch-window, uncertainty and results tabs; immediate 3D
  playback; a verified N-body button; readable YAML export; and two root Windows
  launchers.
- Direct universal-variable Kepler propagation with a robust bracket fallback,
  classical orbital-element conversion, elliptical synthetic ephemerides,
  zero-revolution Lambert transfers, C3/arrival porkchop grids, B-plane geometry,
  single-assist design and finite-difference epoch correction.
- Exact-time inertial/RTN impulses, finite burns, rocket-equation propellant use,
  hard dry-mass enforcement, maneuver event logs, and mission mass histories.
- Explicit optional J2, solar-radiation pressure and relativistic perturbations;
  Hill/Laplace sphere utilities; analytical/tabular/optional-SPICE ephemeris
  interfaces; and a separate mutual full N-body conservation model.
- Seeded interplanetary uncertainty with ordered parallel repeatability, graceful
  failures, percentiles, central 95% bounds, correlations, and worst-case indices.
- Nonlinear trim, central-difference linearisation, modes,
  controllability/observability, manually implemented Hamiltonian LQR, configured
  gain schedules, state-space frequency margins, flight-data system identification,
  and measured controller SIL timing.
- A documented 15-state NED/FRD quaternion error-state filter with IMU propagation,
  GNSS-like position/velocity and barometric updates, Joseph covariance, and bias
  states.
- Executed MATLAB R2024a adaptive `ode113` two-body validation against the Python
  universal propagator, in addition to the existing constant-force comparison.

### Changed

- Package and CLI version advanced to 0.2.0.
- `run_interplanetary.bat` now opens the guided designer immediately; the original
  restricted N-body CLI remains available unchanged.
- Requirements, traceability, architecture, governing equations, navigation,
  control, interplanetary, MATLAB, validation and README documentation now cover the
  expanded engineering evidence and fidelity boundaries.

### Verification

- 131 deterministic tests pass with 75.21% branch-aware core/package coverage;
  Ruff formatting/lint, strict practical mypy, and dependency consistency pass.
- All 21 compact reference artifacts reproduce with identical SHA-256 hashes.
- MATLAB R2024a reports zero code-analyzer findings, and both actually executed
  cross-language benchmarks pass their declared tolerances. Simulink remains
  explicitly unexecuted.

## [0.1.0] - 2026-07-19

### Added

- Initial requirements baseline, verification plan, traceability matrix, and
  modular architecture for the first public portfolio release.
- Explicit NED/FRD, quaternion, force, moment, state, altitude, and SI-unit
  conventions.
- Validated vector/skew utilities, Hamilton quaternion and coordinate transforms,
  one- and two-dimensional interpolation tables, and custom event-aware fixed-step
  RK4 integration.
- Analytical rotation cases, RK4 convergence evidence, and an independent SciPy
  solver comparison (23 focused tests passing at this milestone).
- Strict YAML configuration loading; lower-atmosphere ISA, gravity and deterministic
  seeded-wind models; synthetic propulsion, mass properties, replaceable aerodynamic
  coefficients, and bounded delayed actuators (41 unit tests passing at this milestone).
- Event-driven 3-DOF NED ascent, deterministic CSV/JSON output, CLI quick start, and
  visually inspected kinematics/load/trajectory figures. The nominal case detects
  burnout, apogee, and impact and completes below the 5 s runtime requirement.
- Nonlinear quaternion 6-DOF EOM with variable inertia, analytical force/torque and
  conservation cases, 3-DOF cross-model consistency, a configured closed-loop ascent,
  and a sign-convention correction found by composed-flight verification.
- Time-indexed safe ascent references, anti-windup PID, cascaded attitude/rate loops,
  quaternion PD hold, manual Ackermann pole placement, optional gain scheduling, and
  quantitative/visually inspected controller comparisons.
- Seeded multi-rate sensor models with noise, bias/random walk, quantisation, delay,
  and dropout; a three-state vertical navigation filter; acquisition-time delay
  compensation; covariance/error figures; and deterministic navigation validation.
- Coupled flight/navigation/control Monte Carlo with ordered seed trees, process
  parallelism, graceful failures, percentiles/mean confidence intervals,
  correlations, requirement margins/pass rates, and automatic worst-case runs.
- Measurement-only synthetic flight-test CSV generation/reload, strict schema checks,
  event reconstruction, performance comparison, and automatic summary/figures.
- Executed Python/MATLAB constant-force cross-validation using a shared JSON case;
  supplied an unexecuted Simulink attitude-channel model builder because Simulink is
  not installed in the development environment.
- Versioned, checksummed future HIL packet definitions and deterministic software-only
  latency, jitter, and loss emulation, with no claim of physical hardware testing.
- A clean-environment installation/CLI audit, 107 passing tests, 78.79% branch-aware
  coverage, strict mypy, Ruff, link/traceability audits, byte-identical repeated
  reference generation, and Python 3.12/3.13 CI.
- Interactive 3-DOF flight playback with a moving trajectory/vehicle display, live
  SI telemetry, event and phase status, pause/restart/seek/speed controls, a documented
  desktop command, and deterministic optional GIF export.
- Interactive quaternion 6-DOF 3D playback with East/North/altitude trajectory,
  quaternion-oriented vehicle/body axes, five camera modes, synchronized controls and
  telemetry, a documented desktop command, and deterministic headless GIF export.
- Directly implemented circular planetary ephemerides, Hohmann and hyperbolic-flyby
  equations, primary-centred restricted N-body dynamics with the indirect term,
  encounter-adaptive custom RK4, deterministic interplanetary events, and SI logs.
- A tuned fictional Asteria–Brontes–Caelus civilian mission that safely clears the
  assist body, demonstrates primary-frame speed/energy gain with conserved
  planet-relative boundary speed, and enters the destination corridor.
- A redesigned dark 3D mission-control UI with live ephemerides, trajectory and
  velocity displays, energy history, six cameras, next-event navigation, playback
  controls, PNG snapshots, and deterministic headless GIF export.
- A root-level Windows batch launcher for immediate double-click startup of the
  interplanetary mission, with portable path resolution and clear setup errors.

- Planned first locally publication-ready release.
