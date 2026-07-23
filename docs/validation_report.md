# Validation Report

**Release candidate:** 0.8.0

**Evidence date:** 2026-07-23

**Local environment:** Windows, CPython 3.13.1, NumPy 2.5.1, SciPy 1.18.0,
MATLAB R2024a Update 3

## Executive result

The Python core satisfies the local publication-candidate acceptance cases: nominal
3-DOF ascent, bounded quaternion 6-DOF ascent, closed-loop attitude control, linear
flight-control analysis, vertical and error-state navigation, reproducible Monte
Carlo, guided patched-conic mission design, restricted/full N-body propagation,
synthetic flight-data reconstruction, rotating-planet translation, multidimensional
aerodynamic data, envelope trim/scheduled control, constrained ascent, rotating-frame
strapdown navigation, delayed fixed-lag filtering, asynchronous log alignment, robust
physical identification, deterministic reference artifacts, and two independent
MATLAB comparisons. It now also covers orbit-assisted capture/dwell/departure
accounting, constrained launch-window refinement, time/frame/OEM interoperability,
and a checksummed NASA confirmed-exoplanet context layer. Version 0.7 additionally
covers portable project/run evidence, adaptive integration, checkpoint/restart,
multi-rate and sensitivity mathematics, rotating quaternion flight, staged recovery,
expanded ephemeris/Lambert/burn/visibility analysis, AEM/OPM/TDM exchange, strict
telemetry ingestion and RTS smoothing, robust experiment design and resumable
ensembles, a project-aware workbench, localhost UDP packet exchange, and an
environment diagnostic. Version 0.8 adds a near-planet satellite sandbox spanning
force-free through full N-body and drag-decay models, plus a fictional
coefficient-driven 18-state aircraft with live pilot input, visual mesh import, and
a separately verified 100 km research-ascent case. The current branch additionally
covers waypoint-based fixed-wing guidance, path management, mission sequencing,
safety envelopes, deterministic simulation, structured logs, a selectable nonlinear
18-state backend, and matched cross-model evidence. The release does not claim
physical-flight, real-HIL,
operational-ephemeris, GMAT, SPICE, Simulink, FMU runtime, or official FMI-schema
validation.

## Representative executed evidence

| Area | Executed result | Acceptance |
|---|---|---|
| Automated suite | 635 deterministic tests | All pass |
| Branch coverage | 80.66% core/package coverage | Enforced minimum 75% |
| Waypoint backend comparison | Reduced 247.25 s / 151.714 m max cross-track; coefficient 180.30 s / 96.631 m; 0.403 m terminal horizontal separation; 1.371 duration ratio | Both complete without safety intervention; every declared cross-model bound passes |
| Adaptive numerical core | DP5(4) convergence/reference cases, dense directed events, checkpoint hashes, deterministic task order, and finite-difference sensitivities | Every numerical/tolerance/integrity assertion passes |
| Nominal 3-DOF | Burnout 3.350 s; apogee 1101.49 m at 15.569 s; impact 31.794 s | Ordered events and bounded mass |
| Interactive playback | Pause/restart/seek/speed state tests, PNG frame, and headless GIF export pass | Source trajectory remains unchanged |
| Quaternion 3D playback | Five camera modes, attitude glyph, controls, PNG frame, and headless GIF export pass | Logged 6-DOF state remains unchanged |
| Direct mission design | Lambert endpoint, correction-maneuver path/mass effect, feasibility, and deterministic planning pass | Requested body and epoch constraints enforced |
| Interplanetary mission | Brontes centre distance about 371,000 km; +10.1 km/s heliocentric boundary gain; Caelus corridor day 1843.5 | Clear body, gain >9 km/s, relative-speed change <2 m/s, arrive |
| Orbit-assisted tour | Two Neria parking revolutions; 31.456 km/s total ideal burns; 8,975.65 kg final mass | Event/SOI/revolution/Lambert/delta-v/dry-mass assertions pass |
| Launch-window optimization | 210 unique evaluations; 7.312386 km/s refined cost; 0.00413 m endpoint error | Convergence, feasibility, constraints, non-worsening, and closure pass |
| Time/frame/OEM | Bulletin C 72 offsets, J2000/ecliptic round trips, and OEM SI/km round trips | Automated tolerances pass; table validity is explicit |
| Mission-analysis boundary | Multi-revolution endpoints, ephemeris coverage, finite-burn mass balance, visibility/access, and AEM/OPM/TDM strict round trips | All bounded-domain, endpoint, mass, geometry, and exchange assertions pass |
| Milky Way/catalog data | 6,324 rows and SHA-256 match metadata; 6,297 positioned rows; Galactic-centre transform passes | Provenance, filters, missingness, and scope assertions pass |
| Mission Designer | CLI dispatch test and responsive live-window smoke check | Unit-labelled inputs and background work remain usable |
| Engineering projects | Five configured workflows; immutable manifests/artifact hashes; compatible comparison; offline report; registry isolation; cancellation | Schema, provenance, storage, reload, comparison, and failure-path assertions pass |
| Simulation Workbench | Eight purpose-labelled pages construct; one-click rocket/orbit/aircraft/tour paths, default-hidden specialist fields, scrollable 1000 x 780 geometry, plain-language summaries, project/run, catalog, and 3D flows inspected | Inputs remain unit-labelled, resettable, reachable, background-safe, and solver-safe |
| Software loopback | 500 samples; 499 state and 498 command packets accepted; zero transport drops and deadline misses; two startup watchdog activations | Deterministic checksum and fail-silent zero command pass |
| Localhost UDP | 100 state and 100 command datagrams accepted with zero timeout/source/CRC/type/sequence/watchdog rejection | OS-socket packet boundary passes; hard real-time and physical HIL remain unclaimed |
| UI architecture probe | Production eight-page/five-scenario/catalog workbench 5.782 s/78.425 MB Python-allocation peak; Tk skeleton 0.087 s/0.973 MB; local-web shell 0.045 s/0.247 MB; PySide6 absent | Scoped construction probes pass; values are not RSS or acceptance limits and no Qt result is claimed |
| Environment diagnostic | Python/package/core data/catalog hash/result-write checks pass; MATLAB detected; GMAT, FFmpeg, and SPICE unavailable | Overall READY; optional gaps carry remediation and do not masquerade as execution |
| FMI interface contract | Deterministic FMI 3.0 Co-Simulation XML; 24 unique scalar variables with causality, units, dependencies, and adjacent status | Project contract checks pass; FMU build/import and official XSD execution remain false |
| 3-DOF runtime | Below 1 s in repeated local runs, excluding plotting/startup | Less than 5 s |
| 6-DOF ascent | 8 s finite trajectory; final altitude about 913 m; maximum attitude error about 5.12 deg | Quaternion error below \(10^{-10}\); attitude below 10 deg |
| Cascaded PID | Settling 2.495 s; overshoot 7.86%; disturbance recovery 0.94 s | Less than 3 s, 15%, and 2 s |
| Linear flight control | Trim residual \(5.55\times10^{-17}\) N m; Riccati residual \(2.16\times10^{-14}\); phase margin 66.9 deg | Converged trim, stable closed loop, no 1 ms SIL misses in 10,000 calls |
| Rotating geodesy/dynamics | Equator/mid-latitude/near-pole and inertial/ECEF round trips; separated frame-term signs; 3.350 s burnout, about 1091.8 m apogee | All numerical/frame/event assertions pass |
| Rotating quaternion 6-DOF | 911.31 m altitude at 8 s; 5.099 deg maximum attitude error; (2.22\times10^{-16}) maximum quaternion-norm error | Inertial/fixed/NED composition, rotating wind, J2, loads, frame outputs, and bounded attitude assertions pass |
| Multistage recovery | Ordered two-stage ignition/burnout/separation; 65.624 m apogee; 71.54 N peak opening load; 7.91 m/s ground contact | Dry-mass floors, jettison, continuous reefed/full inflation, event order, and contact assertions pass |
| Aerodynamic database | Exact multilinear values/gradients; three boundary modes; malformed grid/provenance checks; configured 3-DOF composition | All unit/integration assertions pass |
| Flight envelope | 36/36 trims; 12/12 between-grid and 120/120 uncertain closed loops stable; 99.23% minimum unused authority | Exceeds 70% authority and 0.35 damping requirements |
| Constrained ascent | 785.3 m versus 800 +/- 18 m; 8.84 kPa; 6.00 g0; 4.10 deg loaded alpha | Apogee, max-Q, proper-load, angle, and mass checks pass |
| Vertical navigation | Raw barometer RMS 3.018 m; estimate RMS about 0.453 m | Estimate improves RMS and covariance stays valid |
| Error-state navigation | Stationary and yaw-rate propagation plus GNSS/barometer corrections pass | Unit quaternion and positive-semidefinite covariance retained |
| Advanced navigation | Position RMS 2.490 m; velocity RMS 0.290 m/s; attitude RMS 1.811 deg; observability rank 15; maximum replay 18 steps | Every configured accuracy, replay, integrity, recovery and consistency assertion passes |
| Flight-data identification | Offset 0.370085 s; drift 78.19 ppm; R-squared 0.99827; held-out pitch/rate RMS 0.0763 deg/0.0719 deg/s | Every timing, data-quality, parameter, residual, and validation assertion passes |
| Telemetry and smoothing | Strict mapped CSV import, SI conversion, missing/quality policy, source/mapping hashes, affine alignment, gap residuals, and RTS covariance history | Validation/failure cases pass; smoothing reduces scoped synthetic error without truth leakage |
| Robust experiment tools | Seeded LHS/Sobol/Morris designs, Pearson/rank screening, bootstrap intervals, resumable checksummed ensemble, resource budgets | Mathematical cases, reproducibility, worker invariance, failure records, and budget logic pass |
| Monte Carlo | 12/12 numerical runs complete; fixed seed/process ordering reproducible | Statistics, margins, failures, sensitivities, worst cases reported |
| Satellite orbit sandbox | Force-free analytical match; one-period two-body closure within 100 m; restricted/full N-body finite; drag lowers energy; default 120 km threshold at 0.940 modeled days | Model terminology, deterministic propagation, exact event boundary, density behavior, and finite-horizon wording pass |
| Fictional aircraft | Stable hands-off coefficient-driven 18-state case; calculated CL/CD/Cm/mass sensitivities and post-stall break; research-ascent aid crosses 100 km | Quaternion, actuator, mesh, input, batch/CLI, stall/turn, and boundary assertions pass; no real-aircraft feasibility claim |
| Synthetic flight test | Reconstructed apogee error about 0.53 m; measurement CSV reload is truth-independent | Events and summary produced from reloaded measurements |
| MATLAB constant force | Python/MATLAB maximum state difference \(5.12\times10^{-13}\) | Below \(10^{-10}\) |
| MATLAB two body | Position error \(7.79\times10^{-8}\) m; velocity error \(1.11\times10^{-10}\) m/s | Below 0.1 m and \(10^{-4}\) m/s |

One of the twelve reference Monte Carlo members fails the deliberately defined
landing-range requirement. This is an analyzed requirement-margin outcome, not a
simulation failure; the dashboard reports the pass rate and worst run rather than
hiding it.

## Validation coverage

- Quaternion algebra includes identity, 90-degree, inverse, norm, and Euler
  round-trip tests away from singularities.
- RK4 demonstrates fourth-order convergence and agrees with tight-tolerance SciPy
  `solve_ivp` on an independent nonlinear ODE.
- Adaptive DP5(4) cases assert fifth-order solution behavior, embedded error control,
  accepted/rejected step accounting, dense event direction/termination, exact
  checkpoint restart, projected quaternion state, multi-rate order, and state/
  parameter derivative agreement.
- ISA reference values and range rejection, gravity direction, seeded wind, thrust
  impulse/depletion, mass floor, inertia definiteness, aerodynamic signs, and all
  actuator constraints are tested.
- Rigid-body benchmarks cover force-free/constant-force translation, constant torque,
  torque-free symmetric behavior, conservation/bounded error, and simplified
  3-DOF/6-DOF consistency.
- Sensor acquisition rate, delay, quantisation, dropout, bias/noise reproduction,
  filter covariance, Monte Carlo seed/worker invariance, and flight-record schema are
  asserted numerically.
- Universal Kepler propagation, classical-element round trips, Lambert endpoints,
  B-plane geometry, exact-time impulses, finite-burn mass flow, optional force signs,
  full N-body conservation, mission correction effects, and uncertainty repeatability
  are asserted numerically.
- Analytical/tabulated ephemerides enforce coverage and frame/center identity;
  multi-revolution Lambert branches close independently; finite burns honor exact
  boundaries and mass balance; visibility/access crossings and strict AEM/OPM/TDM
  parsers are covered.
- Hyperbolic sphere-of-influence patches, ideal capture/injection identities,
  sequential rocket-equation mass accounting, integer parking-orbit dwell, event
  order, launch-window determinism, constraints, and independent endpoint closure
  are asserted numerically.
- Leap/time-scale offsets, ecliptic/equatorial orthonormal transforms, CCSDS OEM
  metadata/unit round trips, and strict external-report parsing are covered without
  treating interface generation as external-tool execution.
- The NASA catalog is checked against its retrieval metadata and hash; catalog
  sorting/uniqueness/missing values, deterministic filters, Solar System/Milky Way
  provenance, and ICRS/Galactic known-direction/round-trip conversions are tested.
- Nonlinear trim, central-difference linearisation, controllability/observability,
  modal properties, Hamiltonian LQR against SciPy, frequency margin, system
  identification, and application-level SIL timing are checked.
- The 15-state error-state filter is checked for stationary and rotating propagation,
  measurement correction, covariance reduction, symmetry, and positive
  semidefiniteness.
- Geodetic/ECEF conversion includes equatorial, mid-latitude, near-pole, local-NED,
  curvature, fixed/inertial state, transport-rate, and invalid-domain cases. Central,
  J2, Coriolis, and centrifugal terms are asserted independently before composition.
- N-dimensional interpolation is checked against affine tensor fields and exact
  gradients; aerodynamic CSV completeness, duplicates, nonfinite values, axis
  domains, coefficient-provider composition, source hash, and Monte Carlo wrapping
  are covered.
- Envelope evidence asserts trim residuals, finite local matrices, ranks, Riccati
  residuals, grid/interpolation stability, control authority, seeded robustness, and
  deterministic report/figure creation.
- Constrained ascent evidence compares an ungoverned reference with the selected
  governed case, checks its declared powered/loaded requirement domains, event order,
  monotonic throttle-coupled mass, dry-mass floor, optimizer improvement, report, and
  figure.
- Advanced navigation covers rotating-frame two-sample mechanisation, delayed
  correction/replay, stale/outlier handling, injected sensor failures and recovery,
  covariance PSD, local observability rank, and fixed-seed NIS/NEES consistency.
- The flight-data evidence crosses a CSV reload boundary, verifies affine-clock
  recovery and missing-data preservation, tests robust signal/regression utilities,
  and evaluates a model forward prediction on the excluded final 30% of the record.
- Versioned telemetry ingestion rejects ambiguous units/schema/quality values and
  records source/mapping hashes. Clock alignment, gap preservation, whiteness
  residuals, and RTS covariance/error behavior are checked independently.
- Experiment evidence covers LHS strata, Sobol prefixes, Morris elementary effects,
  rank ties, seeded bootstrap bounds, definition/member hashes, resume and corrupt
  member handling, worker invariance, failed runs, and local benchmark budgets.
- Project evidence covers safe relative paths, strict schemas, versioned workflow
  compatibility, isolated discovery failures, atomic immutable data, SQLite index
  rebuild, artifact verification, comparison alignment, and standalone reports.
- Workbench evidence validates every editable input domain, immutable configuration
  construction, project validation/execution/cancellation/history/comparison/report,
  deterministic rocket/orbit/aircraft/tour execution, catalog filtering, host-star grouping, CLI
  dispatch, and construction of the eight native pages.
- Future-HIL evidence covers CRC/type/length rejection, unsigned sequence wrap,
  stale/duplicate rejection, independently seeded state/command impairments,
  logical deadlines, controller separation, exact localhost UDP source filtering,
  bounded socket timeout, and fail-silent watchdog behavior. FMI evidence checks the
  project XML contract without treating it as an executable FMU.

The final branch-aware report is 80.66% against an enforced 75% threshold. It
excludes `visualisation/mission_designer.py` and `visualisation/workbench.py`, whose Tk
event loops are exercised by CLI dispatch and live-window/widget-construction smoke
checks. The input, planning, propagation, catalog, uncertainty, and plotting services
behind the UIs remain in automated coverage. Ruff format/check, strict practical
mypy, editable installation, and `pip check` pass.

The reference-generation entry point now produces 52 compact PNG/JSON artifacts,
including the v0.3 through v0.8 workflows. Two consecutive complete generations
produced identical SHA-256 hashes for all 52 artifacts. Numerical reference records
omit measured runtime and workspace-dependent paths. GitHub Actions is configured to
repeat lint, formatting, strict typing, one canonical branch-aware coverage run,
deterministic unit and CLI smoke tests on Python 3.12, 3.13, and 3.14 plus Windows,
and a clean install of the typed wheel. Separate automation performs CodeQL,
dependency review, Python dependency auditing, pinned dependency maintenance,
tag/version checks, release acceptance, and provenance-attested trusted publication.
Local syntax validation has passed; each remote job is reported as executed only
after its corresponding GitHub check completes.

Every supplied MATLAB and Simulink-interface `.m` file passed the MATLAB R2024a code
analyzer with zero findings. The constant-force RK4 case and an independent adaptive
`ode113` two-body case were executed and passed their declared tolerances. The
Simulink model remains unexecuted because Simulink product files are absent; no
Simulink result is fabricated.
GMAT was not detected and `spiceypy`/kernels were unavailable. Their generated
interfaces and machine status records explicitly retain `executed=false`; no GMAT or
SPICE numerical result is claimed.
The FMI directory contains a deterministic `modelDescription.xml` interface contract,
not a packaged FMU: it has no mandatory model binary/source wrapper and has not been
officially schema-validated or imported into an independent FMI runtime.

## Known validation gaps

The baseline local NED 3-DOF/6-DOF frame remains flat and nonrotating. Separate ECEF
translation and planet-centred inertial quaternion 6-DOF workflows are explicit
alternatives. The baseline ISA ends at 47 km; the orbit/aircraft reference atmosphere
extends with a fixed synthetic profile through 1500 km and is not a space-weather
forecast.
aerodynamics are sparse synthetic tables; the envelope is a two-state surrogate; and
the constrained ascent is pitch-plane. Structural flexibility, slosh, aeroelasticity,
plume effects, detailed rail contact, coupled canopy/vehicle dynamics, combustion transients,
production inertial navigation, real flight-data calibration, and real-time
operating-system guarantees are outside scope.
The desktop UI is not covered by automated pixel/event-loop or certified assistive-
technology testing; native widget construction, service tests, manual inspection,
and a scoped local-web comparison provide limited presentation evidence. Simulink is
not installed and its supplied model was not run. The deterministic logical-time
loopback performs no OS I/O; the separate UDP check uses only synchronous localhost
sockets. Neither is hard-real-time, processor-in-the-loop, network-security, or
physical-HIL validation. The FMI contract is not an executable FMU.

The aircraft uses synthetic quasi-steady coefficient equations and a deterministic
post-stall break, not flight-test-calibrated stall/spin, aeroelastic, structural,
thermal, landing-gear, ground-effect, or certification models. Imported OBJ/STL
geometry is visual only. Its optional 100 km crossing is suborbital boundary evidence,
not proof of a realizable aircraft, sustainable orbit, safe trajectory, or launch
authorization. Satellite decay uses a reference density and threshold event; heating,
breakup, lift, attitude, debris, and operational lifetime prediction are omitted.

The reference interplanetary scenario uses prescribed synthetic elliptical
ephemerides and a massless regression spacecraft. Maneuver mass flow, finite burns,
J2, radiation pressure, relativity, and full mutual N-body gravity are available as
separate studies but are not silently enabled in that baseline. The Mission Designer
uses zero-revolution Lambert legs and a preliminary patched-conic one-assist search;
it is not a global optimiser. No operational ephemeris is bundled, and destination
arrival means corridor entry unless an ideal capture estimate is explicitly stated.
The separate orbit-assisted case uses ideal impulses and a deliberately conservative
plane-change estimate, not finite-burn capture execution. The exoplanet snapshot is
dated, sparse observational context and cannot supply complete interstellar or
planetary ephemerides.
