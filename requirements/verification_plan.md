# Verification Plan

## Purpose and independence

Verification asks whether each implementation satisfies its measurable requirement;
validation asks whether the chosen simplified model is suitable for the documented
research scenario. Evidence is planned before advanced implementation. Plots may
support interpretation but never constitute the sole pass criterion.

## Methods

| Code | Method | Typical evidence |
|---|---|---|
| A | Analysis | Dimensional/sign review, boundedness, requirement inspection |
| T | Automated test | Unit or integration assertion with explicit tolerance |
| C | Comparison | Analytical solution or independently implemented/reference solver |
| D | Demonstration | Reproducible CLI workflow and generated artefact inspection |

## Test levels

1. **Unit:** deterministic functions, reference values, invalid input paths, signs,
   limits, and invariants.
2. **Integration:** configured model composition, event sequencing, logging, CLI,
   sensor/filter timing, and controller/actuator interaction.
3. **Validation:** analytical ballistic and rigid-body cases, RK4 convergence,
   independent SciPy integration, cross-model consistency, and optional MATLAB.
4. **Regression:** compact reference metrics with engineering tolerances rather than
   platform-fragile byte-for-byte figure or floating-point comparisons.
5. **Requirement audit:** traceability matrix contains implementation, test, method,
   and status for every baselined identifier.

## Numerical tolerance policy

- Exact algebraic identities: absolute tolerance \(10^{-12}\) where conditioning
  permits.
- Integrated analytical cases: tolerance derived from step size and demonstrated
  convergence, typically relative \(10^{-5}\) to \(10^{-3}\).
- ISA reference values: within the published simplified-model rounding tolerance.
- Cross-tool results: tolerance is stated with the actual execution record; a result
  is never entered when the external tool was not run.
- Stochastic tests compare same-seed reproducibility and distribution-level
  properties, never a claim of physical flight dispersion.

## Planned benchmark cases

- Quaternion identity, principal-axis 90-degree rotations, composition, inverse,
  norm, and nonsingular Euler round trips.
- Exponential ODE RK4 error-ratio study and `scipy.integrate.solve_ivp` comparison.
- Dormand--Prince embedded-error convergence/reference checks, dense directed-event
  location, exact checkpoint restart, projected-state invariants, deterministic
  multi-rate order, and finite-difference state/parameter sensitivity cases.
- ISA sea-level and layer-boundary values.
- Vacuum ballistic trajectory and constant acceleration.
- Force-free and constant-force 6-DOF translation; torque-free spherical/symmetric
  body and constant principal-axis torque.
- Simplified 3-DOF/6-DOF translational equivalence.
- Step/disturbance attitude-control response with quantitative limits.
- Same-seed Monte Carlo identity and different-seed non-identity.
- Synthetic flight record event reconstruction against truth.
- Universal Kepler circular orbit, classical-element round trip, Lambert endpoint,
  launch-window repeatability, B-plane geometry, and MATLAB adaptive two-body check.
- Ephemeris coverage/frame rejection, zero/multi-revolution Lambert branch closure,
  exact-boundary finite-burn mass balance, eclipse/access crossing geometry, and
  strict CCSDS AEM/OPM/TDM round trips.
- Hyperbolic sphere-of-influence identities, sequential rocket-equation accounting,
  integer parking-orbit dwell, orbit-tour event order, bounded launch-window
  refinement, and independent universal-propagation closure.
- UTC/TAI/TT/TDB offset checks at leap boundaries, orthonormal ecliptic/equatorial
  frame round trips, CCSDS OEM SI/exchange-unit round trips, and strict parsing of an
  independently executed GMAT report interface without claiming execution.
- NASA catalog SHA/count/order/uniqueness checks, Milky Way/Solar System provenance,
  ICRS/Galactic centre and round-trip cases, deterministic catalog filters, and
  missing-data-aware report/figure generation.
- Maneuver rocket-equation/mass-flow checks, optional-force direction/scale checks,
  and full N-body momentum/energy conservation.
- Nonlinear trim and finite-difference Jacobian analytical cases; Hamiltonian LQR
  comparison with SciPy; known SISO phase margin; oscillator system identification;
  and measured SIL deadline statistics.
- Waypoint straight-flight trim convergence/failure policy, bumpless total-energy
  control, reference and actuator limits, tangent fillet/orbit geometry, estimated-
  state envelope margins, and a deterministic two-plant mission in steady crosswind.
- Stationary and rotating 15-state ESKF propagation plus GNSS/barometer covariance
  reduction and positive-semidefinite checks.
- Mission Designer backend path/mass/event effects and CLI dispatch; visual desktop
  smoke inspection remains demonstration evidence rather than a numerical test.
- Portable project path/schema checks, workflow-version/discovery isolation, atomic
  manifest/artifact integrity, index rebuild, compatible comparison, offline report,
  cancellation, and bundled five-workflow execution.
- Unified workbench input-domain services, one-click prepared examples, default-hidden
  specialist inputs, plain-language result summaries, project/run operations, editable
  rocket/orbit/aircraft/tour execution, catalog filters, 3D host grouping, CLI
  dispatch, and native eight-page Tk widget-construction smoke.
- Future-HIL CRC/type/length checks, wrap-aware stale/duplicate rejection, seeded
  dual-link logical-time loopback, exact-source localhost UDP, bounded receive
  timeout, deadline/watchdog behavior, and fail-silent zero command; FMI variable-
  contract checks remain distinct from unexecuted FMU runtime and official schema
  validation.
- Geodetic/ECEF equator, mid-latitude and near-pole round trips; inertial/fixed state
  round trips; Coriolis/centrifugal signs; \(J_2\) equatorial/polar behavior; and a
  configured rotating-planet ascent with ordered events.
- Multidimensional multilinear interpolation/gradient fields, boundary policies,
  malformed aerodynamic grids, provenance hash, common-provider flight composition,
  and provider-compatible Monte Carlo drag dispersion.
- A 36-point nonlinear trim/linearisation/control grid, between-grid schedule
  stability, actuator authority margins, and same-seed uncertain-model screening.
- Reference-versus-governed ascent constraints, apogee tolerance, throttle/mass-flow
  consistency, dry-mass floor, event order, and deterministic optimizer evidence.
- Rotating-frame strapdown stationary/maneuver propagation, two-sample coning and
  sculling, delayed fixed-lag measurement replay, NIS rejection/health recovery,
  covariance PSD, full local observability rank, and fixed-seed NIS/NEES consistency.
- Affine clock-marker recovery, gap-preserving resampling, detrended Hampel rejection,
  local-polynomial differentiation, Huber regression, physical-parameter covariance,
  residual whiteness, and truth-independent held-out forward prediction.
- Strict telemetry schema/unit/quality/provenance checks, affine time alignment,
  gap-aware residual/whiteness metrics, and RTS covariance/error reduction.
- LHS stratum coverage, Sobol deterministic prefixes, Morris elementary effects,
  Pearson/rank screening, seeded bootstrap intervals, checksummed resume/corruption/
  failure behavior, worker invariance, and explicit local resource budgets.
- Diagnostic package/data/hash/result-permission status with actionable remediation,
  plus a scoped Tk/PySide6/local-web architecture comparison without fabricated
  availability or execution claims.
- Force-free satellite analytical translation, one-period two-body closure,
  restricted-three-body and full-N-body finite propagation, scalable drag-energy
  loss, finite-horizon lifetime wording, and strict public-safe orbit configuration.
- ISA-to-thermosphere reference density monotonicity, declared 200 km value, explicit
  density scaling, and bounded high-altitude tail behavior.
- Fixed-wing coefficient signs and sensitivities, post-stall lift loss/drag rise,
  theoretical stall-speed identity, actuator lag/rate limit, hands-off trimmed flight,
  aileron-pulse bank/actual turn rate, and propagated coefficient-path sensitivity.
- Deterministic 100 km fictional research-ascent boundary crossing under calculated
  fuel depletion, thrust, mass, attitude, atmosphere, and gravity, without treating
  the result as orbital insertion or a real-aircraft feasibility claim.
- Bounded OBJ polygon triangulation, ASCII/binary STL import, source-axis conversion,
  optional XInput normalization/disconnection, hidden live-player construction, and
  deterministic keyboard-command state advancement.
- Hidden construction of all eight beginner-labelled workbench pages with orbit,
  aircraft, rocket, and planet-trip specialist inputs collapsed by default.

## Execution and records

The default gate is:

```bash
ruff check .
mypy src
pytest --cov=aerognc --cov-report=term-missing
python -m aerognc.cli run --config configs/three_dof_nominal.yaml
```

Optional MATLAB evidence records the release, command, output hashes/metrics, and
tolerance in `docs/matlab_validation.md`. Simulink and physical HIL remain explicitly
unverified unless executed on available software/hardware.
