# Verified Mission-Analysis Building Blocks

## Scope

This layer supports public, civilian orbit-design studies around fictional or
user-defined bodies. It is not a flight-certified optimiser. It contains no target
tracking, interception, homing, or operational engagement model. All internal
quantities use SI units; every ephemeris state carries an explicit frame, centre,
time system, source, and coverage interval.

## Coverage-aware ephemerides

`CoverageAwareEphemeris` is the common analytical, tabulated, and optional SPICE
boundary. A request outside the declared body or epoch interval raises
`EphemerisCoverageError`. The SPICE adapter also propagates missing-kernel and
provider errors; it never replaces unavailable data with an analytical orbit.

The tabulated provider applies cubic Hermite interpolation to position and endpoint
velocity. The returned velocity is the analytical derivative of the same position
polynomial, avoiding an inconsistent independently interpolated state.

## Multi-revolution transfer search

For endpoint vectors \(\mathbf r_1\) and \(\mathbf r_2\), the direct universal-variable
Lambert implementation scans each requested integer-revolution interval, isolates
every feasible time-of-flight root, and recovers the Lagrange coefficients

\[
 \mathbf v_1 = \frac{\mathbf r_2-f\mathbf r_1}{g}, \qquad
 \mathbf v_2 = \frac{\dot g\mathbf r_2-\mathbf r_1}{g}.
\]

The search evaluates prograde and retrograde geometry independently. Every accepted
branch is then propagated with the separate universal Kepler solver. A branch is
discarded if its propagated terminal position exceeds the declared endpoint
tolerance. Feasible branches are ranked by the selected departure, arrival, or
combined excess-speed objective with deterministic tie breaking. Revolution count
is bounded to 0--12 to keep an interactive engineering search finite.

This is a trajectory-screening method, not a global finite-thrust optimiser. It does
not include launch-site, thermal, communications, radiation, or propulsion duty-cycle
constraints unless the caller evaluates them separately.

## Finite burns

`execute_two_body_finite_burn` divides the propagation exactly at burn start and stop.
During the burn,

\[
 \dot{\mathbf r}=\mathbf v,\qquad
 \dot{\mathbf v}=-\mu\frac{\mathbf r}{\|\mathbf r\|^3}
 +\frac{T}{m}\hat{\mathbf u},\qquad
 \dot m=-\frac{T}{I_{sp}g_0}.
\]

Coast and burn segments use the directly implemented adaptive Dormand--Prince 5(4)
solver. Propellant availability is checked before propagation, dry mass is a hard
floor, start/stop occurrences are retained, and numerical mass use is compared with
the commanded mass-flow integral.

## Visibility and access

The geometry module provides finite-segment spherical occultation, aggregate
line-of-sight, apparent-disc sunlit/penumbra/umbra classification, spherical
ground-station elevation, and linearly interpolated rise/set crossings. Bodies and
stations declare their Cartesian frame; mixed-frame inputs are rejected. The model
is useful for early contact and eclipse screening but omits ellipsoidal terrain,
refraction, antenna patterns, link budgets, and detailed celestial body shapes.

## CCSDS exchange subset

The interoperability package reads and writes deliberately scoped KVN subsets:

- AEM 2.0 quaternion histories, scalar-first, with explicit reference frames;
- OPM 3.0 Cartesian state and mass records;
- TDM 2.0 sequential range, angle, and Doppler observations.

Mandatory metadata, fictional-object declaration, time system, monotonic epochs,
coverage, finite values, and quaternion norm are validated. SI values are converted
at the file boundary where CCSDS exchange units require kilometres or degrees. These
parsers do not claim full CCSDS standard conformance and intentionally reject
unsupported constructs instead of silently ignoring them.

## Verification

Unit tests cover provider coverage failures, Hermite consistency, missing SPICE,
zero/one-revolution roots, deterministic ranking, analytical occultation/eclipses,
station horizon crossings, finite-burn mass balance, and AEM/OPM/TDM round trips.
`tests/integration/test_mission_analysis.py` composes transfer search, independent
endpoint propagation, tabulated ephemeris, rotating-body station access, and finite
burn execution in one deterministic civilian-orbit case.
