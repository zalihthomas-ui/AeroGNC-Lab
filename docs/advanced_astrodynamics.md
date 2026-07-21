# Advanced Astrodynamics Models

## Scope and conventions

All examples use fictional civilian worlds and synthetic parameters. Position and
velocity are primary-centred inertial Cartesian vectors in metres and metres per
second. Time is seconds, angular quantities are radians, mass is kilograms, and
gravitational parameters are m³/s². These models are mission-analysis teaching tools,
not operational ephemerides or flight certification software.

## Kepler propagation and orbital elements

`astrodynamics/kepler.py` solves the universal Kepler time equation with Stumpff
functions, Newton iteration, and a monotonic bracket fallback. The same routine
handles elliptic, near-parabolic, and hyperbolic energy regimes without selecting an
orbit-specific library. Lagrange coefficients recover the propagated state. Tests
check a circular quarter orbit and use the propagator independently to verify a
Lambert endpoint.

Classical elements are ordered as semi-major axis, eccentricity, inclination,
ascending node, argument of periapsis, and true anomaly. State/element round trips
are verified for a nonsingular inclined ellipse. Circular/equatorial conventions are
defined explicitly; exactly parabolic classical elements are rejected because their
semi-major axis is undefined.

Configured planets now use elliptical Keplerian ephemerides. Setting eccentricity
and argument of periapsis to zero reproduces the original circular baseline.

## Lambert and launch-window design

The zero-revolution universal-variable Lambert solver obtains endpoint velocities
for selected prograde/retrograde and short/long geometry. It evaluates Stumpff
functions directly and brackets the time-of-flight root. A transfer opportunity
reports departure excess velocity, C3, arrival excess speed, and an explicit cost.

The porkchop framework evaluates this solution over independent departure and
arrival arrays. Cells with invalid chronology or no numerical solution remain NaN;
constraints produce a separate Boolean feasibility mask. It does not hide failed
cells or reinterpret a plot as proof.

The extended search also enumerates configured zero- and multi-revolution roots for
both prograde and retrograde geometry. Every accepted branch is independently
propagated to the requested endpoint before deterministic objective ranking. See
[`mission_analysis.md`](mission_analysis.md) for the coverage-aware ephemeris,
finite-burn, visibility, and CCSDS exchange boundaries used around this search.

## Gravity assist and B-plane

Two Lambert legs are joined at a selected fictional assist world. Incoming and
outgoing excess velocities are resolved relative to the moving planet. Their angle
defines an equivalent unpowered hyperbola,

\[
e=\frac{1}{\sin(\delta/2)},\qquad
r_p=\frac{\mu_p}{v_\infty^2}(e-1),\qquad
B=\frac{\mu_p}{v_\infty^2}\sqrt{e^2-1}.
\]

The implementation constructs standard B-plane T/R components, reports periapsis
altitude, and treats unequal incoming/outgoing excess-speed magnitude as powered
flyby mismatch. An unpowered flyby passes only when both speed mismatch and minimum
altitude satisfy declared tolerances. A finite-difference damped least-squares
corrector can refine assist and arrival epochs against altitude and speed residuals.
It is not interception or homing logic; the only “aim point” is a planet-relative
scientific flyby plane.

## Maneuvers and propellant

Impulses accept inertial XYZ or instantaneous radial-transverse-normal components.
They occur at exact propagation boundaries, update velocity, and consume mass using

\[
m_f=m_0\exp\!\left(-\frac{\Delta v}{I_{sp}g_0}\right).
\]

Dry mass is a hard floor. Finite burns define start, duration, thrust, direction,
frame, and specific impulse; acceleration varies as thrust divided by current mass
and mass flow is \(-T/(I_{sp}g_0)\). Invalid propellant budgets fail clearly. The
configured restricted N-body simulator logs burn start/stop and impulse events plus
mass and remaining propellant.

## Optional force and ephemeris fidelity

Opt-in force terms include central-body J2, cannonball solar-radiation pressure, and
the first post-Newtonian Schwarzschild point-mass correction. All are disabled in the
reference case unless configuration explicitly enables them. Hill and Laplace
sphere-of-influence utilities document the different approximations.

`FullNBodyModel` is a separate mutually interacting Newtonian point-mass model. It
does not fix a central body and exposes barycentre, momentum, and total energy.
Two-body tests bound RK4 energy error and verify momentum/periodic state recovery.
The production reference continues using prescribed planet ephemerides because this
is faster and its assumptions are easier to audit.

The original common ephemeris protocol has analytical and tabular implementations.
The stricter coverage-aware interface adds semantic frame/centre/time metadata and a
position/velocity-consistent cubic-Hermite table. Linear table interpolation remains
available for legacy cases with explicit error or endpoint-hold behavior. `SpiceEphemeris`
loads `spiceypy` lazily, checks every user kernel path, converts SPICE kilometres to
SI, and raises `EphemerisUnavailableError` when data/software are absent. Its strict
provider wrapper requires declared coverage and never falls back silently. No kernel
or real mission solution is bundled.

## Uncertainty and limits

The seeded framework disperses injection components, initial mass, planet phase,
gravitational parameters, and maneuver magnitude. It preserves input/run order with
one or several workers, records failed simulations, computes percentiles and 95%
bounds, correlations, and worst-case run indices.

Lambert and patched-conic screening do not replace high-fidelity optimisation.
Operational work would require certified ephemerides, launch-site states, body
orientation, eclipses, finite-thrust optimisation, covariance mapping, navigation
observability, propulsion/thermal/power constraints, and independent review. Those
claims are explicitly outside AeroGNC-Lab.
