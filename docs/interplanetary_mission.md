# Interplanetary Gravity-Assist Mission

## Purpose and safety scope

This workflow demonstrates civilian interplanetary trajectory propagation, planetary
ephemerides, encounter analysis, and gravity-assist energy exchange. The Helios
system, Asteria, Brontes, Caelus, the Selene Pathfinder spacecraft, their phases, and
the mission are fictional and synthetic. They are not an operational launch window,
real ephemeris, or design for a specific vehicle.

An important physical distinction is made in both the code and UI: an unpowered
gravity assist does not enter a temporary bound orbit. The spacecraft follows a
planet-relative hyperbola. Its incoming and outgoing excess-speed magnitudes remain
approximately equal, but the moving planet rotates that velocity vector. When the
planet-relative velocity is transformed back to the primary-centred frame, the
spacecraft can gain or lose heliocentric energy and speed. Capturing into orbit and
departing again would require propulsion and is not claimed by this demonstration.

## Frame, state, and units

The integration frame is a nonrotating, primary-centred ecliptic Cartesian frame.
The state is

\[
\mathbf x = [x, y, z, v_x, v_y, v_z]^T,
\]

with position in metres, velocity in metres per second, time in seconds, and
gravitational parameters in cubic metres per square second. The UI explicitly
converts distance to astronomical units, gigametres, or kilometres and time to days;
the numerical core remains SI.

Each planet follows a configured analytical elliptical Keplerian ephemeris. Mean
anomaly advances with \(n=\sqrt{\mu_0/a^3}\), the elliptic Kepler equation is solved,
and classical elements rotate the perifocal state into the inertial frame. Setting
eccentricity to zero reduces the position to

\[
\mathbf r_i(t)=a_i[\cos\theta_i,\sin\theta_i,0]^T,
\qquad
\theta_i=\theta_{i0}+\sqrt{\frac{\mu_0}{a_i^3}}t.
\]

Inclination and ascending-node rotations map this state into the ecliptic frame.
The restricted N-body regression remains circular and coplanar so its gravity-assist
geometry remains readily inspectable; the separate Mission Designer catalog uses
inclined elliptical synthetic orbits.

## Restricted N-body equations

The spacecraft is massless with respect to the celestial bodies. The primary is
fixed at the frame origin, and planet states are prescribed rather than mutually
propagated. The acceleration is

\[
\ddot{\mathbf r}=
-\mu_0\frac{\mathbf r}{\|\mathbf r\|^3}
+\sum_i\mu_i\left(
\frac{\mathbf r_i-\mathbf r}{\|\mathbf r_i-\mathbf r\|^3}
-\frac{\mathbf r_i}{\|\mathbf r_i\|^3}
\right).
\]

The second term inside the parentheses is the indirect acceleration required because
the primary-centred frame is not exactly inertial when planets accelerate the
primary. Omitting it would introduce a systematic frame inconsistency.

The initial state is configured relative to the departure body in its radial,
transverse, normal (RTN) frame. The baseline begins just outside the fictional
departure sphere-of-influence boundary after injection; atmospheric launch and the
finite-duration injection burn are deliberately outside this model.

Classical RK4 performs the propagation. The configured step is a maximum of six
hours, while a deterministic encounter-timescale rule reduces the step to as little
as 30 seconds near a body. Accepted states are checked for primary or planetary
collision. Entry, closest approach, exit, destination-arrival, and mission-end events
are logged without relying on plots.

## Transfer and flyby design calculations

The package directly implements coplanar Hohmann transfer time and velocity changes
from vis-viva. It also evaluates an ideal planet-centred hyperbolic flyby from
periapsis radius \(r_p\) and excess speed \(v_\infty\):

\[
e=1+\frac{r_pv_\infty^2}{\mu_p},
\qquad
\delta=2\sin^{-1}\!\left(\frac{1}{e}\right),
\qquad
v_p=\sqrt{v_\infty^2+\frac{2\mu_p}{r_p}}.
\]

These calculations provide transparent first estimates. The configured example is
then propagated with simultaneous primary and planetary accelerations; it is not a
sequence of disconnected conic drawings.

The expanded design layer also directly implements universal Kepler propagation,
zero-revolution Lambert endpoints, launch-window grids, planet-relative B-plane
coordinates, equivalent flyby altitude, powered mismatch, and finite-difference
multi-leg epoch correction. Exact-time impulses and finite burns use inertial or RTN
inputs and ideal propellant accounting. See
[advanced astrodynamics](advanced_astrodynamics.md) for equations and validation.

## Reproducible example

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli interplanetary `
  --config configs\interplanetary_gravity_assist.yaml
```

The solver writes `interplanetary_trajectory.csv` and
`interplanetary_summary.json`, then opens mission control. Use `--output PATH` to
change the result directory or `--no-window` for a calculation-only run.

The synthetic regression scenario produces the following approximate outcomes:

- Brontes encounter-boundary entry near day 1330.9;
- closest approach near day 1335.8 at 371,000 km from Brontes' centre;
- about +10.1 km/s heliocentric speed change across the 5 Gm encounter boundary;
- planet-relative boundary-speed difference below 1 m/s, demonstrating the
  unpowered energy-exchange interpretation; and
- entry into the Caelus 8 Gm arrival corridor near day 1843.5.

The arrival event means corridor entry, not propulsive capture or landing.

For a guided route form instead of YAML, double-click `run_aerognc.bat` or run
`python -m aerognc.cli mission-designer`. The designer clearly labels its fast
Lambert/patched-conic preview separately from this restricted N-body reference.

## Mission-control UI

The dark 3D dashboard displays live planet ephemerides, complete and elapsed
spacecraft paths, velocity direction, mission phase, SI-derived telemetry, event
status, closest-approach outcome, and the speed/specific-energy jump at the assist.

| Control | Action |
|---|---|
| Space or Play/Pause | Pause or resume playback |
| Timeline slider | Seek through the complete mission |
| Left / Right | Seek by ten mission days |
| Up / Down | Double or halve playback days per real second |
| `N` or Next Event | Jump to the next logged mission event |
| `C` or Camera | Cycle system, spacecraft, assist, destination, top, and free views |
| Mouse drag in Free mode | Rotate the 3D camera |

Headless export is available with `--save-snapshot`, or with `--save-gif` and
`--no-window`. Transient CSV and GIF products are ignored by default because they can
be large.

## Verification and limitations

Tests cover circular-orbit geometry, Keplerian period and speed, the direct and
indirect acceleration terms, Hohmann equations, hyperbolic-turn equations,
configuration rejection, deterministic repeat propagation, collision clearance,
ordered encounter events, speed/energy gain, planet-relative speed consistency,
arrival, UI controls, immutable playback, PNG generation, and GIF generation.

This is a medium-fidelity educational reference. It does not use JPL or other
operational ephemerides, and its prescribed planets do not mutually perturb one
another. Separate verified modules implement full Newtonian N-body dynamics, J2,
solar radiation pressure, relativity, burns, seeded uncertainty, launch-window
screening, and Lambert/B-plane design, but these are opt-in analyses rather than
hidden changes to the reference. Low-thrust global optimisation, operational orbit
capture, atmospheric entry, and landing remain outside scope. A real
mission requires high-fidelity ephemerides, optimisation, covariance analysis,
navigation design, propulsion constraints, and independent flight-dynamics review.
