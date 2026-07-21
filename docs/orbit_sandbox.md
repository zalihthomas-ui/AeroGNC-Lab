# Satellite Orbit Sandbox

## Purpose and users

The Satellite Orbit Sandbox is a near-planet teaching and verification workflow for
students, flight-dynamics engineers, educators, and reviewers who need to see how a
model choice changes a propagated path. It can answer questions such as:

- What is the difference between force-free motion and an orbit?
- Does a circular two-body orbit close numerically?
- How does adding a moon change a massless satellite path?
- What changes when all configured bodies have finite mass and interact?
- Under one explicit atmospheric-density assumption, does a low satellite cross a
  chosen reentry threshold during a finite number of days?
- How do satellite mass, drag area, $C_D$, density scale, or ideal correction burns
  change that finite-horizon result?

The primary `Orbis-A`, moons `Luma` and `Vesper`, and satellite `Meridian-1` are
fictional. Their parameters are synthetic and do not reproduce an operational vehicle
or mission.

## Terminology shown in the UI

The UI deliberately avoids calling the one-moving-body case an orbit.

| UI model | Bodies whose motion is represented | Applied physics |
|---|---:|---|
| No force / one moving body | satellite | no force; straight-line control case |
| Two body | primary + satellite | spherical central gravity |
| Restricted three body | primary + prescribed moon + satellite | satellite is massless; primary/moon paths are prescribed |
| Full N-body | primary + every moon + finite-mass satellite | every body accelerates every other body |
| Perturbed decay | primary + satellite | central gravity, $J_2$, rotating atmosphere and drag |

In conventional astrodynamics, the "two" in two-body motion counts the primary and the
satellite. A lone body with no force preserves its velocity and does not orbit.

## Frames, state and units

The propagated relative satellite state is

\[
\mathbf{x}=
\begin{bmatrix}
\mathbf{r}_{P S}^{P} & \mathbf{v}_{P S}^{P}
\end{bmatrix}^{\mathsf T},
\]

where (P) is a planet-centred inertial frame whose (+z) axis is the primary spin
axis. Position is in metres and inertial velocity is in metres per second. For the
full N-body model, the internal state concatenates the six-state inertial vector of
the primary, each configured moon, and the satellite. Logged satellite values are
then formed relative to the propagated primary.

Initial inclination, ascending-node angle and phase use radians internally and degrees
at the YAML/UI boundary. Circular and escape speeds are calculated in SI units. A
custom speed is tangential to the initial circular-orbit direction.

## Governing equations

### Force-free control

\[
\dot{\mathbf r}=\mathbf v,
\qquad
\dot{\mathbf v}=\mathbf 0.
\]

The analytical solution is

\[
\mathbf r(t)=\mathbf r_0+t\mathbf v_0,
\qquad
\mathbf v(t)=\mathbf v_0.
\]

This case tests integration and display plumbing without attributing curvature to a
force that is not present.

### Two-body relative motion

For primary gravitational parameter $\mu_P$,

\[
\ddot{\mathbf r}=-\mu_P\frac{\mathbf r}{\lVert\mathbf r\rVert^3}.
\]

At radius (r), the UI speed rules use

\[
v_{\mathrm{circular}}=\sqrt{\frac{\mu_P}{r}},
\qquad
v_{\mathrm{escape}}=\sqrt{\frac{2\mu_P}{r}}.
\]

### Restricted three-body motion

The restricted model evaluates the differential acceleration of the massless
satellite caused by the primary and the first configured moon. The moon follows its
configured circular ephemeris. The satellite cannot perturb either massive body.
This is useful for studying the difference between a prescribed background
ephemeris and a mutually interacting model.

### Full N-body motion

For each finite-mass body (i),

\[
\ddot{\mathbf r}_i = G\sum_{j\ne i}m_j
\frac{\mathbf r_j-\mathbf r_i}
{\lVert\mathbf r_j-\mathbf r_i\rVert^3}.
\]

The initial positions and velocities are shifted to their common barycentre before
propagation. No body is fixed. Close approaches and collisions are outside the
prepared scenario domain; the solver rejects a zero separation instead of hiding a
singularity.

### Perturbed low orbit

The decay model combines central gravity, primary oblateness and drag:

\[
\ddot{\mathbf r}=\mathbf a_{2B}+\mathbf a_{J_2}+\mathbf a_D.
\]

The $J_2$ acceleration is evaluated directly from planet-centred coordinates. The
atmosphere rotates with

\[
\boldsymbol\omega_P=\begin{bmatrix}0&0&\omega_P\end{bmatrix}^{\mathsf T},
\qquad
\mathbf v_{\mathrm{rel}}=\mathbf v-\boldsymbol\omega_P\times\mathbf r,
\]

and drag is

\[
\mathbf a_D=-\frac{1}{2}\rho C_D\frac{A}{m}
\lVert\mathbf v_{\mathrm{rel}}\rVert\mathbf v_{\mathrm{rel}}.
\]

Consequently, mass, area, $C_D$, density, altitude and atmosphere-relative speed all
enter the path calculation. They are not display-only fields.

## Reference orbital atmosphere

Below 47 km the implementation delegates to the existing 1976 standard-atmosphere
model. Above 47 km, density is log-linearly interpolated through a transparent
Earth-like reference table from 47 to 1000 km. A continuous exponential tail is
provided to 1500 km, after which density is set to zero. The entire profile is
multiplied by the configured density scale.

This deterministic table makes sensitivity studies reproducible. It is not a
thermosphere forecast and contains no solar-flux, geomagnetic-index, local-time,
latitude, composition, or storm model. A predicted lifetime is therefore conditional
on this selected reference profile.

## Events and "how many days" interpretation

The model detects two terminal boundaries:

- descending altitude crossing the configured reentry threshold; and
- outward radius crossing the configured escape-radius multiplier.

When reentry occurs, the reported time is a linearly located threshold crossing. It
is not impact time and does not model heating, breakup, lift, attitude, ablation, or
debris. If neither boundary occurs, the report says that modeled survival is
**greater than the simulated duration**. It never converts a finite run into a claim
of infinite lifetime.

Osculating eccentricity, perigee and apogee are calculated relative to the primary at
each logged state. They are diagnostics even when perturbations make the orbit
non-Keplerian. Revolutions are accumulated from successive position-vector angular
increments.

## Optional correction impulses

Corrections are disabled in the prepared case. When enabled, a perigee below the
trigger altitude can request an ideal tangential impulse that restores circular speed.
Propellant is accounted with

\[
m_{\mathrm{after}}=m_{\mathrm{before}}
\exp\left(-\frac{\Delta v}{I_{sp}g_0}\right).
\]

The burn is rejected if it would cross dry mass, and a configured integer caps the
number of burns. These are instantaneous educational impulses: finite burn duration,
attitude slew, navigation error, maneuver execution error and propulsion transients
are omitted.

## Running it

Double-click `run_solver.bat`, select **Satellite Orbit**, leave the prepared inputs
unchanged, and choose **Calculate + Play Orbit in 3D**. In the player, drag the time
slider to seek, press Space to pause, use `+`/`-` to change playback speed, and press
`C` to switch between satellite-scale and whole-system views.

The equivalent reproducible CLI is:

```bash
python -m aerognc.cli orbit-sandbox --config configs/orbit_sandbox.yaml --play
```

Use `--no-plots` for a data-only run. The output includes a trajectory CSV, standard
summary JSON, limitations report, 3D PNG, and decay-diagnostic PNG.

## Verification evidence

Automated evidence includes:

- exact force-free analytical translation;
- one-period circular two-body closure and revolution count;
- finite deterministic restricted-three-body and full-N-body propagation;
- lower specific orbital energy when the drag-density scale is increased;
- monotonic/reference density tests;
- strict configuration and public-safety metadata rejection;
- finite-horizon wording; and
- CLI result generation.

Plots support interpretation but are not used as the numerical pass criterion.
