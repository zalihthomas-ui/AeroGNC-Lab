# Orbit-assisted interplanetary tour

## Purpose and safety scope

This workflow demonstrates preliminary patched-conic mission accounting for a
fictional civilian research spacecraft. Asteria, Neria, Caelus, Helios, every epoch,
and every vehicle value are synthetic. It is not an operational ephemeris or a model
of a real launch vehicle. The route is intentionally an orbit-assisted tour: the
spacecraft reaches Neria, performs an ideal impulsive capture, completes two parking
revolutions, reorients its orbital plane, and performs a departure burn toward
Caelus. That is different from the existing unpowered hyperbolic gravity-assist
example. Capture and departure consume propellant; the model never describes that
energy change as a free flyby.

Run the verified case with:

```powershell
python -m aerognc.cli orbit-tour --config configs/orbit_assisted_tour.yaml
```

The command writes a deterministic SI trajectory, event list, requirement report,
reviewable figure, CCSDS OEM exchange file, and explicitly unexecuted GMAT/SPICE
interfaces.

## Model sequence

Two zero-revolution Lambert arcs join body-centre states at configured synthetic
epochs. Each local patch uses a classical hyperbola from periapsis to the Laplace
sphere of influence,

\[
r_{SOI}=a_p\left(\frac{\mu_p}{\mu_0}\right)^{2/5},\qquad
e_h=1+\frac{r_p v_\infty^2}{\mu_p}.
\]

The hyperbolic periapsis speed and circular parking speed are

\[
v_{p,h}=\sqrt{v_\infty^2+\frac{2\mu_p}{r_p}},\qquad
v_c=\sqrt{\frac{\mu_p}{r_p}}.
\]

Their difference supplies ideal injection or capture magnitude. Hyperbolic anomaly
and mean anomaly provide a finite diagnostic flight time between the sphere of
influence and periapsis; the primary-centred Lambert arc is still patched at the
body-centre epoch, so these diagnostic times are not added to the interplanetary
time of flight.

At Neria the captured orbit period is

\[
T=2\pi\sqrt{\frac{r_p^3}{\mu_p}}.
\]

The configured dwell is exactly two periods. Incoming and outgoing asymptote planes
are not generally coplanar. The demonstrator charges a conservative impulsive plane
change at circular speed,

\[
\Delta v_{plane}=2v_c\sin\left(\frac{\Delta i}{2}\right),
\]

rather than silently assuming cost-free vector alignment. Successive burn masses
use the ideal rocket equation directly,

\[
m_{after}=m_{before}\exp\left(-\frac{\Delta v}{I_{sp}g_0}\right),
\]

and every update enforces the configured dry-mass floor. The reported Oberth energy
diagnostic is the change in planet-relative specific orbital energy caused by the
periapsis departure impulse; it is not a claim of extra propellant-free velocity.

## Verified synthetic result

The configured route departs on day 0, captures at Neria on day 240, dwells for
11,429.22 s, and reaches Caelus on day 2035.4. Five ideal burns total
31.4560 km/s. The largest term is the deliberately conservative 111.663 deg parking
plane alignment (13.6456 km/s). Starting from 130,000 kg at 1,200 s specific impulse,
the final mass is 8,975.65 kg against an 8,000 kg dry mass. The independent universal
propagation check closes the Lambert endpoint within 0.1 m and all configured event,
sphere-of-influence, revolution, delta-v, and mass assertions pass.

These values are useful software-verification evidence, not a practical mission
proposal. In particular, the very large plane-change cost demonstrates why a real
preliminary design would optimize encounter geometry, exploit a moon or flyby,
consider low thrust, or reject the route.

## Limitations

- Planets follow prescribed synthetic circular/inclined analytical ephemerides.
- Lambert arcs are two-body and burns are instantaneous; no finite-burn steering,
  navigation error, launch, atmosphere, entry, landing, communications, or thermal
  constraint is included.
- Parking-orbit capture and alignment are scalar preliminary budgets, not a
  high-fidelity arrival sequence.
- The sphere-of-influence model is a patched-conic approximation, not simultaneous
  multi-body propagation.
- No result is transferable to a real vehicle or planetary mission without new
  requirements, public operational ephemerides, covariance analysis, and independent
  verification.

Implementation is in `astrodynamics/patched_conics.py` and
`simulation/orbit_assisted_tour.py`; tests cover hyperbolic identities, deterministic
configuration, ordered events, Lambert closure, revolution count, and mass bounds.
