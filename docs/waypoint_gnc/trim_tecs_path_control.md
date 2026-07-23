# Trim, Total-Energy Control, and Continuous Path Geometry

> **Safety scope.** This capability drives only the two bundled internal simulators.
> It has no real-vehicle command path, is not flight-certified, and is not evidence
> that the fictional Sparrow-X2 model represents a physical aircraft.

This design closes the waypoint controller's longitudinal-initialization and path-
continuity gaps. The reproducible entry point is
`configs/waypoint_gnc_tecs.yaml`; the independent acceptance record is
`results/reference/waypoint_control_campaign.json`.

## Straight-flight trim

The reduced plant has an analytic throttle equilibrium. The coefficient-driven
18-state plant solves the bounded decision vector

\[
z = [\alpha,\;\delta_{e,up},\;\delta_t]^T
\]

until local forward acceleration, local down acceleration, and pitch angular
acceleration are simultaneously zero. The existing damped Gauss-Newton
`solve_trim` implementation supplies bounds, convergence tolerance, and iteration
limits. A converged solution initializes angle of attack, pitch, physical elevator,
throttle, and autopilot feedforward consistently, so the actuator states do not jump
on the first integration step.

Failure behavior is explicit:

- `reject` raises `TrimConvergenceError` before propagation;
- `fallback_configured` applies the declared static commands and marks the result as
  unconverged fallback evidence.

The acceptance configuration uses strict rejection. Its coefficient trim converges
in three iterations with a residual infinity norm of
`4.20e-10`: angle of attack/pitch `2.693 deg`, elevator-up `3.288 deg`, and throttle
`0.2984`. The reduced equilibrium is analytic with zero residual.

## Total-energy controller

The selectable `total_energy` mode coordinates altitude and airspeed instead of
running two unrelated outer loops. With specific potential and kinetic energy

\[
E_p = gh, \qquad E_k = \frac{V_a^2}{2},
\]

the throttle loop controls the energy-sum error

\[
e_T = (E_{p,c}-E_p) + (E_{k,c}-E_k),
\]

while pitch controls the energy-balance error

\[
e_B = (E_{p,c}-E_p) - (E_{k,c}-E_k).
\]

Each channel is trim feedforward plus bounded PI feedback. Altitude and airspeed
references have deterministic slew limits; climb rate provides throttle and flight-
path-angle feedforward; the shared PID implementation clamps integrators and tracks
the realizable output when feedforward causes clipping. `activate()` preloads both
integrators against the current commands for bumpless mode entry. The legacy
`altitude_airspeed` mode remains the default for backward compatibility.

## Fillets and loiter tangencies

At a feasible fly-through corner, the path manager replaces the instantaneous line-
to-line change with a finite circular `FilletSegment`. The requested coordinated-
turn radius is

\[
R = \frac{V_a^2}{g\tan |\phi_f|},
\]

then bounded by the configured maximum and by available length on both adjacent
legs. Degenerate, reversing, or undersized corners fall back safely to the existing
switching behavior. Entry and exit occur on tangent half-planes, with altitude and
airspeed interpolated continuously along the arc.

For line-orbit-line sequences, approach and departure endpoints are moved to
direction-consistent circle tangencies. When loiter time expires, switching waits
for the aircraft to reach the departure region; it does not release at an arbitrary
point on the circle. Optional course and roll-feedforward slew limits bound the
remaining discrete-time command change.

## Envelope telemetry

Every mission sample records controller-facing—not hidden truth—margins for:

- stall speed and airspeed margin;
- bank-derived load factor plus bank and pitch limits;
- individual and minimum surface authority plus throttle authority; and
- lower and upper specific-energy boundaries.

The coefficient stall reference uses configured `CL_max`, conservative initial mass,
ISA density at estimated altitude, and estimated bank load. The reduced model uses
its declared minimum airspeed. These are runtime engineering diagnostics, not a
certified flight-envelope protection system.

## Reproduce the evidence

```bash
python -m aerognc.cli waypoint --config configs/waypoint_gnc_tecs.yaml
python scripts/verify_waypoint_control.py
```

The second command runs the same mission at 20 m/s in a steady 1 m/s crosswind on
both internal plants. The committed campaign currently reports:

| Metric | Coefficient plant | Reduced plant | Acceptance |
|---|---:|---:|---:|
| Completion time | 170.80 s | 178.45 s | <=180 / <=185 s |
| Maximum cross-track | 10.673 m | 20.831 m | <=15 / <=25 m |
| Maximum course step | 3.000 deg | 3.000 deg | <=3.01 deg |
| Minimum stall margin | 8.224 m/s | 8.000 m/s | >=7.5 m/s |
| Maximum load factor | 1.180 | 1.221 | <=1.3 |
| Minimum surface margin | 71.66% | 70.14% | >=65% |
| Maximum total-energy error | 111.60 m2/s2 | 50.53 m2/s2 | <=150 / <=75 m2/s2 |

Both missions exercise line, fillet, and orbit segments with tangent loiter
transitions, bumpless initialization, positive upper/lower energy margins, zero
actuator-saturation samples, and zero safety events. Terminal 3D separation is
`1.346 m`; the duration ratio is `1.045`. Passing is regression evidence within this
declared synthetic scenario, not physical-aircraft validation.
