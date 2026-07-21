# Constrained Research-Rocket Ascent Guidance

This workflow demonstrates safe, non-operational trajectory shaping for a fictional
civilian sounding rocket. It accepts no destination vehicle, interception state, or
homing measurement. A scalar desired apogee is a flight-performance requirement.

## Offline/online separation

The offline layer uses a bounded deterministic coordinate search over two readable
reference parameters:

1. terminal pitch-elevation offset;
2. thrust-scale multiplier.

Every candidate runs the same event-aware plant. The objective penalizes normalized
desired-apogee error plus squared violations of max-Q, proper load, and loaded
angle-of-attack limits. The full evaluation history is saved, so the selected result
is auditable and repeatable.

The online layer begins with piecewise-linear pitch-elevation and throttle schedules,
then applies independent governors:

\[
|\alpha_c|\le\alpha_{max},
\]

\[
s_q=\operatorname{clip}\left(
\frac{q_{max}-q}{q_{max}-q_{soft}},0,1\right),
\]

\[
s_n=\operatorname{clip}\left(
\frac{n_{max}g_0m-\|F_A\|}{T_{nom}},0,1\right).
\]

The throttle command is the minimum of its reference and active governor scales. A
documented ballistic-apogee reserve prevents the simple drag-free predictor from
cutting thrust at the desired final altitude itself.

## Pitch-plane variable-mass plant

The state is

\[
\mathbf x=[r_N,r_D,v_N,v_D,m_p,\theta]^T,
\]

where \(m_p\) is remaining propellant and \(\theta\) is actual thrust-axis elevation.
Thrust and propellant flow use the same throttle:

\[
T=s_T T_{nom}(t),\qquad
\dot m_p=-s_T\dot m_{p,nom}(t),\qquad m=m_{dry}+m_p.
\]

This preserves a monotonic mass history and a hard dry-mass floor. Pitch response is
a rate-limited first-order surrogate:

\[
\dot\theta=\operatorname{sat}_{\dot\theta_{max}}
\left((\theta_c-\theta)/\tau_\theta\right).
\]

Drag acts opposite air-relative velocity. The table normal coefficient generates a
pitch-plane force perpendicular to that velocity. Gravity and wind use the same
configured environment as the baseline ascent.

## Requirement domain

Max-Q and proper load are assessed from launch through the nominal motor-window end.
Angle of attack is undefined/poorly conditioned when air-relative speed and dynamic
pressure approach zero, especially in launch-site wind. Its requirement is therefore
assessed only during powered ascent at (q\ge100\) Pa. Full coast/descent values are
still logged; the simplified model does not claim post-apogee recovery-attitude
control.

## Configured result

Run:

```bash
python -m aerognc.cli constrained-ascent \
  --config configs/constrained_ascent_guidance.yaml
```

At a 0.08 s RK4 step, the ungoverned reference reaches about 1116.4 m but exceeds
the declared max-Q and proper-load limits. The optimized/governed case reaches about
785.3 m against an (800\pm18) m requirement; powered-ascent maximum dynamic
pressure is about 8.84 kPa, proper load 6.00 (g_0), and loaded absolute angle of
attack 4.10 deg. All declared checks pass. The result retains roughly 2.14 kg of
unused propellant because throttling is coupled to mass flow and the configured
motor window does not extend to consume it.

## Limitations

- The direct search has two variables and is not a general optimal-control method.
- Instantaneous constraints use a reduced pitch-plane model, not the full 6-DOF
  closed-loop controller/actuator stack.
- The thrust curve is time-windowed; throttled unused propellant is retained rather
  than extending the burn.
- Rail contact, recovery dynamics, structural flexibility, combustion transients,
  and real flight constraints are outside scope.
