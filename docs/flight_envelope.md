# Flight-Envelope Trim and Scheduled Control

This workflow connects the synthetic aerodynamic database, atmosphere, mass/CG/
inertia schedule, actuator limits, nonlinear trim, central-difference linearisation,
modal analysis, controllability/observability, manually implemented continuous LQR,
gain interpolation, and seeded robustness checks.

## Reduced pitch model

The local analysis state and control are

\[
\mathbf x=[\alpha,q]^T,\qquad u=\delta,
\]

where \(\alpha\) is angle of attack, \(q\) pitch rate, and \(\delta\) a fictional
pitch control command. At each Mach-altitude-mass point, atmosphere supplies speed
of sound and density, while the vehicle schedule supplies CG and pitch inertia.
With airspeed (V), reference area (S), length (c), and dynamic pressure
\(\bar q\), the nonlinear surrogate is

\[
\dot\alpha=q+\frac{\bar q S}{mV}
\left(C_N(M,\alpha)+C_{N_\delta}\delta\right),
\]

\[
\dot q=\frac{\bar qSc
\left(C_m(M,\alpha)+C_{m_q}\frac{qc}{2V}
+C_{m_\delta}\delta\right)+M_d}{I_y}.
\]

The database supplies (C_N) and (C_m); the clearly labelled synthetic control
and pitch-rate derivatives are configured separately. This is a control-design
surrogate, not a replacement for the nonlinear quaternion 6-DOF plant.

## Trim, linear model, and design

Bounded damped Gauss-Newton iteration solves

\[
\dot\alpha(\alpha^*,0,\delta^*)=0,\qquad
\dot q(\alpha^*,0,\delta^*)=0.
\]

Central differences form

\[
A=\left.\frac{\partial f}{\partial x}\right|_*,\qquad
B=\left.\frac{\partial f}{\partial u}\right|_*.
\]

Every point reports open/closed-loop eigenvalues, mode damping, controllability and
observability ranks, Riccati residual, trim command, unused actuator-position
fraction, and remaining control moment. The continuous LQR gain is computed from the
stable invariant subspace of the Hamiltonian matrix, independently checked against
SciPy in the generic control tests.

## Three-dimensional gain schedule

One gain is designed at every configured Mach-altitude-mass tensor point. Each gain
component is then interpolated trilinearly. Cell-centre verification deliberately
checks points not used for design. The online `ScheduledStateFeedback` interface
accepts state error and the three schedule variables and optionally enforces a
command limit.

Seeded robustness trials draw aerodynamic derivative, control-effectiveness, and
inertia multipliers. The nominal interpolated gain is applied to each uncertain
local model; stable fraction, minimum damping, and worst real pole are reported.
These trials are repeatable robustness screening, not a proof over continuous
uncertainty sets.

## Configured evidence

Run:

```bash
python -m aerognc.cli flight-envelope --config configs/flight_envelope.yaml
```

The current synthetic grid contains 36 design points (4 Mach x 3 altitude x 3 mass),
12 between-grid cell centres, and 120 seeded uncertain samples. All trims converge;
all local models have full rank; every design, interpolated, and uncertain closed
loop is stable. Minimum unused actuator range is about 99.23%, exceeding the 70%
requirement. The JSON report stores all (A), (B), gain, pole, trim, authority,
and requirement records; the CSV is compact for external review.

## Limitations

- Only a two-state longitudinal channel is scheduled.
- Configured control derivatives are synthetic and are not present in the sparse
  demonstration CSV.
- The uncertainty distribution is a screening assumption, not measured scatter.
- Actuator dynamics, nonlinear command saturation, flexible modes, and sensor delay
  require time-domain 6-DOF verification before real controller use.
