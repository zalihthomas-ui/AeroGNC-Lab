# Flight-Control Engineering Analysis

## Reproducible workflow

The configured pitch-channel analysis is run with:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli flight-analysis `
  --config configs\flight_control_analysis.yaml
```

It writes `flight_control_analysis.json` and a four-panel engineering figure. The
plant, weights, gain-schedule points, frequency range, SIL deadline, sample count,
and random seed are synthetic and readable in YAML.

## Trim and linearisation

`solve_trim` uses a central-difference Jacobian, damped Gauss-Newton correction,
bounded decision variables, and backtracking line search. Convergence is based on an
explicit infinity-norm residual, not visual steadiness. The example balances a
constant fictional disturbance moment.

For nonlinear \(\dot x=f(x,u)\), the perturbation matrices are calculated directly:

\[
A_{ij}=\left.\frac{\partial f_i}{\partial x_j}\right|_{x_0,u_0},\qquad
B_{ij}=\left.\frac{\partial f_i}{\partial u_j}\right|_{x_0,u_0}.
\]

Central differences scale each perturbation relative to its operating-point value.
The result stores \(f(x_0,u_0)\) so a caller can verify that a claimed trim is
actually steady. Controllability and observability matrices and ranks are exposed.

## Modes and LQR

Every eigenvalue is reported with natural frequency, damping ratio, time constant,
and stability flag. The continuous LQR implementation does not call a control
toolbox. It constructs the Hamiltonian matrix, selects its stable invariant
subspace, recovers the symmetric Riccati solution, and computes

\[
K=R^{-1}B^TP,\qquad u=-Kx.
\]

The algebraic Riccati residual is part of the result. An automated test compares
both \(P\) and \(K\) against SciPy’s independent `solve_continuous_are` solution to
tight numerical tolerance. Multiple configured inertia points produce independent
LQR rows that are suitable for the existing clamped linear gain interpolator.

## Frequency response and margins

The state-space frequency response is evaluated explicitly as

\[
G(j\omega)=C(j\omega I-A)^{-1}B+D.
\]

Unwrapped SISO phase and magnitude crossings give gain crossover, phase crossover,
phase margin, and gain margin. A missing crossover is reported as unbounded in memory
and `null` in JSON, never as a large invented finite value. The synthetic reference
has a phase margin of about 66.9 degrees and no finite -180-degree crossover over the
defined model.

## Identification and software timing

The identification utility estimates continuous \(A,B\) from time, state, and input
histories using nonuniform numerical derivatives and least squares. It reports
per-state derivative RMS residual and regressor condition number. Its validation
case reconstructs a known oscillator.

SIL timing repeatedly calls the actual Python controller on a seeded input array and
records mean, 95th percentile, maximum, deadline misses, and an output checksum. The
configured local reference completes 10,000 calls with no 1 ms deadline miss. This
is operating-system/application timing, not deterministic embedded execution and not
hardware-in-the-loop evidence. Hardware selection remains deferred until measured
I/O and scheduling requirements exist.

All analysis is restricted to civilian research-rocket attitude stabilisation. It
contains no target state, terminal guidance, pursuit, or engagement logic.
