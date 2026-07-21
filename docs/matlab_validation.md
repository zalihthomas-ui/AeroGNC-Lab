# MATLAB Cross-Validation

## Execution status

The optional constant-force translation and two-body orbital cases were re-executed
locally for release 0.7.0 on 2026-07-20 with MATLAB R2024a Update 3. MATLAB is not a
dependency of AeroGNC-Lab. The Python
package, CLI, tests, and reference workflows operate without it.

Both implementations read `matlab_validation/constant_acceleration_case.json` and
independently implement fixed-step classical RK4. For NED position \(\mathbf p\) and
velocity \(\mathbf v\),

\[
\dot{\mathbf p}=\mathbf v,\qquad
\dot{\mathbf v}=\frac{\mathbf F}{m}+\mathbf g.
\]

The eight-second case uses a 0.02 s step, 15 kg constant mass, nonzero initial NED
position/velocity, a three-axis constant force, and constant NED gravity. The exact
constant-acceleration solution is evaluated independently by both tools.

## Reproduction

From the repository root:

```matlab
addpath('matlab_validation');
validate_constant_acceleration;
```

Then compare the emitted CSV with Python:

```bash
python scripts/compare_matlab_validation.py --require-matlab
```

Transient CSV/JSON records are written below `matlab_validation/output/` and ignored
by Git. The shared case and both implementations are versioned.

## Executed results

| Comparison | Maximum absolute component error | Tolerance | Result |
|---|---:|---:|---|
| Python RK4 vs exact solution | \(3.2401\times10^{-12}\) | \(1.0\times10^{-10}\) | Pass |
| MATLAB CSV vs exact solution | \(1.0800\times10^{-12}\) | \(1.0\times10^{-10}\) | Pass |
| Python RK4 vs MATLAB RK4 | \(5.1159\times10^{-13}\) | \(1.0\times10^{-10}\) | Pass |
| Python universal Kepler vs MATLAB `ode113`, position norm | \(7.79\times10^{-8}\) m | 0.1 m | Pass |
| Python universal Kepler vs MATLAB `ode113`, velocity norm | \(1.11\times10^{-10}\) m/s | \(10^{-4}\) m/s | Pass |

MATLAB's in-memory pre-export check reported \(1.4211\times10^{-12}\); the small
difference from the CSV-based value is due to recomputing the exact trajectory from
decimal time values after export. Both are over an order of magnitude inside the
declared tolerance. No MATLAB result is embedded in Python tests, so normal test
execution does not silently require proprietary software.

The orbital case is intentionally more independent than a line-by-line port.
`matlab_validation/validate_two_body_case.m` integrates Newton's two-body ODE with
adaptive MATLAB `ode113`; Python uses its directly implemented universal-variable
Kepler equation. Both read `two_body_case.json`. Reproduce it with:

```matlab
addpath('matlab_validation');
validate_two_body_case;
```

The executed JSON record stores MATLAB R2024a, final state, errors, tolerances and
pass status. It is versioned because it was actually generated in this environment.

## Limits

Together these benchmarks independently check state ordering, NED signs,
force/mass/gravity composition, time grid, RK4, central gravity, and conic
propagation. They are not validation of the complete nonlinear aerodynamic vehicle
or a real planetary ephemeris. Broader cases must record actual external-tool
version, command, tolerance, and result.
