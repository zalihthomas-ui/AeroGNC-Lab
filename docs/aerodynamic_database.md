# Multidimensional Aerodynamic Database

The aerodynamic interface accepts a directly implemented regular-grid database so
future synthetic wind-tunnel or CFD-derived tables can replace the baseline
coefficients without changing flight dynamics. The supplied CSV is deliberately
small, fictional, and synthetic.

## Long-form CSV contract

Each row represents one complete tensor-grid point. Columns not named as
coefficients are interpolation axes; the six required coefficient columns are:

```text
drag, side, normal, roll, pitch, yaw
```

Supported axes are `mach`, `alpha_rad`, `beta_rad`, `p_hat`, `q_hat`, `r_hat`,
`roll_control`, `pitch_control`, `yaw_control`, and `reynolds`. A file may use any
nonempty subset, but every tensor-product combination must appear exactly once.
Duplicate points, missing combinations, nonnumeric/nonfinite values, unknown axes,
and incomplete coefficient sets fail during loading.

The fictional example uses Mach, angle of attack, and sideslip:

```yaml
aerodynamics:
  database_file: aero_database_synthetic.csv
  out_of_range: clamp
```

File bytes are hashed with SHA-256 and carried into the analysis report so a result
can be tied to the exact source table.

## Interpolation and derivatives

For (d) axes, the implementation locates the containing cell and evaluates the
multilinear interpolant

\[
f(\mathbf x)=\sum_{\mathbf c\in\{0,1\}^d}
f_{\mathbf c}\prod_{j=1}^{d}
\left[c_j t_j+(1-c_j)(1-t_j)\right],
\]

where (t_j) is the normalized coordinate on axis (j). The analytical in-cell
partial derivative is evaluated by differentiating this expression, not by calling
a second external interpolation library. The coefficient Jacobian therefore has
rows ordered as the six coefficients and columns ordered exactly as the CSV axes.

Each table declares one out-of-range policy:

- `error`: reject the query;
- `clamp`: use the nearest boundary coordinate;
- `extrapolate`: extend the boundary cell linearly.

Diagnostics list every axis outside its domain before a coefficient is used.

## Common provider interface

`AerodynamicModel` accepts either the original transparent Mach/drag model or an
`AerodynamicCoefficientProvider`. Both expose the same six coefficients to 3-DOF,
6-DOF, envelope, and Monte Carlo consumers. Monte Carlo drag dispersion wraps the
provider and scales drag without discarding the other tabulated coefficients.

Run the database audit with:

```bash
python -m aerognc.cli aero-analysis \
  --config configs/three_dof_aero_database.yaml
```

It writes axis bounds, provenance hash, the nominal coefficient vector, local
Jacobian, and coefficient/derivative plots. The configured database also runs
through the normal `run` CLI, proving it is part of the flight plant rather than an
isolated plotting example.

## Verification and limitations

Tests cover exact affine fields, gradients, all three boundary policies, malformed
CSV grids, hash provenance, legacy-provider compatibility, and a configured flight.
The supplied grid has only 27 rows and is not aerodynamic validation. Multilinear
interpolation is continuous in value but generally discontinuous in gradient at cell
boundaries; no smoothing, uncertainty model, hysteresis, aeroelasticity, or
high-angle separated-flow model is implied.
