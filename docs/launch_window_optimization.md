# Launch-window optimization

## Objective

The launch-window workflow finds a low ideal impulsive cost for a direct transfer
between two fictional synthetic worlds. It is a deterministic engineering example,
not a real launch opportunity. The decision variables are departure epoch
\(t_d\) and arrival epoch \(t_a\); the reported objective is

\[
J(t_d,t_a)=\Delta v_{departure\ parking}+\Delta v_{arrival\ capture}.
\]

A directly implemented zero-revolution Lambert solver supplies transfer velocities.
Planet-relative excess speeds are converted to parking-orbit injection and capture
costs with the vis-viva relation. Feasibility requires positive time of flight,
departure \(C_3\) below its configured limit, and arrival excess speed below its
configured limit. Invalid Lambert cells stay `NaN` rather than receiving a visually
plausible number.

Run the case with:

```powershell
python -m aerognc.cli launch-window --config configs/launch_window_optimization.yaml
```

## Search algorithm

The implementation first evaluates every cell of a rectangular departure/arrival
grid. It selects the best feasible cell, then performs a bounded eight-neighbour
coordinate/pattern search. When no neighbour strictly improves the penalized
objective, both epoch step sizes are halved. Search stops when the largest step is at
or below the declared epoch tolerance or the iteration budget is exhausted. A cache
ensures a repeated epoch pair is evaluated once.

Constraint violations are converted to an explicit speed penalty for navigation of
an infeasible grid, but final acceptance separately requires `feasible=True`.
Therefore a penalized optimum cannot be reported as a feasible mission merely
because its scalar objective is small. This procedure is transparent and
deterministic, but it is a finite global screen followed by local refinement—not a
mathematical proof of the global optimum.

## Numerical verification

For synthetic Asteria to Neria, the 9 by 12 screen plus bounded refinement performs
210 unique evaluations and converges to departure day 28.212890625 and arrival day
266.38671875. The time of flight is 238.173828125 days. Reported departure
\(C_3\) is 8.773617 km2/s2, arrival excess speed is 2.665234 km/s, injection cost is
3.596011 km/s, capture cost is 3.716375 km/s, and total cost is 7.312386 km/s. The
best refined value improves the best feasible grid value, 7.320871 km/s.

Independent universal-variable propagation of the selected Lambert departure state
reaches the analytical destination position with 0.00413 m endpoint error, below the
0.1 m requirement. Tests also assert same-input determinism, bound validation,
feasibility, convergence, and non-worsening refinement. The figure shows the full
grid, hatches infeasible cells, overlays the actual refinement history, and separates
injection from capture cost; it supplements rather than replaces numerical checks.

## Assumptions and next steps

- Epochs are seconds from a synthetic catalog epoch, not UTC launch dates.
- Body states are analytical and contain no covariance or operational ephemeris
  uncertainty.
- Burns are instantaneous and use fixed circular parking-orbit altitudes.
- No launch-site geometry, declination, finite burn, low-thrust arc, eclipse,
  communication, entry, thermal, or planetary-protection constraint is included.
- Multi-revolution Lambert branches are not searched.

A higher-fidelity extension would add public SPICE ephemerides, multiple Lambert
branches, explicit constraint margins, global/metaheuristic comparison, and
finite-burn targeting. Those extensions must retain deterministic regression cases
and must not replace this transparent baseline with an opaque optimizer.
