# Rendezvous & Proximity Operations (RPO)

> **Scope / public-safety.** This is cooperative satellite **rendezvous,
> inspection, and station-keeping** mathematics — the same used for docking,
> servicing, and debris-avoidance. It navigates a chaser to *approach and hold
> near* a target and reports how a burn reshapes an orbit. It contains **no**
> interception-to-destroy, terminal-homing, or engagement logic, consistent with
> the project's public-safety statement.

## What it does

- Models the chaser's motion relative to a target in a near-circular orbit using
  the linear **Clohessy–Wiltshire (Hill) equations** in the target's LVLH frame
  (x = radial / R-bar, y = along-track / V-bar, z = cross-track).
- Plans a **two-impulse rendezvous** to a hold point (arrive with a chosen
  velocity — zero for station-keeping), and a **multi-leg stepped approach**
  through a corridor of hold points (e.g. a safe V-bar approach).
- Reports the **closest approach (conjunction)** over a horizon.
- Shows **how altitude/orbit changes when a burn is introduced**
  (`orbit_change_from_impulse`): before/after semi-major axis, eccentricity, and
  apoapsis/periapsis altitude.

## Quick start

```bash
python -m aerognc.cli rpo --altitude-km 500 --start-behind-m 800 --output results/rpo
```

Example result: a chaser 800 m behind a 500 km target performs a stepped V-bar
approach to 30 m behind it for ~3.6 m/s total Δv, with a 30 m closest approach
(it holds, it does not intercept). Writes `rendezvous.png` and `rendezvous.json`.

## Library usage

```python
import numpy as np
from aerognc.astrodynamics.relative_motion import (
    ClohessyWiltshireModel, simulate_rendezvous, orbit_change_from_impulse,
)

model = ClohessyWiltshireModel.from_orbit(6_878_137.0)          # ~500 km circular
traj = simulate_rendezvous(
    model,
    np.array([300.0, -800.0, 0.0, 0.0, 0.0, 0.0]),             # LVLH state
    [np.array([0.0, -300.0, 0.0]), np.array([0.0, -30.0, 0.0])],
    leg_time_s=500.0,
)
print(traj.total_delta_v_mps, traj.closest_approach_m)

# How a 20 m/s prograde burn reshapes a circular orbit:
change = orbit_change_from_impulse(
    np.array([6_878_137.0, 0, 0]), np.array([0, 7612.6, 0]), np.array([0, 20.0, 0])
)
print(change.apoapsis_altitude_after_m - change.apoapsis_altitude_before_m)
```

## Limitations

- Linear CW dynamics assume a near-circular target orbit and small relative
  separations; large or eccentric cases need nonlinear/elliptic models (future
  work). The interactive RPO sandbox page is deferred (plots + CLI are provided).
