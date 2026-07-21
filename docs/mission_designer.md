# Mission Designer User Guide

## Immediate start

On Windows, double-click `run_interplanetary.bat` to launch this detailed native
Mission Designer directly. `run_aerognc.bat` opens the unified Simulation Workbench,
whose **Advanced Designer** button opens this interface. Neither publishes data nor
contacts a remote service. From a terminal, the equivalent command is:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli mission-designer
```

The form uses a wholly fictional Helios system loaded from
`configs/fictional_planetary_system.yaml`. Every editable value shows its unit. A
catalog day is elapsed time from a synthetic epoch, not a real calendar date.

## Guided workflow

The numbered tabs follow an engineering workflow:

1. **Mission** selects direct Lambert or one preliminary gravity assist, the worlds,
   and ordered departure/flyby/arrival epochs.
2. **Spacecraft** collects wet/dry mass, ideal specific impulse, parking altitudes,
   and minimum flyby altitude.
3. **Maneuvers** accepts named impulsive corrections using elapsed day, an RTN or
   inertial frame, three delta-velocity components in m/s, and specific impulse.
4. **Analysis** defines C3 and arrival-speed screening limits, plot resolution,
   trajectory samples, Monte Carlo count, seed, and ephemeris source.
5. **Results** reports pass/review status, energy, delta-v, ideal propellant,
   destination miss, flyby compatibility, and a mission event table.

The default Asteria-to-Neria example is intentionally understandable and solves in
about a second. Press **Design Mission**, inspect the result, then press **Open 3D
Simulation**. A manual direct-transfer correction is applied at its exact epoch; it
changes the propagated path, remaining mass, arrival-relative speed, and destination
miss. It is not a decorative arrow. Corrections are disabled for the preliminary
gravity-assist designer because they would invalidate the matched leg boundary.

## Three distinct fidelity labels

The UI deliberately separates calculations that are often conflated:

- **Direct Lambert** is a two-body endpoint solution and playable conic.
- **One gravity assist** joins two Lambert legs, evaluates incoming/outgoing
  planet-relative excess velocity, B-plane geometry, altitude, and powered mismatch.
  It is a preliminary patched-conic design.
- **Verified N-body example** runs the slower configured restricted N-body RK4 case,
  in which primary and prescribed planet gravity act simultaneously through the
  Brontes encounter. This is the one-click reference simulation.

No preliminary plot is labelled as a high-fidelity operational ephemeris. “Arrival”
means reaching the configured synthetic corridor unless ideal capture delta-v is
explicitly discussed; it does not imply landing.

## Launch windows and uncertainty

**Porkchop Plot** evaluates a two-dimensional epoch grid with the directly
implemented zero-revolution Lambert solver. Filled contours show departure C3,
labelled lines show arrival excess speed, shaded cells violate the entered limits,
and a star marks the best feasible grid member. For a gravity-assist request the
plot screens the first leg.

**Uncertainty** makes repeatable Gaussian epoch draws from the displayed seed. Each
sample is independently solved; failures are recorded rather than aborting the
ensemble. Results include mean and central 95% percentile bounds. This fast design
screen is distinct from the slower full trajectory-dispersion framework in
`simulation/mission_uncertainty.py`.

## Saving and optional data

**Save Inputs** writes a readable YAML design record containing the safety scope,
route, epochs, mass assumptions, and maneuvers. It does not overwrite the verified
scenario. The built-in analytical ephemeris always works offline. The optional SPICE
adapter requires both `spiceypy` and user-supplied public kernels; if either is
missing, it raises an explicit availability error. AeroGNC-Lab never substitutes
fabricated external-ephemeris results.

This interface is for education, research, and portfolio demonstration. It contains
no target state, pursuit, interception, terminal homing, or operational engagement
mode.
