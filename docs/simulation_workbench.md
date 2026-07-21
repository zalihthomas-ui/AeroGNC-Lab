# Simulation Workbench User Guide

## What this application is

AeroGNC-Lab is a motion-prediction and verification workbench. A user supplies a
starting state, a mathematical vehicle/environment model, and commands. The solver
integrates the governing equations forward in time and returns the resulting
position, velocity, orientation, rates, loads, events, and performance measures.
The 3D players display those calculated states; they are not prerecorded animations.

The four approachable solver questions are:

1. **How will a fictional research rocket move, and will its attitude controller
   keep it stable?** The nonlinear 6-DOF workflow calculates translation and rotation
   through atmosphere, wind, propulsion, mass change, aerodynamics, actuators, and
   closed-loop attitude control.
2. **Can a fictional spacecraft complete a specified three-world trip within ideal
   velocity-change and remaining-mass limits?** The preliminary patched-conic
   workflow calculates two transfer legs, orbit capture, parking revolutions,
   powered departure, destination capture, and sequential propellant use.
3. **What path will a fictional satellite follow, and how many modeled days does it
   remain above a defined reentry boundary?** The sandbox distinguishes force-free,
   two-body, restricted-three-body, full-N-body and perturbed-decay physics.
4. **How will a fictional research aircraft respond to its aerodynamic coefficients,
   mass, stall boundary and my control inputs?** The same nonlinear plant supports a
   hands-off evidence run and live keyboard/optional-XInput flight.

The intended users are aerospace students learning flight mechanics, engineers
reproducing or comparing algorithm results, educators demonstrating numerical
models, and technical reviewers checking portfolio evidence. It is not a general
"design any mission automatically" product, an operational navigation system, a
certification tool, or a substitute for high-fidelity mission design.

## One-click start

After the one-time Python installation, Windows users can double-click
`run_solver.bat`, `run_aerognc.bat`, or `run_simulation.bat`. All open the unified native desktop
workbench immediately. The equivalent terminal command is:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli workbench
```

The launcher stays local, does not publish anything, and does not require a network
connection. If the `.venv` environment is missing, it prints the exact two setup
commands instead of closing without an explanation.

## First simulation: no input editing required

The **Start** page first explains the three-part workflow: what the user provides,
what the equations calculate, and what the result shows. It then presents four green
one-click actions:

1. **Play Rocket Example** restores the verified rocket preset, calculates it, and
   opens its 3D player.
2. **Play Planet Trip Example** restores the verified fictional route, calculates
   it, and opens its 3D player.
3. **Play Satellite Example** restores the 200 km drag-decay case, propagates it,
   and opens a seekable 3D orbit player.
4. **Fly Aircraft Example** restores the coefficient-driven Aquila-X1 and opens its
   live 3D player.

Close the 3D player to return to the workbench. Use **Change rocket inputs** or
**Change trip/orbit/aircraft inputs** only after the prepared examples are clear. The other Start-page
links are explicitly secondary: **Saved Runs** is for reproducible evidence,
**Astronomy Data** is a read-only catalog, **Engineering Checks** explains
implementation and limitations, and **Advanced Designer** exposes specialist
astrodynamics controls.

Every editable field has a visible unit and a `?` help cue. Hovering over either the
entry or its cue places a plain-language explanation in the status bar. Both
simulator pages load a regression-tested preset and provide a reset button. Decimal
commas are accepted for individual numeric entries, while the underlying calculation
remains in SI units.

The Start and four solver pages are vertically scrollable at smaller supported window
sizes. Use the mouse wheel or the visible right-hand scrollbar. After a calculation,
the page automatically moves to the plain-language result explanation; scroll upward
to change the inputs for another run.

## Rocket Simulator page

The basic page displays only calculation duration, starting speed, and 3D playback
speed. Beside them, a plain-language panel lists exactly what will be calculated:
motion, stability, loads, and events. Leave the prepared values unchanged for the
first run and select **Play This Rocket in 3D**.

**Show Advanced Orientation and Numerical Inputs** reveals the custom RK4 step,
aerospace 3-2-1 roll/pitch/yaw, and forward-right-down body rates. These values are
for controlled experiments such as changing the initial disturbance or conducting a
step-size convergence check. The form validates finite values and the verified model
domain before constructing an immutable `SixDofConfiguration`; it never rewrites the
source YAML. The reference schedule limits this interactive case to eight seconds
and the integration step to 0.02 s.

The result panel first explains what happened and whether the rocket stayed close to
its commanded orientation. It then reports calculated duration/sample count, maximum
altitude and speed, maximum attitude error, maximum body rate, detected events, and
the saved CSV/JSON directory. **Calculate + Save Only** produces the same
numerical evidence without opening another window. In the player, Space pauses,
arrow keys seek or alter speed, `C` changes camera, `R` restarts, and Free view can be
rotated with the mouse.

## Planet Trip Planner page

The basic page displays only the three fully specified fictional worlds and the
departure/intermediate-arrival/destination-arrival days. Beside them, a
plain-language panel explains the calculated route, ideal energy/propellant outputs,
and the meaning of PASS. Leave the prepared route unchanged for the first run and
select **Play This Trip in 3D**.

**Show Optional Orbit, Spacecraft and Limit Inputs** reveals three parking-orbit
altitudes, dwell revolutions, wet/dry mass, ideal specific impulse, delta-v and
final-mass limits, and playback rate. Invalid routes, reversed epochs, fractional
dwell counts, inconsistent masses, or out-of-domain inputs fail with a direct
explanation.

The result is not a decorative animation. AeroGNC-Lab solves both Lambert legs,
samples the captured assist orbit, calculates sphere-of-influence branches,
accounts for every ideal impulsive burn sequentially, applies a dry-mass floor, and
checks the destination endpoint. The displayed narrative, burn list, propellant use,
specific-energy change, and requirement outcome come from that calculation. PASS
means the simplified result meets the entered numerical limits and internal
event/endpoint checks; it does not mean a real mission is flight-ready. The 3D player
shows capture, parking revolutions, powered departure, and destination capture as
distinct phases and events.

This remains preliminary patched-conic analysis. It does not include an operational
ephemeris, finite-burn targeting, navigation operations, landing, interception, or
terminal homing.

## Satellite Orbit page

The first field selects the governing physics in plain language. The no-force case is
explicitly described as straight-line motion, not an orbit. Two body adds the primary;
restricted three body adds one prescribed moon; full N-body lets the primary, all
configured moons and finite-mass satellite accelerate one another; orbit lifetime adds
central gravity, $J_2$, rotating-atmosphere drag and optional ideal corrections.

Basic inputs are starting altitude, calculated/custom speed rule, inclination and the
finite number of days to simulate. The optional panel exposes mass, dry mass, area,
$C_D$, density sensitivity, reentry threshold, integration/output steps and correction
policy. The prepared example leaves corrections off. A result without reentry is stated
only as lifetime greater than the selected horizon; it is never called infinite.

**Calculate + Play Orbit in 3D** writes deterministic CSV/JSON/PNG evidence and opens a
seekable player. `C` switches between the near-satellite and whole-system scales so a
distant moon does not make the low orbit invisible. The density table is a reproducible
synthetic reference, not a space-weather forecast. See [Satellite Orbit Sandbox](orbit_sandbox.md).

## Aircraft Flight page

Basic inputs are altitude, airspeed, heading, angle of attack, throttle and hands-off
duration. The bundled OBJ is ready to use; **Choose File** accepts a bounded OBJ/STL and
an explicit source-axis convention. Mesh triangles change appearance only. They do not
silently infer wing area, mass, inertia or coefficients.

The optional panel exposes initial/dry mass, wing area, $C_{L0}$, $C_{L\alpha}$,
$C_{L\max}$, stall angle, $C_{D0}$, induced-drag factor, $C_{m\alpha}$, initial
climb/bank, wind and live speed. Every engineering field enters the physical equations.
The live HUD shows calculated airspeed, Mach, angle of attack, changing stall speed,
load factor, attitude, mass, control surfaces and warnings.

In live flight, arrows command bank/pitch, A/D rudder, W/S throttle, R held enables
rocket assist, P/Space pauses, C changes camera and Home resets. `T` toggles the
verified fictional 100 km research-ascent attitude aid. XInput is optional and neutral
when unavailable. A 100 km crossing is a model event, not orbital insertion or evidence
that a real aircraft is feasible. See
[Fictional Aircraft Simulation and Live Flight](aircraft_simulation.md).

## Saved Runs page

This page is mainly for engineers and reviewers after they understand a single run.
It opens the checked-in portfolio project by default, or any schema-v1
`.aerognc.yaml` project chosen by the user. The toolbar opens, saves, saves-as, and
validates a portable project file. Validation covers the strict schema, normalized
workspace-contained paths, required configuration files, public-safety statement,
and registered workflow names.

**Run Selected Scenario** calls the same `ProjectRunService` used by the command line
on a background worker. Progress messages are marshalled back to the Tk event thread.
**Cancel Run** stops at the next safe solver boundary and retains an honest terminal
manifest and report. Selecting one completed run opens its local report. Selecting
exactly two compatible runs aligns common same-unit channels and reports bias, RMS,
maximum/final difference, and correlation. Integrity hashes are checked when runs
are loaded.

## Astronomy Data page

This page does not solve spacecraft motion. Its header presents sourced approximate
Milky Way context. One sub-page filters the point-in-time NASA Exoplanet Archive
snapshot by planet/host text, maximum reported distance, exact discovery method,
year range, and row limit. Missing values remain visibly missing rather than being
invented. **Open Selection in 3D** plots reported host positions; it does not
propagate a spacecraft to those systems.

The second sub-page contains the eight IAU Solar System planets and selected sourced
mean properties. Real catalog and Solar System data are read-only descriptive
context. They are intentionally isolated from executable fictional ephemerides,
because confirmed-exoplanet rows generally lack the phase, covariance, orientation,
host state, and common precision epoch required for honest trajectory propagation.

## Engineering Checks page

This page summarizes which mathematics is implemented and which claims are excluded.
It is aimed at engineers, educators, and reviewers who want to distinguish calculated
evidence from scope limitations. Plots are presentation, not proof; the repository
also contains analytical cases, convergence checks, independent-solver comparisons,
cross-model checks, requirement traces, and deterministic regression artifacts.

## Responsiveness and errors

Project, rocket, satellite, aircraft-batch, and planetary calculations run on a background worker while the
workbench shows progress and disables duplicate actions. Completion is returned to
the Tk event thread before a Matplotlib player opens. Validation or solver exceptions
are shown without terminating the application. Cancellation is available for an
active project run.

All executable vehicle and mission parameters are fictional and synthetic. The
project is for education, research, verification practice, and portfolio review; it
contains no proprietary data and excludes target interception and terminal homing.
