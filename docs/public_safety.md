# Public-Safety Scope

AeroGNC-Lab is a civilian educational and engineering-portfolio project. Its
Asteria-SR1 atmospheric vehicle, Selene Pathfinder spacecraft, Helios planetary
system, bodies, phases, and trajectories are fictional; every mass property, thrust
value, aerodynamic coefficient, sensor characteristic, disturbance, and result is
synthetic. The repository was produced from open academic flight-mechanics concepts
and contains no classified, export-controlled, employer-proprietary, or flight-program
data.

## Included scope

- local-frame point-mass and rigid-body mechanics;
- research/sounding-rocket ascent and passive descent;
- generic attitude stabilisation and angular-rate damping;
- a time-indexed, non-operational pitch/reference schedule;
- civilian-GNSS-like, inertial, and barometric measurement simulation;
- generic restricted N-body propagation, transfer estimates, and an unpowered
  gravity-assist demonstration in a synthetic planetary system;
- generic Kepler/Lambert launch-window analysis, standard planetary-flyby B-plane
  mapping, fictional maneuver/propellant studies, and a guided civilian Mission
  Designer; "B-plane targeting" refers only to a scientific planet-relative flyby
  plane and never a vehicle/person target;
- rotating-oblate-planet geodesy, aerodynamic-database inspection, flight-envelope
  analysis, and constraint-aware civilian research-ascent studies;
- a fictional civilian satellite sandbox covering force-free, two-body,
  restricted-three-body, mutually interacting N-body, and reference-atmosphere
  orbit-decay studies with idealized correction impulses;
- a fictional civilian research-aircraft model with synthetic aerodynamics, stall
  behavior, pilot input, and optional research-ascent rocket assist; and
- numerical verification, dispersions, and synthetic flight-test analysis.

## Deliberate exclusions

The package has no target-state input and implements no target interception,
pursuit, proportional navigation against a target, terminal homing, engagement
logic, warhead/effect model, threat model, operational envelope, or representation
of a real missile, interceptor, or launch system. Contribution review must reject
changes that cross this boundary.

Results are software verification evidence for the declared fictional configuration.
They are not design values, safety-of-flight approval, operational predictions, or a
substitute for certified analysis. MATLAB is only an optional independent numerical
check; future HIL documentation is a plan and does not claim hardware testing.
The interplanetary example is not a real ephemeris, launch window, navigation product,
orbit-capture plan, or flight-ready mission design.
The satellite lifetime result is conditional on a fixed synthetic density profile,
and the aircraft model is not flight-test-calibrated, type-certified, structurally or
thermally qualified, or a launch/safety-of-flight analysis. Imported 3D meshes are
visual only. Crossing the 100 km reference altitude is not orbital insertion.
