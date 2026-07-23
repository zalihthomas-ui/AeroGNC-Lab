# AeroGNC-Lab Roadmap

This roadmap turns the current alpha platform into a maintainable, independently
reviewable research product. It records intent rather than promising dates. Safety,
traceability, reproducibility, and simulation-only defaults are release gates for
every milestone.

## Current baseline — 0.8

The repository already provides deterministic 3-DOF and 6-DOF flight simulation,
fixed-wing and waypoint workflows, navigation and control analysis, astrodynamics,
Monte Carlo and robust experiments, portable project evidence, a native workbench,
and extensive requirement-linked validation. The current hardening work adds typed
distributions, cross-platform CI, supply-chain scanning, dependency maintenance,
release attestations, and an auditable release procedure.

## 0.9 — Fixed-wing autonomy and interoperability

- Add a higher-fidelity coefficient-driven fixed-wing backend and cross-model
  validation against the reduced waypoint plant.
- Wire sensor simulation, strapdown mechanisation, and the existing error-state
  filters into a truth-isolated estimated-navigation provider with dropout/recovery
  evidence.
- Add trim-aware controller initialization, total-energy control for altitude and
  airspeed, improved fillet/orbit transitions, and quantitative envelope margins.
- Add optional JSBSim interoperability and simulation-only ArduPilot/PX4 SITL
  adapters behind strict dependency, connection, acknowledgement, timeout, and
  real-output gates.
- Complete provenance-rich mission logs, deterministic replay, scenario/fault
  campaigns, property-based tests, parser fuzzing, mutation checks, and performance
  budgets.

## 1.0 — Stable engineering product

- Stabilize documented Python result/configuration APIs and split the large CLI and
  workbench adapters into focused application services and views.
- Finish accessible keyboard flows, text alternatives for engineering plots,
  responsive layouts, live setup preview, and structured telemetry panels.
- Provide a versioned local service boundary for notebook or web clients without
  weakening file, network, or hardware safety controls.
- Establish compatibility and deprecation policy, contributor/reviewer guidance,
  reproducible benchmark baselines, and an independently reviewed validation pack.
- Publish signed/provenance-attested artifacts through the protected release process
  once the package registry trusted publisher is configured.

## Beyond 1.0

Candidate research areas include calibrated aerodynamic databases, flexible-body and
canopy coupling, higher-order atmosphere and wind models, lever-arm/time-aligned
navigation, global sensitivity indices, low-thrust optimization, atmospheric entry,
executed FMI/Simulink/GMAT/SPICE comparisons, and measured processor/HIL studies.
Each requires explicit data provenance and independent acceptance criteria before it
can become a verified capability.

## Boundaries that do not change

AeroGNC-Lab remains a fictional, civilian research and education platform. The
roadmap does not include target interception, terminal homing, engagement logic,
weapon-system data, autonomous real-aircraft command, certification claims, or
unrecorded proprietary inputs. ArduPilot/PX4 work is limited to local software-in-
the-loop by default; physical output remains disabled unless a separately reviewed,
explicit future requirement establishes a safe test boundary.

Implementation status is tracked in GitHub issues/milestones and, for the current
waypoint program, in `TODO.md`. Completed roadmap claims must link to requirements,
tests, and validation evidence rather than relying on the roadmap text itself.
