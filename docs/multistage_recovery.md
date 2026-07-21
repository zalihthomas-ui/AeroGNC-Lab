# Multistage and Recovery Analysis

This workflow is a fictional civilian research-rocket demonstration. Its parameters
are synthetic and do not represent an operational vehicle. It is intended to make
stage-event, mass-floor, and deployable-drag logic inspectable before those components
are coupled into higher-fidelity flight models.

## Quick start

```bash
python -m aerognc.cli multistage-recovery \
  --config configs/multistage_recovery.yaml
```

The command writes a unit-labelled CSV trajectory, JSON event/maximum summary, and a
four-panel figure. The YAML exposes payload/stage masses, motor tables, ignition and
separation times, recovery timing/areas, atmosphere, gravity, step size, and horizon.

## Stage state machine

A stage is `attached` before ignition, `burning` until motor burnout, `spent` until
separation, then `separated`. Propellant is full before ignition and follows the
piecewise-linear impulse law during burn. Stages are ordered; overlapping burns and
ignition before prior separation are rejected. Simultaneous separation and next-stage
ignition use a stable order: burnout, separation, then ignition.

The continuity assessment verifies:

- zero thrust at ignition and burnout table endpoints;
- no mass below payload plus attached dry masses;
- a separation mass drop equal to the depleted stage dry mass;
- deterministic event naming and time order.

## Recovery schedule

After trigger plus deployment delay, area ramps linearly from zero to the reefed
value over `reefing_time_s`. It holds for `reefed_hold_time_s`, then ramps to full area
over `inflation_time_s`. Drag is always opposite air-relative vertical velocity. The
opening-load history is the recovery-device drag magnitude, not a structural stress
or canopy-line-load prediction.

The demonstration integrates

\[
\dot h=-v_D,\qquad
\dot v_D=g+\frac{F_{D,body}+F_{D,recovery}-T}{m(t)},
\]

with positive-down velocity. Apogee is an increasing zero crossing of \(v_D\);
ground contact is a decreasing zero crossing of altitude and terminates propagation.

## Limitations

- One vertical axis and constant density; no horizontal wind or attitude coupling.
- Prescribed stage times; no ignition-transient or separation-contact dynamics.
- No canopy mass, added mass, porosity, oscillation, line dynamics, or failure modes.
- The timer-triggered reference can begin deployment before or after apogee; a future
  coupled workflow can bind the trigger to a detected event.
- Results support software verification and preliminary trades only.
