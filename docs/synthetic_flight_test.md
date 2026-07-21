# Synthetic Flight-Test Workflow

This event-reconstruction workflow is complemented by the independent
[asynchronous flight-data identification workflow](flight_data_identification.md),
which recovers logger timing, preserves gaps, rejects isolated corruptions, estimates
physical pitch-plant parameters, diagnoses residuals, and predicts a held-out record.

This workflow demonstrates flight-test planning and evaluation without using real
vehicle or weapon-system data. The source is the fictional nominal truth simulation;
all sensor errors and outages are synthetic.

## Data plan and CSV definition

The common log rate follows the 50 Hz truth time base. Lower-rate sensors leave blank
fields, preserving missingness rather than interpolating it into the source record.

| Column | Unit | Definition |
|---|---|---|
| `time_s` | s | Acquisition time from synthetic ignition |
| `accelerometer_up_mps2` | m/s² | Ideal attitude/gravity-compensated upward channel plus configured errors |
| `barometric_altitude_m` | m | Geometric altitude above launch datum |
| `gnss_altitude_m` | m | Civilian-GNSS-like local altitude |
| `gnss_vertical_velocity_up_mps` | m/s | Civilian-GNSS-like upward velocity |

The committed repository does not retain the full generated CSV. Reproduce it with
the `flight-test` CLI. Acquisition/delivery delay is applied during generation; the
measurement file is indexed by acquisition time, matching a post-flight time-aligned
engineering data product.

## Processing pipeline

1. Run nominal truth and configurable sensors.
2. Write the measurement-only CSV (no truth columns).
3. Reload it through `csv.DictReader` with strict column and monotonic-time checks.
4. Reconstruct altitude/vertical velocity with the scoped navigation filter.
5. Detect burnout from the largest early negative acceleration step.
6. Detect apogee from the interpolated positive-to-negative reconstructed-velocity
   crossing.
7. Identify impact at the end of the event-truncated record.
8. Compare event times, apogee, peak vertical velocity, and data availability against
   expected synthetic performance; write an automatic JSON summary.

The detection methods are intentionally simple and documented. Impact detection from
record termination assumes the logging system ends on the simulation's terminal
impact event; a real campaign would require independent discrete/event channels and
post-test data-integrity checks.
