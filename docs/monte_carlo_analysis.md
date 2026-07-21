# Coupled Monte Carlo and Performance Verification

The ensemble framework runs three public-safe verification channels for every sample:

1. the fictional 3-DOF ascent with perturbed initial conditions, vehicle mass,
   thrust magnitude/misalignment, drag coefficients, and a newly seeded wind field;
2. the vertical navigation workflow using that perturbed truth with scaled sensor
   noise and bias; and
3. the scalar attitude benchmark with perturbed actuator delay and controller gains.

This coupling demonstrates subsystem-level requirements and performance analysis
without representing an operational vehicle, target, or engagement.

## Reproducibility and parallelism

A NumPy `SeedSequence` deterministically spawns one child seed per ordered run. Every
sample is fully generated before execution, so scheduling across worker processes
cannot change inputs or result order. `workers: 1` uses an easy-to-debug sequential
path; larger values use process-level parallelism at run boundaries. The 12-run
reference case was executed with two workers. Repeating with the same configuration
and master seed produces exactly equal sampled inputs and engineering metrics.

Each member catches model/configuration exceptions and returns a failed record with
its reason. Failed members remain in the CSV and reduce pass rates; successful members
continue into aggregate statistics. Full trajectory arrays are intentionally not
saved for every run.

## Distributions and clipping

The YAML file defines independent one-sigma normal dispersions. Scale factors are
clipped to `[0.25, 3.0]`; initial elevation is bounded to `[75, 90]` degrees and speed
remains positive. These choices make numerical domains explicit. They are synthetic
screening distributions, not estimated manufacturing or flight-test uncertainties.

## Evidence products

For each metric, the summary reports count, mean, sample standard deviation, minimum,
5th/50th/95th percentiles, maximum, and a normal-approximation 95% confidence interval
for the mean. It also provides:

- positive requirement margins and per-requirement/overall pass rates;
- automatic worst-case run indices;
- Pearson input/output correlations for linear sensitivity screening;
- histograms, an empirical CDF, a dispersion scatter plot, a requirement dashboard,
  and a correlation heat map.

Pearson correlation is not causation and a 12-run example is intentionally too small
for formal uncertainty certification. It is a reproducible workflow demonstration;
larger sample counts should be selected after convergence and compute-budget studies.

In the current 12-run seed, all members completed; 11/12 met every configured
requirement. The one failure was a negative landing-range margin, which remains
visible rather than being discarded or hidden.

