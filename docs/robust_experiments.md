# Robust Experiment Design, Resumption, and Benchmarking

## Input domains and experiment designs

Every factor has a unique name, finite lower/upper bound, and explicit unit. Physical
sample matrices are rejected if any value is nonfinite or outside the closed domain.
The module provides three complementary designs:

- seeded Latin hypercube sampling with one independently permuted jittered point in
  every stratum of every factor;
- a directly implemented unscrambled Bratley--Fox Sobol sequence for 1--16 factors,
  with a configurable deterministic skip and no false claim of randomized error
  bounds;
- seeded Morris one-at-a-time trajectories on an even-level grid, retaining each
  changed-factor index and signed normalized step.

Morris elementary effects are calculated per unit-domain step. Reports include the
signed mean \(\mu\), absolute mean \(\mu^*\), sample standard deviation \(\sigma\), and
all individual effects. Pearson linear and average-rank Spearman coefficients provide
screening views; neither establishes causality. A seeded nonparametric percentile
bootstrap provides intervals for user-supplied scalar statistics.

The tests independently check Latin stratum occupancy, the first two-dimensional
Sobol Gray-code points, exact Morris effects for a linear response, perfect
linear/rank correlation, repeatable bootstrap intervals, and domain rejection.

## Resumable ensembles

`EnsembleDefinition` hashes schema version, name, evaluator identity, ordered finite
samples, and metadata. The first run writes this manifest. A later run in the same
directory must match it exactly or fails before executing a member.

Each member is an atomic, checksummed JSON record containing index, parameters,
success/failure, metrics, and a contextual error. On resume, every compatible valid
member is reused. Missing or corrupt records are evaluated again; failed records are
retained and reused unless `retry_failed=True` is explicitly selected. A
`new_member_limit` supports controlled partial execution and testable interruption.
Thread scheduling cannot change input order, member identity, aggregate statistics,
correlations, worst-case indices, or the deterministic summary file.

Generic user evaluators run in-process and must therefore be thread safe when more
than one worker is selected. Process isolation and remote schedulers are not claimed.
Large trajectories should remain in the project result store; member JSON should
contain compact metrics.

## Local performance benchmark

The command

```text
python -m aerognc.cli benchmark --config configs/three_dof_nominal.yaml \
  --repetitions 3 --max-wall-time-s 5 \
  --output results/benchmarks/three_dof_benchmark.json
```

runs a deterministic preflight to determine sample/step counts and then records each
measured invocation. The report contains median wall time, process CPU time, peak
Python-traced memory, sample/step throughput, all trials, Python/NumPy/platform/CPU
metadata, and each configured budget decision. Peak traced memory does not include
every native-library allocation.

This is development-machine performance evidence only. It is explicitly not WCET,
hard real-time, deadline, operating-system scheduling, or deployment-hardware proof.
A failed budget is returned as visible verification evidence and a nonzero CLI status;
it is not hidden or reinterpreted as a guarantee.
