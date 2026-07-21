# Verification Method

Verification is requirement-led: measurable statements are baselined before their
implementation, evidence uses explicit tolerances, and the traceability matrix is the
release index. Figures communicate behavior but never serve as the only pass proof.

The normative plan and method codes are in
[`requirements/verification_plan.md`](../requirements/verification_plan.md). Evidence
is layered as follows:

1. deterministic unit tests for algebra, signs, ranges, exceptions, and invariants;
2. composed integration tests for configuration, events, logging, GNC, and CLIs;
3. analytical validation for ballistic, constant-force, and rigid-body cases;
4. numerical convergence and independent SciPy-solver comparisons;
5. simplified 3-DOF/6-DOF cross-model consistency;
6. same-seed stochastic reproducibility and requirement-margin analysis;
7. optional cross-language comparison only when the external tool actually ran;
8. a release audit of requirements, documentation, static checks, coverage, and a
   clean installation.

The rotating-planet layer adds round-trip geodesy, near-pole behavior, transport
velocity, separately testable J2/Coriolis/centrifugal terms, and ordered ascent-event
checks. The aerodynamic-database audit verifies tensor completeness, interpolation,
gradients, source hashing, and out-of-domain accounting. Envelope evidence includes
trim residuals, linearisation agreement, ranks, modes, scheduled midpoint stability,
uncertain-case stability, and actuator-authority margins. Constraint-aware ascent
evidence records every offline evaluation and assesses each limit only over its
declared physical domain.

Regression limits are physical/engineering tolerances rather than byte-identical
floating-point or figure comparisons. Randomized models use explicit seed trees.
Monte Carlo pass rates describe the configured synthetic population only; they are
not confidence claims for a physical vehicle.

Reference summaries deliberately exclude wall-clock execution-time values because
they vary with host load. Runtime is measured and checked during the live test/CLI
gate, while deterministic states, events, engineering metrics, and seeded ensembles
form the versioned reference artifacts.

The default local gate is:

```bash
ruff check .
mypy src
pytest --cov=aerognc --cov-report=term-missing
python -m aerognc.cli run --config configs/three_dof_nominal.yaml --no-plots
```

See the [validation report](validation_report.md) for executed evidence and the
[traceability matrix](../requirements/traceability_matrix.csv) for requirement-level
status.
