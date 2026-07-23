# Contributing

Thank you for improving AeroGNC-Lab. Contributions are reviewed for numerical
correctness, traceability, reproducibility, maintainability, and public safety.

## Scope and conduct

Contributions must preserve the project's fictional, civilian, public-safe scope.
Target interception, terminal homing, engagement logic, operational weapon data,
classified material, proprietary vehicle information, and real-aircraft command
paths are out of scope. Be constructive, document assumptions, and critique the
engineering rather than the contributor.

Security vulnerabilities belong in a private advisory, not a public issue; see
[SECURITY.md](SECURITY.md).

## Development setup

Use a supported CPython version (3.12, 3.13, or 3.14):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Change workflow

1. Open or reference an issue for a material feature or defect.
2. Create a focused branch from the latest `main`.
3. Add or update a measurable requirement in
   `requirements/system_requirements.md` for externally visible behaviour.
4. Implement focused code with explicit frames, SI units, bounded inputs, and
   deterministic seeds where randomness is required.
5. Add tests and exactly one matching row in
   `requirements/traceability_matrix.csv`.
6. Update user-facing documentation and `CHANGELOG.md` when behaviour changes.
7. Open a pull request using the repository template and respond to review findings.

Keep commits coherent and use an imperative summary such as `Add fixed-lag replay
validation`. Avoid mixing generated results, formatting-only edits, and functional
changes unless they are inseparable.

## Required local checks

```bash
ruff check .
ruff format --check .
mypy src
pytest --cov=aerognc --cov-branch --cov-report=term-missing --cov-fail-under=75
python -m build
python -m twine check dist/*
```

CI repeats the quality, coverage, compatibility, Windows, package-install, security,
and dependency checks. External GitHub Actions must remain pinned to immutable
commits. Dependency changes should explain why the dependency is needed, its licence,
and how its supported range was selected.

Generated evidence must be reproducible from a checked-in script or CLI command. Do
not commit large transient ensembles, secrets, personal data, or results that cannot
be regenerated. Never claim validation against hardware or external software unless
that execution actually occurred and its provenance is recorded.

## Releases

Maintainers release from a verified `main` branch. Versioning, artifact checks,
attestation, GitHub Releases, and trusted PyPI publication are defined in the
[release process](docs/release_process.md). Contributors should not create release
tags from feature branches.
