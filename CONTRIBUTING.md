# Contributing

Thank you for improving AeroGNC-Lab. Contributions must preserve its civilian,
public-safe scope: fictional research-rocket mechanics, verification, navigation,
and attitude stabilisation only. Target interception, terminal homing, operational
weapon data, and proprietary material are out of scope.

## Development workflow

1. Create a Python 3.12 or newer virtual environment.
2. Install with `python -m pip install -e ".[dev]"`.
3. Add or update a requirement in `requirements/system_requirements.md` for any
   externally visible behaviour.
4. Implement focused code with SI units in names and docstrings.
5. Add deterministic tests and update `requirements/traceability_matrix.csv`.
6. Run `ruff check .`, `mypy src`, and `pytest` before opening a change.

Generated results should be reproducible from a script or CLI command. Do not
commit large ensembles or claim validation that has not actually been executed.
