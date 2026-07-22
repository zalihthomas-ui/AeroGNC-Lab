## Summary

Describe the change and the user or developer impact.

## Motivation

Explain why the change is needed and link any related requirement or issue.

## Verification

List the checks and evidence used to validate the change.

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy src`
- [ ] `pytest`

## Engineering checklist

- [ ] Units, frames, sign conventions, and assumptions are explicit.
- [ ] Tests are deterministic and cover externally visible behavior.
- [ ] Requirements, traceability, documentation, and reference evidence are updated when needed.
- [ ] Generated outputs are reproducible and no credentials or machine-specific files are included.
- [ ] The change preserves AeroGNC-Lab's fictional, civilian, public-safe scope.
