# Release Process

This document defines how AeroGNC-Lab produces reviewable releases without storing
long-lived publication credentials in the repository. Zalih Thomas is the project
author and release owner. A release is a packaging and evidence event; it does not
change the project's research-only, non-certified safety scope.

## Release gates

A release candidate must satisfy all of the following before a tag is created:

- the intended commit is on `main` and the working tree is clean;
- the CI, compatibility, package, CodeQL, and dependency-audit checks pass;
- every baselined requirement has exactly one verified traceability row;
- the version and release notes agree across package and citation metadata;
- the complete branch-aware suite exceeds the 75% coverage floor;
- the wheel and source distribution pass `twine check` and a clean wheel install;
- safety limitations and unexecuted external-validation claims remain explicit.

The maintainer reviews `CHANGELOG.md`, `docs/validation_report.md`, open high-priority
issues, and dependency/security alerts as part of the release decision. A failed or
cancelled check is not waived by tagging a different commit.

## Version preparation

Versions follow Semantic Versioning while the package is pre-1.0. Before release,
update the authoritative package version in `pyproject.toml`, the version in
`CITATION.cff`, user-facing status text, and any versioned interface contracts that
intentionally track the package. Move the relevant `CHANGELOG.md` entries from
`Unreleased` to a dated version heading. Search the repository for the previous
version to catch stale publication metadata; historical release notes should not be
rewritten.

Run the local acceptance sequence documented in `CONTRIBUTING.md`. Build from a clean
tree and inspect both artifacts:

```bash
python -m build
python -m twine check dist/*
```

Install the wheel into a new virtual environment, run `pip check`, import `aerognc`,
confirm the reported version, and confirm that `aerognc/py.typed` is present. The CI
package job independently repeats these checks.

## Tag and automated publication

After the release commit reaches a fully green `main`, create and push an annotated
tag whose name exactly matches the package version:

```bash
git tag -a vX.Y.Z -m "AeroGNC-Lab X.Y.Z"
git push origin vX.Y.Z
```

The tag starts `.github/workflows/release.yml`. That workflow rebuilds the wheel and
source distribution from the tagged commit, validates their metadata, uploads them as
retained workflow artifacts, creates a GitHub build-provenance attestation, and opens
a GitHub Release with generated change notes that the maintainer checks against the
matching changelog entry. PyPI publication runs
only in the protected `pypi` environment and uses OpenID Connect trusted publishing;
no API token is stored in GitHub.

The repository automation is ready for trusted publishing, but a maintainer must
configure the PyPI project once before the first upload. On PyPI, create or claim the
`aerognc-lab` project and add a GitHub trusted publisher for owner
`zalihthomas-ui`, repository `AeroGNC-Lab`, workflow `release.yml`, and environment
`pypi`. On GitHub, configure the `pypi` environment with the desired reviewer policy.
Until both sides are configured, the GitHub Release and attestation may succeed while
the PyPI job remains intentionally unavailable.

## Post-release verification

Verify that the GitHub Release points to the intended tag and commit, both artifacts
are attached, the provenance record is visible, and the release notes describe known
limitations. If PyPI publication is enabled, install the exact released version into
a fresh environment and run the documented CLI smoke command. Confirm that the PyPI
metadata links back to the repository, documentation, issue tracker, and changelog.

If a release is incorrect, do not silently replace artifacts. Stop or disable the
publishing environment, document the incident, and issue a corrected patch version.
Yank a PyPI version only when installation should be discouraged, and retain the
GitHub record with a clear explanation. Git tags and published version numbers are
immutable; a failed release number is never reused.

## Release authority and safety

Only maintainers authorized by Zalih Thomas may approve the protected publication
environment or push release tags. Release automation never enables real-vehicle
output, bypasses the simulation-only hardware gate, or converts validation evidence
into a certification claim. Secrets, proprietary data, and operational mission data
must not appear in artifacts, changelogs, logs, or attestations.
