"""Repository governance, packaging, and traceability policy checks."""

from __future__ import annotations

import csv
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
REQUIREMENT_PATTERN = re.compile(r"^- \*\*(SYS-[A-Z]+-\d{3}) \(", re.MULTILINE)
PINNED_ACTION_PATTERN = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
VERIFICATION_COMMANDS = {"mypy src", "pytest", "ruff check ."}


def _base_yaml(path: Path) -> dict[str, object]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def test_ci_has_one_canonical_coverage_run_and_compatibility_matrix() -> None:
    workflow = _base_yaml(WORKFLOWS / "ci.yml")
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    push = triggers["push"]
    assert isinstance(push, dict)
    assert push["branches"] == ["main"]
    assert "pull_request" in triggers

    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert text.count("--cov=aerognc") == 1
    assert 'python-version: "3.12"' in text
    assert 'python-version: "3.14"' in text
    assert "windows-latest" in text
    assert "Build and clean-install package" in text
    assert "joinpath('py.typed').is_file()" in text


def test_all_external_github_actions_are_commit_pinned() -> None:
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for use in PINNED_ACTION_PATTERN.findall(text):
            action, separator, revision = use.partition("@")
            assert separator and action, f"invalid action reference in {workflow}: {use}"
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                f"action must use an immutable commit SHA in {workflow}: {use}"
            )


def test_security_dependency_and_release_automation_is_present() -> None:
    security = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    dependabot = _base_yaml(ROOT / ".github" / "dependabot.yml")
    codeowners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")

    assert "security-extended" in security
    assert "dependency-review-action" in security
    assert "pip-audit" in security
    assert "attest-build-provenance" in release
    assert "gh-action-pypi-publish" in release
    assert "id-token: write" in release
    assert "scripts/check_release_version.py" in release
    assert "--cov-fail-under=75" in release
    updates = dependabot["updates"]
    assert isinstance(updates, list)
    ecosystems = {entry["package-ecosystem"] for entry in updates if isinstance(entry, dict)}
    assert ecosystems == {"pip", "github-actions"}
    assert "@zalihthomas-ui" in codeowners


def test_distribution_declares_typing_and_project_links() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert "Typing :: Typed" in project["classifiers"]
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]
    assert set(project["urls"]) == {
        "Homepage",
        "Documentation",
        "Repository",
        "Issues",
        "Changelog",
    }
    assert (ROOT / "src" / "aerognc" / "py.typed").is_file()
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.csv text eol=lf" in attributes
    assert "*.json text eol=lf" in attributes
    assert "*.bat text eol=crlf" in attributes


def test_every_requirement_has_one_valid_traceability_row() -> None:
    requirement_text = (ROOT / "requirements" / "system_requirements.md").read_text(
        encoding="utf-8"
    )
    requirement_ids = REQUIREMENT_PATTERN.findall(requirement_text)
    assert len(requirement_ids) == len(set(requirement_ids)), "duplicate requirement identifiers"

    with (ROOT / "requirements" / "traceability_matrix.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    trace_ids = [row["Requirement"] for row in rows]
    assert len(trace_ids) == len(set(trace_ids)), "duplicate traceability identifiers"
    assert set(trace_ids) == set(requirement_ids)

    for row in rows:
        assert row["Status"] == "Verified", row["Requirement"]
        assert set(row["Method"].split("/")) <= {"A", "C", "D", "T"}
        for column in ("Implementation", "Verification evidence"):
            paths = [item.strip() for item in row[column].split(";")]
            assert paths and all(paths), f"empty {column} for {row['Requirement']}"
            for relative_path in paths:
                if column == "Verification evidence" and relative_path in VERIFICATION_COMMANDS:
                    continue
                assert (ROOT / relative_path).exists(), (
                    f"missing {column} path for {row['Requirement']}: {relative_path}"
                )
