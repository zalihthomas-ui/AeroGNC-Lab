"""Release-version consistency checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "aerognc_release_check", ROOT / "scripts" / "check_release_version.py"
)
assert SPEC is not None and SPEC.loader is not None
release_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_check)


def _metadata(tmp_path: Path, package: str = "1.2.3", citation: str = "1.2.3") -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "example"\nversion = "{package}"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {citation}\n", encoding="utf-8"
    )
    return tmp_path


def test_repository_release_versions_agree() -> None:
    package, citation = release_check.declared_versions()
    assert package == citation
    assert release_check.validate_release_tag(f"v{package}") == package


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "latest", "v01.2.3"])
def test_release_tag_must_be_semantic(tmp_path: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="semantic version"):
        release_check.validate_release_tag(tag, _metadata(tmp_path))


def test_release_tag_must_match_both_metadata_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metadata disagree"):
        release_check.validate_release_tag("v1.2.3", _metadata(tmp_path, citation="1.2.4"))


def test_cli_reports_mismatch(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GITHUB_REF_NAME", "v9.9.9")
    monkeypatch.setattr(release_check, "ROOT", _metadata(tmp_path))
    assert release_check.main([]) == 1
    assert "does not match" in capsys.readouterr().err
