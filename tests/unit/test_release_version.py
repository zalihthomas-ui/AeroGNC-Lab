"""Release-version consistency checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_release_version import declared_versions, main, validate_release_tag


def _metadata(tmp_path: Path, package: str = "1.2.3", citation: str = "1.2.3") -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "example"\nversion = "{package}"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {citation}\n", encoding="utf-8"
    )
    return tmp_path


def test_repository_release_versions_agree() -> None:
    package, citation = declared_versions()
    assert package == citation
    assert validate_release_tag(f"v{package}") == package


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "latest", "v01.2.3"])
def test_release_tag_must_be_semantic(tmp_path: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="semantic version"):
        validate_release_tag(tag, _metadata(tmp_path))


def test_release_tag_must_match_both_metadata_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metadata disagree"):
        validate_release_tag("v1.2.3", _metadata(tmp_path, citation="1.2.4"))


def test_cli_reports_mismatch(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GITHUB_REF_NAME", "v9.9.9")
    monkeypatch.setattr("scripts.check_release_version.ROOT", _metadata(tmp_path))
    assert main([]) == 1
    assert "does not match" in capsys.readouterr().err
