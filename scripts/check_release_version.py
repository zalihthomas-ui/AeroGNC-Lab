"""Reject a release tag that disagrees with package and citation metadata."""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"^version:\s*['\"]?([^'\"\s]+)['\"]?\s*$", re.MULTILINE)
SEMVER_PATTERN = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def declared_versions(root: Path = ROOT) -> tuple[str, str]:
    """Return package and citation versions from *root*."""
    package = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    package_version = package["version"]
    citation_text = (root / "CITATION.cff").read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(citation_text)
    if match is None:
        raise ValueError("CITATION.cff has no top-level version field")
    return str(package_version), match.group(1)


def validate_release_tag(tag: str, root: Path = ROOT) -> str:
    """Return the released version or raise for malformed/mismatched metadata."""
    if SEMVER_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"release tag must be semantic version vX.Y.Z, got {tag!r}")
    tagged_version = tag[1:]
    package_version, citation_version = declared_versions(root)
    if package_version != citation_version:
        raise ValueError(
            "release metadata disagree: "
            f"pyproject.toml={package_version!r}, CITATION.cff={citation_version!r}"
        )
    if tagged_version != package_version:
        raise ValueError(f"release tag {tag!r} does not match declared version {package_version!r}")
    return package_version


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    tag = args[0] if args else os.environ.get("GITHUB_REF_NAME", "")
    try:
        version = validate_release_tag(tag, ROOT)
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"release version check failed: {exc}", file=sys.stderr)
        return 1
    print(f"release metadata agree on {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
