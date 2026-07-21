"""Read-only environment diagnostics with explicit remediation guidance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

DiagnosticStatus = Literal["pass", "warning", "fail"]


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    """One machine-readable availability or integrity observation."""

    name: str
    category: str
    status: DiagnosticStatus
    required: bool
    detail: str
    remediation: str = ""


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Environment health report; optional-tool warnings do not fail readiness."""

    generated_utc: str
    project_root: str
    result_directory: str
    checks: tuple[DiagnosticCheck, ...]

    @property
    def passed(self) -> bool:
        """Return whether every required check passed."""
        return all(check.status == "pass" for check in self.checks if check.required)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report."""
        return {
            "schema_version": "1.0",
            "generated_utc": self.generated_utc,
            "project_root": self.project_root,
            "result_directory": self.result_directory,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "scope": (
                "Availability and local permission preflight only; no dependency installation, "
                "configuration change, external-tool execution, or physical HIL is performed."
            ),
        }


def _file_check(root: Path, relative_path: str, label: str) -> DiagnosticCheck:
    path = root / relative_path
    if path.is_file() and path.stat().st_size > 0:
        return DiagnosticCheck(label, "project data", "pass", True, str(path))
    return DiagnosticCheck(
        label,
        "project data",
        "fail",
        True,
        f"Required non-empty file is missing: {path}",
        "Restore the file from the AeroGNC-Lab source distribution.",
    )


def _python_check() -> DiagnosticCheck:
    version = platform.python_version()
    minimum = (3, 12)
    current = (sys.version_info.major, sys.version_info.minor)
    if current >= minimum:
        return DiagnosticCheck(
            "Python runtime",
            "runtime",
            "pass",
            True,
            f"Python {version} at {sys.executable}",
        )
    return DiagnosticCheck(
        "Python runtime",
        "runtime",
        "fail",
        True,
        f"Python {version} is below the supported minimum 3.12",
        "Install Python 3.12 or newer, recreate .venv, and reinstall the project.",
    )


def _declared_package_version(root: Path) -> str | None:
    try:
        payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return str(payload["project"]["version"])
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return None


def _package_check(root: Path) -> DiagnosticCheck:
    declared = _declared_package_version(root)
    try:
        installed = importlib.metadata.version("aerognc-lab")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    source_package = root / "src" / "aerognc" / "__init__.py"
    if declared is None or not source_package.is_file():
        return DiagnosticCheck(
            "AeroGNC-Lab package",
            "runtime",
            "fail",
            True,
            "The project manifest or source package is unavailable.",
            "Run the diagnostic from the AeroGNC-Lab repository root.",
        )
    if installed is None:
        return DiagnosticCheck(
            "AeroGNC-Lab package",
            "runtime",
            "fail",
            True,
            f"Source version {declared} exists, but the distribution is not installed.",
            'Create .venv and run: .venv\\Scripts\\python.exe -m pip install -e ".[dev]"',
        )
    if installed != declared:
        return DiagnosticCheck(
            "AeroGNC-Lab package",
            "runtime",
            "fail",
            True,
            f"Installed version {installed} differs from source version {declared}.",
            'Refresh the editable install: .venv\\Scripts\\python.exe -m pip install -e ".[dev]"',
        )
    return DiagnosticCheck(
        "AeroGNC-Lab package",
        "runtime",
        "pass",
        True,
        f"Editable/importable distribution version {installed}",
    )


def _dependency_check(distribution: str, module: str) -> DiagnosticCheck:
    available = importlib.util.find_spec(module) is not None
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    if available and version is not None:
        return DiagnosticCheck(
            distribution,
            "Python dependency",
            "pass",
            True,
            f"Version {version}",
        )
    return DiagnosticCheck(
        distribution,
        "Python dependency",
        "fail",
        True,
        "Required module or distribution metadata is unavailable.",
        'Install project dependencies: .venv\\Scripts\\python.exe -m pip install -e ".[dev]"',
    )


def _catalog_integrity_check(root: Path) -> DiagnosticCheck:
    csv_path = root / "data" / "catalogs" / "nasa_confirmed_exoplanets.csv"
    metadata_path = root / "data" / "catalogs" / "nasa_confirmed_exoplanets.metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_hash = str(metadata["sha256"])
        expected_rows = int(metadata["row_count"])
        actual_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            actual_rows = sum(1 for _line in stream) - 1
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return DiagnosticCheck(
            "Exoplanet catalog integrity",
            "project data",
            "fail",
            True,
            f"Catalog or metadata could not be validated: {error}",
            "Restore both catalog files together or run the documented catalog updater.",
        )
    if actual_hash != expected_hash or actual_rows != expected_rows:
        return DiagnosticCheck(
            "Exoplanet catalog integrity",
            "project data",
            "fail",
            True,
            (
                f"Expected {expected_rows} rows and {expected_hash}; "
                f"got {actual_rows} and {actual_hash}."
            ),
            (
                "Restore the catalog snapshot or regenerate it with "
                "scripts/update_exoplanet_catalog.py."
            ),
        )
    return DiagnosticCheck(
        "Exoplanet catalog integrity",
        "project data",
        "pass",
        True,
        f"{actual_rows} records; SHA-256 {actual_hash}",
    )


def _writable_directory_check(path: Path) -> DiagnosticCheck:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir():
        return DiagnosticCheck(
            "Result location",
            "filesystem",
            "fail",
            True,
            f"No existing parent directory for {path}",
            "Choose a result directory below an existing writable folder.",
        )
    try:
        with tempfile.NamedTemporaryFile(prefix=".aerognc-write-probe-", dir=candidate) as probe:
            probe.write(b"AeroGNC-Lab diagnostic probe\n")
            probe.flush()
    except OSError as error:
        return DiagnosticCheck(
            "Result location",
            "filesystem",
            "fail",
            True,
            f"Cannot write below {candidate}: {error}",
            "Select a writable --result-directory or correct that folder's permissions.",
        )
    detail = f"Write/delete probe passed in {candidate}"
    if not path.exists():
        detail += f"; requested directory {path} can be created there"
    return DiagnosticCheck("Result location", "filesystem", "pass", True, detail)


def _optional_tool_check(name: str, executable_candidates: tuple[str, ...]) -> DiagnosticCheck:
    executable = next(
        (path for candidate in executable_candidates if (path := shutil.which(candidate))), None
    )
    if executable is not None:
        return DiagnosticCheck(
            name,
            "optional tool",
            "pass",
            False,
            f"Detected at {executable}; not executed by this diagnostic.",
        )
    return DiagnosticCheck(
        name,
        "optional tool",
        "warning",
        False,
        "Not detected; the Python core remains fully usable.",
        f"Install {name} only if its independent optional validation workflow is needed.",
    )


def _optional_module_check(name: str, module: str) -> DiagnosticCheck:
    if importlib.util.find_spec(module) is not None:
        return DiagnosticCheck(
            name,
            "optional tool",
            "pass",
            False,
            f"Python module {module} detected; no kernels or workflows were executed.",
        )
    return DiagnosticCheck(
        name,
        "optional tool",
        "warning",
        False,
        f"Python module {module} is not installed; analytical providers remain available.",
        f"Install {module} only when independent {name} validation is planned.",
    )


def run_diagnostics(
    *,
    project_root: str | Path | None = None,
    result_directory: str | Path = "results",
) -> DiagnosticReport:
    """Inspect runtime, core data, result permissions, and optional tools without execution."""
    root = Path.cwd() if project_root is None else Path(project_root)
    root = root.resolve()
    result_path = Path(result_directory)
    if not result_path.is_absolute():
        result_path = root / result_path
    result_path = result_path.resolve()
    checks: list[DiagnosticCheck] = [_python_check(), _package_check(root)]
    checks.extend(
        _dependency_check(distribution, module)
        for distribution, module in (
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("matplotlib", "matplotlib"),
            ("PyYAML", "yaml"),
            ("Pillow", "PIL"),
        )
    )
    checks.extend(
        _file_check(root, relative, label)
        for relative, label in (
            ("configs/three_dof_nominal.yaml", "3-DOF example configuration"),
            ("configs/six_dof_nominal.yaml", "6-DOF example configuration"),
            ("configs/interplanetary_gravity_assist.yaml", "Interplanetary example configuration"),
            ("projects/portfolio_demo.aerognc.yaml", "Portfolio project workspace"),
            ("data/catalogs/milky_way_metadata.yaml", "Milky Way context metadata"),
            ("data/catalogs/solar_system_planets.csv", "Solar System catalog"),
        )
    )
    checks.append(_catalog_integrity_check(root))
    checks.append(_writable_directory_check(result_path))
    checks.extend(
        (
            _optional_tool_check("MATLAB", ("matlab.exe", "matlab")),
            _optional_tool_check("GMAT", ("GMAT.exe", "GMAT", "GMAT-R2022a.exe")),
            _optional_tool_check("FFmpeg", ("ffmpeg.exe", "ffmpeg")),
            _optional_module_check("SPICE", "spiceypy"),
        )
    )
    return DiagnosticReport(
        datetime.now(UTC).isoformat(timespec="seconds"),
        str(root),
        str(result_path),
        tuple(checks),
    )


def format_diagnostic_report(report: DiagnosticReport) -> str:
    """Render a compact terminal report with remediation adjacent to each issue."""
    lines = [
        "AeroGNC-Lab environment diagnostic",
        f"Overall readiness: {'READY' if report.passed else 'NOT READY'}",
        f"Project root: {report.project_root}",
        f"Result directory: {report.result_directory}",
        "",
    ]
    badges = {"pass": "PASS", "warning": "WARN", "fail": "FAIL"}
    for check in report.checks:
        lines.append(f"[{badges[check.status]}] {check.name}: {check.detail}")
        if check.remediation:
            lines.append(f"       Next: {check.remediation}")
    lines.append("")
    lines.append("No tools were executed and no packages or settings were changed.")
    return "\n".join(lines)


def write_diagnostic_report(report: DiagnosticReport, path: str | Path) -> Path:
    """Persist the report as JSON; this is the diagnostic's only durable mutation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
