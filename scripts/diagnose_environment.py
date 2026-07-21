"""Portable entry point for AeroGNC-Lab environment diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aerognc.diagnostics import (  # noqa: E402
    format_diagnostic_report,
    run_diagnostics,
    write_diagnostic_report,
)


def main(arguments: list[str] | None = None) -> int:
    """Run without importing simulation dependencies or changing the environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "diagnostics" / "health.json",
    )
    options = parser.parse_args(arguments)
    report = run_diagnostics(
        project_root=PROJECT_ROOT,
        result_directory=options.result_directory,
    )
    write_diagnostic_report(report, options.output)
    print(format_diagnostic_report(report))
    print(f"JSON report: {options.output}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
