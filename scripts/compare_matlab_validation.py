"""Generate Python evidence and compare it with optional MATLAB output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerognc.verification.cross_language import (
    compare_cross_language_results,
    load_constant_acceleration_case,
    simulate_constant_acceleration,
    write_state_csv,
)


def main() -> int:
    """Run the shared case and emit an honest cross-language report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-matlab",
        action="store_true",
        help="fail if MATLAB output has not been generated",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validation_directory = root / "matlab_validation"
    output_directory = validation_directory / "output"
    case = load_constant_acceleration_case(validation_directory / "constant_acceleration_case.json")
    python_result = simulate_constant_acceleration(case)
    write_state_csv(
        output_directory / "constant_acceleration_python.csv",
        python_result.time_s,
        python_result.state,
    )
    matlab_path = output_directory / "constant_acceleration_matlab.csv"
    if arguments.require_matlab and not matlab_path.exists():
        raise FileNotFoundError(
            "MATLAB output is absent; run validate_constant_acceleration in MATLAB first"
        )
    comparison = compare_cross_language_results(
        case, python_result, matlab_path if matlab_path.exists() else None
    )
    report_path = output_directory / "cross_language_comparison.json"
    report_path.write_text(
        json.dumps(comparison.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison.as_dict(), indent=2, sort_keys=True))
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
