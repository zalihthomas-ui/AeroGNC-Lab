"""Run the matched reduced/coefficient waypoint validation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerognc.configuration.waypoint_loader import load_waypoint_runtime_configuration
from aerognc.mission import load_mission
from aerognc.verification.waypoint_backends import (
    compare_waypoint_vehicle_models,
    write_waypoint_cross_model_comparison,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the two simulation-only fixed-wing waypoint backends."
    )
    parser.add_argument(
        "--reduced-config",
        type=Path,
        default=Path("configs/waypoint_gnc.yaml"),
    )
    parser.add_argument(
        "--coefficient-config",
        type=Path,
        default=Path("configs/waypoint_gnc_coefficient.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/waypoint_backend_comparison.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the matched mission and return nonzero if acceptance fails."""
    arguments = _parser().parse_args(argv)
    reduced = load_waypoint_runtime_configuration(arguments.reduced_config)
    coefficient = load_waypoint_runtime_configuration(arguments.coefficient_config)
    if reduced.mission_sha256 != coefficient.mission_sha256:
        raise ValueError("backend comparison configurations must reference the same mission")
    mission = load_mission(reduced.mission_path)
    comparison = compare_waypoint_vehicle_models(
        mission,
        reduced.build_mission_config(),
        coefficient.build_mission_config(),
    )
    output = write_waypoint_cross_model_comparison(comparison, arguments.output)
    print(json.dumps({"output": str(output), **comparison.summary()}, indent=2))
    return 0 if comparison.passed else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
