"""Run the deterministic estimated-navigation outage/recovery campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerognc.configuration.waypoint_loader import load_waypoint_runtime_configuration
from aerognc.verification.waypoint_navigation import (
    run_waypoint_navigation_campaign,
    write_waypoint_navigation_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score truth-isolated waypoint navigation through GNSS outage and recovery."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/waypoint_gnc_estimated.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/waypoint_navigation_dropout.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the campaign and return nonzero when a declared bound fails."""
    arguments = _parser().parse_args(argv)
    runtime = load_waypoint_runtime_configuration(arguments.config)
    parameters = runtime.navigation.estimated_parameters
    if parameters is None:
        raise ValueError("navigation verification config must select estimated mode")
    result = run_waypoint_navigation_campaign(parameters)
    output = write_waypoint_navigation_campaign(result, arguments.output)
    print(json.dumps({"output": str(output), **result.summary()}, indent=2, allow_nan=False))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
