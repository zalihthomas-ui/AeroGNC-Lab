"""Run trim/TECS/geometric-path acceptance on both internal waypoint plants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerognc.configuration.waypoint_loader import load_waypoint_runtime_configuration
from aerognc.mission import load_mission
from aerognc.verification.waypoint_control import (
    run_waypoint_control_campaign,
    write_waypoint_control_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify solved trim, TECS, fillets, orbit tangencies, and margins."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/waypoint_gnc_tecs.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/waypoint_control_campaign.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the campaign and return nonzero if a declared bound fails."""
    arguments = _parser().parse_args(argv)
    runtime = load_waypoint_runtime_configuration(arguments.config)
    result = run_waypoint_control_campaign(
        load_mission(runtime.mission_path),
        runtime.build_mission_config(),
    )
    output = write_waypoint_control_campaign(result, arguments.output)
    print(json.dumps({"output": str(output), **result.summary()}, indent=2, allow_nan=False))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
