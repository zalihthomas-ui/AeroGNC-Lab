"""Stable high-level Python API for the waypoint fixed-wing GNC workflow.

This façade lets AeroGNC-Lab be used as a library, not only through the CLI.
Import ``aerognc`` and call :func:`fly_mission` with a mission (or a path to a
mission file) and optional parameters; you get back a
:class:`~aerognc.simulation.waypoint_mission.WaypointMissionResult` you can
inspect, export (CSV/JSON), or plot.

```python
import aerognc

result = aerognc.fly_mission("missions/waypoint_demo.mission.yaml", wind_east_mps=4.0)
print(result.summary())
result.to_csv("mission_log.csv")
```

The lower-level building blocks remain importable from their subpackages; this
module just provides a convenient, documented entry point that stays stable while
internals evolve.
"""

from pathlib import Path

from aerognc.gnc.waypoint_guidance import GuidanceMode
from aerognc.mission.mission import Mission
from aerognc.mission.mission_io import load_mission
from aerognc.simulation.waypoint_mission import (
    WaypointMissionConfig,
    WaypointMissionResult,
    run_waypoint_mission,
)


def fly_mission(
    mission: Mission | str | Path,
    *,
    guidance: GuidanceMode | str = GuidanceMode.VECTOR_FIELD,
    wind_north_mps: float = 0.0,
    wind_east_mps: float = 0.0,
    dt_s: float = 0.05,
    max_time_s: float = 900.0,
    config: WaypointMissionConfig | None = None,
) -> WaypointMissionResult:
    """Fly a waypoint mission on the internal simulator and return the result.

    ``mission`` may be a :class:`Mission` or a path to a mission YAML file. Pass a
    fully-specified ``config`` to override everything, or use the convenience
    keyword arguments for the common knobs (guidance mode and steady wind). This
    runs the internal simulation backend only and commands no hardware.
    """
    resolved_mission = (
        mission if isinstance(mission, Mission) else load_mission(mission)
    )
    if config is None:
        config = WaypointMissionConfig(
            dt_s=dt_s,
            max_time_s=max_time_s,
            guidance_mode=GuidanceMode(guidance),
            wind_ned_mps=(wind_north_mps, wind_east_mps, 0.0),
        )
    return run_waypoint_mission(resolved_mission, config)
