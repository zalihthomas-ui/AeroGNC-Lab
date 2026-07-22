"""Tests for the versioned, simulation-only waypoint runtime configuration."""

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from aerognc.configuration.loader import ConfigurationError
from aerognc.configuration.waypoint_loader import (
    WAYPOINT_CONFIGURATION_VERSION,
    WaypointNavigationMode,
    load_waypoint_runtime_configuration,
)
from aerognc.gnc.waypoint_guidance import GuidanceMode
from aerognc.navigation.providers import NoisyStateProvider, PerfectStateProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED = REPO_ROOT / "configs" / "waypoint_gnc.yaml"
MISSION = REPO_ROOT / "missions" / "waypoint_demo.mission.yaml"


def _configured_copy(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> Path:
    payload: dict[str, object] = yaml.safe_load(BUNDLED.read_text(encoding="utf-8"))
    payload["mission_file"] = str(MISSION)
    mutate(payload)
    path = tmp_path / "waypoint.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_bundled_waypoint_runtime_loads_every_subsystem() -> None:
    runtime = load_waypoint_runtime_configuration(BUNDLED)
    config = runtime.build_mission_config()

    assert runtime.schema_version == WAYPOINT_CONFIGURATION_VERSION
    assert len(runtime.source_sha256) == 64
    assert runtime.mission_path == MISSION
    assert len(runtime.mission_sha256) == 64
    assert runtime.output_directory == Path("results/waypoint_gnc")
    assert runtime.navigation.mode is WaypointNavigationMode.PERFECT
    assert runtime.allow_real_vehicle_output is False
    assert isinstance(config.provider, PerfectStateProvider)
    assert config.guidance_mode is GuidanceMode.VECTOR_FIELD
    assert config.dt_s == pytest.approx(0.05)
    assert config.autopilot_gains.course_kp == pytest.approx(1.2)
    assert config.safety_limits.geofence_radius_m == pytest.approx(5000.0)
    assert config.reduced_params.roll_from_aileron == pytest.approx(45.0)
    assert config.configuration_name == runtime.name
    assert config.configuration_sha256 == runtime.source_sha256
    assert config.mission_sha256 == runtime.mission_sha256


def test_noisy_navigation_builds_a_fresh_provider_per_run(tmp_path: Path) -> None:
    def mutate(payload: dict[str, object]) -> None:
        navigation = payload["navigation"]
        assert isinstance(navigation, dict)
        navigation["mode"] = "noisy"
        noisy = navigation["noisy"]
        assert isinstance(noisy, dict)
        noisy["gps_dropout_window_s"] = [20.0, 30.0]

    runtime = load_waypoint_runtime_configuration(_configured_copy(tmp_path, mutate))
    first = runtime.build_mission_config().provider
    second = runtime.build_mission_config().provider

    assert runtime.navigation.mode is WaypointNavigationMode.NOISY
    assert isinstance(first, NoisyStateProvider)
    assert isinstance(second, NoisyStateProvider)
    assert first is not second


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update({"schema_version": 99}),
            "waypoint.schema_version",
        ),
        (
            lambda payload: payload.update({"unexpected": True}),
            "unknown keys",
        ),
        (
            lambda payload: cast_mapping(payload["hardware"]).update(
                {"allow_real_vehicle_output": True}
            ),
            "real output is unavailable",
        ),
        (
            lambda payload: cast_mapping(payload["vehicle"]).update({"backend": "jsbsim"}),
            "only internal_reduced is available",
        ),
        (
            lambda payload: cast_mapping(cast_mapping(payload["navigation"])["noisy"]).update(
                {"gps_dropout_window_s": [30.0, 20.0]}
            ),
            "expected nonnegative increasing times",
        ),
    ],
)
def test_invalid_waypoint_runtime_fails_contextually(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_waypoint_runtime_configuration(_configured_copy(tmp_path, mutate))


def cast_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value
