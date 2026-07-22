import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from aerognc.cli import main

PROJECT_ROOT = Path(__file__).parents[2]


def _write_cli_project(tmp_path: Path) -> Path:
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    project_path = tmp_path / "project.aerognc.yaml"
    project_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "workspace_root": ".",
                "name": "CLI-Project",
                "description": "CLI project integration fixture.",
                "safety_scope": ("Fictional civilian research vehicle with synthetic parameters."),
                "settings": {
                    "result_directory": "stored-runs",
                    "default_seed": 19,
                    "max_workers": 1,
                },
                "scenarios": [
                    {
                        "name": "nominal",
                        "workflow": "three-dof",
                        "configuration": "configs/three_dof_nominal.yaml",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project_path


def test_cli_run_writes_reproducible_core_outputs(tmp_path: Path) -> None:
    status = main(
        [
            "run",
            "--config",
            str(PROJECT_ROOT / "configs" / "three_dof_nominal.yaml"),
            "--output",
            str(tmp_path),
            "--no-plots",
        ]
    )
    assert status == 0
    assert (tmp_path / "trajectory.csv").stat().st_size > 1_000
    assert (tmp_path / "summary.json").is_file()


def test_cli_waypoint_accepts_versioned_runtime_configuration(tmp_path: Path) -> None:
    status = main(
        [
            "waypoint",
            "--config",
            str(PROJECT_ROOT / "configs" / "waypoint_gnc.yaml"),
            "--output",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert status == 0
    payload = json.loads((tmp_path / "mission_log.json").read_text(encoding="utf-8"))
    assert payload["summary"]["completed"] is True
    assert payload["metadata"]["navigation_provider"] == "PerfectStateProvider"
    assert payload["metadata"]["vehicle_backend"] == "InternalFixedWingBackend"
    provenance = payload["metadata"]["runtime_configuration"]
    assert provenance["name"] == "waypoint_demo_internal"
    assert provenance["schema_version"] == 1
    assert len(provenance["sha256"]) == 64
    assert len(provenance["mission_sha256"]) == 64


def test_cli_waypoint_preserves_mission_only_form(tmp_path: Path) -> None:
    status = main(
        [
            "waypoint",
            "--mission",
            str(PROJECT_ROOT / "missions" / "waypoint_demo.mission.yaml"),
            "--output",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert status == 0
    payload = json.loads((tmp_path / "mission_log.json").read_text(encoding="utf-8"))
    assert payload["summary"]["completed"] is True
    assert payload["metadata"]["navigation_provider"] == "PerfectStateProvider"
    assert "runtime_configuration" not in payload["metadata"]


def test_cli_benchmark_writes_scoped_resource_evidence(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    status = main(
        [
            "benchmark",
            "--config",
            str(PROJECT_ROOT / "configs" / "three_dof_nominal.yaml"),
            "--repetitions",
            "1",
            "--max-wall-time-s",
            "20",
            "--max-peak-memory-mb",
            "100",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["sample_count"] > 100
    assert payload["steps_per_second"] > 0.0
    assert payload["real_time_guarantee"] is False


def test_cli_udp_loopback_writes_localhost_only_evidence(tmp_path: Path) -> None:
    output = tmp_path / "udp_loopback.json"
    status = main(
        [
            "udp-loopback",
            "--samples",
            "12",
            "--receive-timeout-ms",
            "20",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["state_endpoint"]["packets_accepted"] == 12
    assert payload["command_endpoint"]["packets_accepted"] == 12
    assert payload["localhost_only"] is True
    assert payload["physical_hil_executed"] is False


def test_cli_diagnose_reports_ready_repository(tmp_path: Path) -> None:
    output = tmp_path / "health.json"
    status = main(
        [
            "diagnose",
            "--project-root",
            str(PROJECT_ROOT),
            "--result-directory",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert any(check["name"] == "MATLAB" for check in payload["checks"])


def test_cli_rotating_six_dof_and_multistage_workflows(tmp_path: Path) -> None:
    rotating_output = tmp_path / "rotating-six"
    status = main(
        [
            "rotating-six-dof",
            "--config",
            str(PROJECT_ROOT / "configs" / "rotating_six_dof.yaml"),
            "--output",
            str(rotating_output),
            "--no-plots",
        ]
    )
    assert status == 0
    assert (rotating_output / "rotating_six_dof_trajectory.csv").stat().st_size > 10_000

    recovery_output = tmp_path / "recovery"
    status = main(
        [
            "multistage-recovery",
            "--config",
            str(PROJECT_ROOT / "configs" / "multistage_recovery.yaml"),
            "--output",
            str(recovery_output),
            "--no-plots",
        ]
    )
    assert status == 0
    summary = json.loads(
        (recovery_output / "multistage_recovery_summary.json").read_text(encoding="utf-8")
    )
    assert summary["events"][-1]["name"] == "ground_contact"


def test_cli_play_dispatches_headless_export(tmp_path: Path, monkeypatch: Any) -> None:
    from aerognc.visualisation import playback

    animation_path = tmp_path / "playback.gif"
    call: dict[str, object] = {}

    def fake_playback(result: object, **kwargs: object) -> Path:
        call["result"] = result
        call.update(kwargs)
        animation_path.write_bytes(b"GIF89a-test")
        return animation_path

    monkeypatch.setattr(playback, "play_three_dof", fake_playback)
    status = main(
        [
            "play",
            "--config",
            str(PROJECT_ROOT / "configs" / "three_dof_nominal.yaml"),
            "--speed",
            "8",
            "--fps",
            "24",
            "--save-gif",
            str(animation_path),
            "--no-window",
        ]
    )

    assert status == 0
    assert animation_path.is_file()
    assert call["playback_speed"] == 8.0
    assert call["frames_per_second"] == 24
    assert call["show_window"] is False


def test_cli_play_requires_an_export_when_window_is_disabled() -> None:
    status = main(
        [
            "play",
            "--config",
            str(PROJECT_ROOT / "configs" / "three_dof_nominal.yaml"),
            "--no-window",
        ]
    )
    assert status == 2


def test_cli_play_3d_dispatches_headless_export(tmp_path: Path, monkeypatch: Any) -> None:
    from aerognc.visualisation import playback_3d

    animation_path = tmp_path / "playback_3d.gif"
    call: dict[str, object] = {}

    def fake_playback(result: object, **kwargs: object) -> Path:
        call["result"] = result
        call.update(kwargs)
        animation_path.write_bytes(b"GIF89a-3d-test")
        return animation_path

    monkeypatch.setattr(playback_3d, "play_six_dof_3d", fake_playback)
    status = main(
        [
            "play-3d",
            "--config",
            str(PROJECT_ROOT / "configs" / "six_dof_nominal.yaml"),
            "--speed",
            "8",
            "--fps",
            "24",
            "--camera",
            "chase",
            "--save-gif",
            str(animation_path),
            "--no-window",
        ]
    )

    assert status == 0
    assert animation_path.is_file()
    assert call["playback_speed"] == 8.0
    assert call["frames_per_second"] == 24
    assert call["camera_mode"] == "chase"
    assert call["show_window"] is False


def test_cli_play_3d_requires_an_export_when_window_is_disabled() -> None:
    status = main(
        [
            "play-3d",
            "--config",
            str(PROJECT_ROOT / "configs" / "six_dof_nominal.yaml"),
            "--no-window",
        ]
    )
    assert status == 2


def test_cli_interplanetary_solves_and_writes_outputs_without_window(tmp_path: Path) -> None:
    status = main(
        [
            "interplanetary",
            "--config",
            str(PROJECT_ROOT / "configs" / "interplanetary_gravity_assist.yaml"),
            "--output",
            str(tmp_path),
            "--no-window",
        ]
    )
    assert status == 0
    trajectory = tmp_path / "interplanetary_trajectory.csv"
    summary = tmp_path / "interplanetary_summary.json"
    assert trajectory.stat().st_size > 1_000_000
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert [event["name"] for event in payload["events"]] == [
        "departure_injection",
        "assist_entry",
        "assist_closest_approach",
        "assist_exit",
        "destination_arrival",
        "mission_end",
    ]


def test_cli_mission_designer_dispatches_catalog_and_verified_paths(monkeypatch: Any) -> None:
    from aerognc.visualisation import mission_designer

    call: dict[str, object] = {}

    def fake_launch(catalog_path: Path, verified_path: Path) -> None:
        call["catalog"] = catalog_path
        call["verified"] = verified_path

    monkeypatch.setattr(mission_designer, "launch_mission_designer", fake_launch)
    catalog = PROJECT_ROOT / "configs" / "fictional_planetary_system.yaml"
    verified = PROJECT_ROOT / "configs" / "interplanetary_gravity_assist.yaml"
    status = main(
        [
            "mission-designer",
            "--catalog",
            str(catalog),
            "--verified-config",
            str(verified),
        ]
    )

    assert status == 0
    assert call == {"catalog": catalog, "verified": verified}


def test_cli_workbench_dispatches_all_local_resources(monkeypatch: Any) -> None:
    from aerognc.visualisation import workbench

    call: list[Path] = []

    def fake_launch(*paths: Path) -> None:
        call.extend(paths)

    monkeypatch.setattr(workbench, "launch_workbench", fake_launch)
    expected = [
        PROJECT_ROOT / "configs" / "six_dof_nominal.yaml",
        PROJECT_ROOT / "configs" / "orbit_assisted_tour.yaml",
        PROJECT_ROOT / "configs" / "fictional_planetary_system.yaml",
        PROJECT_ROOT / "configs" / "interplanetary_gravity_assist.yaml",
        PROJECT_ROOT / "data" / "catalogs" / "nasa_confirmed_exoplanets.csv",
        PROJECT_ROOT / "data" / "catalogs" / "nasa_confirmed_exoplanets.metadata.json",
        PROJECT_ROOT / "data" / "catalogs" / "milky_way_metadata.yaml",
        PROJECT_ROOT / "data" / "catalogs" / "solar_system_planets.csv",
        PROJECT_ROOT / "projects" / "portfolio_demo.aerognc.yaml",
    ]
    status = main(
        [
            "workbench",
            "--six-dof-config",
            str(expected[0]),
            "--orbit-tour-config",
            str(expected[1]),
            "--planetary-catalog",
            str(expected[2]),
            "--verified-interplanetary-config",
            str(expected[3]),
            "--exoplanet-csv",
            str(expected[4]),
            "--exoplanet-metadata",
            str(expected[5]),
            "--milky-way-metadata",
            str(expected[6]),
            "--solar-system-planets",
            str(expected[7]),
            "--project-file",
            str(expected[8]),
        ]
    )

    assert status == 0
    assert call == expected


def test_cli_software_loopback_writes_deterministic_report(tmp_path: Path) -> None:
    output = tmp_path / "loopback.json"
    arguments = [
        "software-loopback",
        "--samples",
        "50",
        "--seed",
        "44",
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    first = output.read_bytes()
    assert main(arguments) == 0
    assert output.read_bytes() == first
    payload = json.loads(first)
    assert payload["sample_count"] == 50
    assert payload["model"].endswith("no physical HIL")


def test_cli_writes_honest_fmi_interface_contract(tmp_path: Path) -> None:
    output = tmp_path / "fmi"

    assert main(["fmi-interface", "--output", str(output)]) == 0
    assert (output / "modelDescription.xml").is_file()
    payload = json.loads((output / "STATUS.json").read_text("utf-8"))
    assert payload["artifact_type"].endswith("interface contract only")
    assert payload["fmu_built"] is False


def test_cli_catalog_filters_bundled_snapshot(tmp_path: Path) -> None:
    catalog_directory = PROJECT_ROOT / "data" / "catalogs"
    status = main(
        [
            "catalog",
            "--csv",
            str(catalog_directory / "nasa_confirmed_exoplanets.csv"),
            "--metadata",
            str(catalog_directory / "nasa_confirmed_exoplanets.metadata.json"),
            "--galaxy-metadata",
            str(catalog_directory / "milky_way_metadata.yaml"),
            "--solar-system",
            str(catalog_directory / "solar_system_planets.csv"),
            "--query",
            "TRAPPIST-1",
            "--output",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert status == 0
    payload = json.loads((tmp_path / "galaxy_catalog_summary.json").read_text("utf-8"))
    assert payload["selection_summary"]["planet_count"] == 7


def test_cli_project_lifecycle_and_run_comparison(tmp_path: Path, capsys: Any) -> None:
    empty_directory = tmp_path / "empty-project"
    assert main(["project", "init", str(empty_directory), "--name", "Empty-Project"]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert Path(initialized["project"]).is_file()

    project_path = _write_cli_project(tmp_path / "analysis")
    assert main(["project", "validate", str(project_path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["issues"] == []
    assert validated["scenarios"][0]["name"] == "nominal"

    assert main(["project", "run", str(project_path), "nominal"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert Path(first["report"]).is_file()
    assert first["requirements_pass"] is True

    assert main(["project", "run", str(project_path), "nominal"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["run_id"] != first["run_id"]

    assert main(["project", "list", str(project_path), "--status", "completed"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert {item["run_id"] for item in records} == {first["run_id"], second["run_id"]}

    comparison_path = tmp_path / "comparison.json"
    assert (
        main(
            [
                "project",
                "compare",
                str(project_path),
                first["run_id"],
                second["run_id"],
                "--channels",
                "altitude_m,mass_kg",
                "--output",
                str(comparison_path),
            ]
        )
        == 0
    )
    comparison_output = json.loads(capsys.readouterr().out)
    assert comparison_path.is_file()
    assert Path(comparison_output["report"]).is_file()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert all(item["rms_difference"] == 0.0 for item in comparison["channels"])

    regenerated = tmp_path / "regenerated.html"
    assert (
        main(
            [
                "project",
                "report",
                str(project_path),
                first["run_id"],
                "--output",
                str(regenerated),
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert regenerated.is_file()
