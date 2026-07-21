import json
from pathlib import Path
from typing import Any

from aerognc.cli import main


def test_orbit_sandbox_cli_writes_scoped_results(tmp_path: Path, capsys: Any) -> None:
    source = Path("configs/orbit_sandbox.yaml").read_text(encoding="utf-8")
    configuration = tmp_path / "orbit.yaml"
    configuration.write_text(
        source.replace("duration_days: 3.0", "duration_days: 0.002"),
        encoding="utf-8",
    )
    output = tmp_path / "orbit_results"

    assert (
        main(
            [
                "orbit-sandbox",
                "--config",
                str(configuration),
                "--output",
                str(output),
                "--no-plots",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "perturbed_decay"
    assert "finite horizon" in payload["limitations"][3]
    assert (output / "orbit_sandbox_trajectory.csv").is_file()
    assert (output / "orbit_sandbox_report.json").is_file()


def test_aircraft_cli_writes_coefficient_driven_results(tmp_path: Path, capsys: Any) -> None:
    source = Path("configs/aircraft_sandbox.yaml").read_text(encoding="utf-8")
    configuration = tmp_path / "aircraft.yaml"
    configuration.write_text(
        source.replace("duration_s: 90.0", "duration_s: 0.2"),
        encoding="utf-8",
    )
    output = tmp_path / "aircraft_results"

    assert (
        main(
            [
                "aircraft",
                "--config",
                str(configuration),
                "--output",
                str(output),
                "--no-plots",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["vehicle"].startswith("Aquila-X1 fictional")
    assert payload["reached_100_km"] is False
    assert (output / "aircraft_trajectory.csv").is_file()
    assert (output / "aircraft_model_report.json").is_file()


def test_live_aircraft_cli_dispatches_mesh_and_controls(monkeypatch: Any) -> None:
    from aerognc.visualisation import aircraft_live

    call: dict[str, object] = {}

    def fake_play(configuration: object, mesh_path: Path, **options: object) -> None:
        call["configuration"] = configuration
        call["mesh"] = mesh_path
        call.update(options)

    monkeypatch.setattr(aircraft_live, "play_aircraft_live", fake_play)
    mesh = Path("assets/models/aquila_x1.obj")
    assert (
        main(
            [
                "fly-aircraft",
                "--config",
                "configs/aircraft_sandbox.yaml",
                "--mesh",
                str(mesh),
                "--mesh-axes",
                "body_frd",
                "--fps",
                "40",
                "--real-time-factor",
                "1.5",
                "--camera",
                "orbit",
                "--no-gamepad",
            ]
        )
        == 0
    )
    assert call["mesh"] == mesh
    assert call["axis_convention"] == "body_frd"
    assert call["frames_per_second"] == 40
    assert call["real_time_factor"] == 1.5
    assert call["camera_mode"] == "orbit"
    assert call["enable_gamepad"] is False
