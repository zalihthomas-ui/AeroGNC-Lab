import json
import subprocess
import sys
from pathlib import Path


def test_ui_capability_probe_serves_offline_prototype_without_claiming_qt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "probe.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/ui_capability_probe.py",
            "--skip-tk-live",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    results = {item["name"]: item for item in payload["results"]}
    assert results["Tk/ttk"]["available"] is True
    assert results["Tk/ttk"]["live_probe_executed"] is False
    assert results["stdlib local-web prototype"]["live_probe_executed"] is True
    assert (
        "not connected to an engineering solver" in results["stdlib local-web prototype"]["detail"]
    )
    assert results["PySide6/Qt"]["live_probe_executed"] is False
