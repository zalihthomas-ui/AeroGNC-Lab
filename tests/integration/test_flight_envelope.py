import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from aerognc.configuration import load_flight_envelope_configuration
from aerognc.gnc.flight_envelope import analyze_flight_envelope
from aerognc.verification.flight_envelope import write_flight_envelope_results
from aerognc.visualisation.flight_envelope import plot_flight_envelope


def test_flight_envelope_workflow_writes_machine_and_human_evidence(tmp_path: Path) -> None:
    configuration = load_flight_envelope_configuration("configs/flight_envelope.yaml")
    result = analyze_flight_envelope(configuration)

    report_path, points_path = write_flight_envelope_results(result, tmp_path)
    figure_path = plot_flight_envelope(result, tmp_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["grid"]["point_count"] == 36
    assert report["requirements"]["all_pass"] is True
    assert report["robustness_verification"]["random_seed"] == 314159
    assert len(points_path.read_text(encoding="utf-8").splitlines()) == 37
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure_path.stat().st_size > 30_000
