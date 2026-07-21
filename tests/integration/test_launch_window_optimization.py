import json
from pathlib import Path

from aerognc.configuration.launch_window_loader import load_launch_window_configuration
from aerognc.verification.launch_window import (
    run_launch_window_optimization,
    write_launch_window_report,
)
from aerognc.visualisation.launch_window import plot_launch_window_optimization


def test_configured_launch_window_meets_requirements_and_writes_evidence(
    tmp_path: Path,
) -> None:
    configuration = load_launch_window_configuration(
        Path("configs/launch_window_optimization.yaml")
    )
    run = run_launch_window_optimization(configuration)
    report_path = write_launch_window_report(run, tmp_path)
    figure_path = plot_launch_window_optimization(run, tmp_path)

    assert run.assessment.all_pass
    assert run.optimization.optimum.total_delta_v_mps < 7_500.0
    assert run.endpoint_error_m < 0.1
    assert figure_path.stat().st_size > 25_000
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["requirements"]["all_pass"] is True
    assert payload["optimizer"]["evaluation_count"] >= 100
