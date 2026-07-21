import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from aerognc.configuration.analysis_loader import (
    load_flight_control_analysis_configuration,
)
from aerognc.verification.flight_control_analysis import (
    run_flight_control_analysis,
    write_flight_control_analysis,
)
from aerognc.visualisation.flight_control_analysis import plot_flight_control_analysis


def test_configured_flight_control_analysis_writes_finite_evidence(tmp_path: Path) -> None:
    configuration = load_flight_control_analysis_configuration(
        "configs/flight_control_analysis.yaml"
    )
    result = run_flight_control_analysis(configuration)

    assert result.trim.converged
    np.testing.assert_allclose(result.nominal_model.derivative_at_trim, np.zeros(2), atol=1.0e-12)
    assert result.lqr.riccati_residual_norm < 1.0e-10
    assert all(mode.stable for mode in result.modes)
    assert result.margins.phase_margin_deg > 45.0
    assert result.sil_timing.missed_deadline_count == 0

    report_path = write_flight_control_analysis(result, tmp_path, include_timing=False)
    figure_path = plot_flight_control_analysis(result, tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["stability_margins"]["gain_margin"] is None
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure_path.stat().st_size > 20_000
