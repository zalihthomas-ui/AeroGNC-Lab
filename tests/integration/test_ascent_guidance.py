import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from aerognc.configuration import load_ascent_guidance_configuration
from aerognc.simulation.guided_ascent import optimize_ascent_guidance
from aerognc.verification.ascent_guidance import (
    assess_ascent_guidance,
    write_ascent_guidance_results,
)
from aerognc.visualisation.ascent_guidance import plot_ascent_guidance


def test_constrained_ascent_optimization_writes_verified_evidence(tmp_path: Path) -> None:
    configuration = load_ascent_guidance_configuration("configs/constrained_ascent_guidance.yaml")
    optimization = optimize_ascent_guidance(configuration)
    assessment = assess_ascent_guidance(optimization)

    assert not optimization.reference_run.all_constraints_satisfied
    assert optimization.optimized_run.all_constraints_satisfied
    assert abs(optimization.optimized_run.apogee_error_m) <= configuration.apogee_tolerance_m
    assert optimization.optimized_run.objective < optimization.reference_run.objective
    assert assessment.all_pass

    reference, optimized, history, report = write_ascent_guidance_results(optimization, tmp_path)
    figure = plot_ascent_guidance(optimization, tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["requirements"]["all_pass"] is True
    assert payload["optimizer"]["evaluation_count"] == len(optimization.evaluations)
    assert reference.stat().st_size > 10_000
    assert optimized.stat().st_size > 10_000
    assert len(history.read_text(encoding="utf-8").splitlines()) > 5
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 40_000
