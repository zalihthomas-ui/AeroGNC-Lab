"""DOE-to-persistent-ensemble reproducibility and sensitivity integration case."""

from collections.abc import Mapping

import numpy as np

from aerognc.simulation.resumable_ensemble import (
    EnsembleDefinition,
    run_resumable_ensemble,
)
from aerognc.verification.design_of_experiments import (
    Factor,
    bootstrap_confidence_interval,
    latin_hypercube_design,
    sensitivity_correlations,
)


def _screening_evaluator(parameters: Mapping[str, float]) -> Mapping[str, float]:
    if parameters["gain"] > 1.65:
        raise RuntimeError("synthetic unstable design point")
    response = 2.0 * parameters["gain"] - 0.5 * parameters["delay_s"]
    return {"response": response, "requirement_margin": 3.0 - abs(response)}


def test_lhs_resumes_and_preserves_worker_independent_sensitivity_evidence(tmp_path) -> None:
    factors = (Factor("gain", -2.0, 2.0), Factor("delay_s", 0.0, 1.0, "s"))
    design = latin_hypercube_design(factors, 24, seed=731)
    samples = tuple(
        {factor.name: float(design.samples[row, column]) for column, factor in enumerate(factors)}
        for row in range(design.samples.shape[0])
    )
    definition = EnsembleDefinition(
        "robust controller screen",
        "tests.linear-screen.v1",
        samples,
        {"design": "latin_hypercube", "seed": 731},
    )

    partial = run_resumable_ensemble(
        definition,
        _screening_evaluator,
        tmp_path / "resumed",
        workers=1,
        new_member_limit=7,
    )
    assert not partial.complete
    resumed = run_resumable_ensemble(
        definition, _screening_evaluator, tmp_path / "resumed", workers=4
    )
    serial = run_resumable_ensemble(
        definition, _screening_evaluator, tmp_path / "serial", workers=1
    )

    assert resumed.complete
    assert resumed.members == serial.members
    assert resumed.metric_statistics == serial.metric_statistics
    assert resumed.correlations == serial.correlations
    assert resumed.failed_count > 0
    successful = [member for member in resumed.members if member.success]
    successful_samples = np.array(
        [[member.parameters[factor.name] for factor in factors] for member in successful]
    )
    response = np.array([member.metrics["response"] for member in successful])
    correlations = sensitivity_correlations(successful_samples, response, factors)
    assert correlations[0].linear > 0.98
    assert correlations[0].rank > 0.95
    interval = bootstrap_confidence_interval(response, seed=44, resamples=400)
    assert interval.lower <= interval.estimate <= interval.upper
    assert (tmp_path / "resumed" / "ensemble_summary.json").is_file()
