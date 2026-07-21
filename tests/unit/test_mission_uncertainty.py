import numpy as np

from aerognc.simulation.mission_uncertainty import run_seeded_uncertainty


def test_seeded_uncertainty_is_reproducible_parallel_and_reports_statistics() -> None:
    def evaluator(parameters: dict[str, float]) -> dict[str, float]:
        return {"response": 3.0 * parameters["input_a"] - parameters["input_b"]}

    first = run_seeded_uncertainty(
        {"input_a": 2.0, "input_b": 0.5}, evaluator, sample_count=40, seed=218, workers=1
    )
    second = run_seeded_uncertainty(
        {"input_a": 2.0, "input_b": 0.5}, evaluator, sample_count=40, seed=218, workers=2
    )

    np.testing.assert_array_equal(
        [run.metrics["response"] for run in first.runs],
        [run.metrics["response"] for run in second.runs],
    )
    assert first.metric_statistics == second.metric_statistics
    assert first.successful_count == 40
    assert first.failed_count == 0
    assert first.correlations["input_a"]["response"] > 0.9
    assert first.correlations["input_b"]["response"] < 0.0


def test_uncertainty_records_failed_evaluations_without_stopping() -> None:
    def evaluator(parameters: dict[str, float]) -> dict[str, float]:
        if parameters["value"] < 0.0:
            raise ValueError("synthetic failure")
        return {"square": parameters["value"] ** 2}

    summary = run_seeded_uncertainty({"value": 1.0}, evaluator, sample_count=20, seed=4, workers=1)

    assert 0 < summary.failed_count < 20
    assert summary.successful_count + summary.failed_count == 20
    assert all(run.error is not None for run in summary.runs if not run.successful)
