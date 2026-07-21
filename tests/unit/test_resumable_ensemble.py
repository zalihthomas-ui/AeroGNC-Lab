import json

import pytest

from aerognc.simulation.resumable_ensemble import (
    EnsembleDefinition,
    run_resumable_ensemble,
)


def _definition(scale: float = 1.0) -> EnsembleDefinition:
    return EnsembleDefinition(
        "quadratic screening",
        "tests.quadratic.v1",
        tuple({"x": float(index), "scale": scale} for index in range(10)),
        {"purpose": "deterministic unit fixture"},
    )


def _evaluator(parameters: dict[str, float]) -> dict[str, float]:
    if parameters["x"] == 4.0:
        raise FloatingPointError("declared synthetic failure")
    return {"response": parameters["scale"] * parameters["x"] ** 2}


def test_members_persist_resume_without_rerun_and_keep_failed_evidence(tmp_path) -> None:
    calls: list[float] = []

    def evaluator(parameters: dict[str, float]) -> dict[str, float]:
        calls.append(parameters["x"])
        return _evaluator(parameters)

    first = run_resumable_ensemble(
        _definition(), evaluator, tmp_path, workers=1, new_member_limit=4
    )
    assert not first.complete
    assert first.newly_executed_count == 4
    assert calls == [0.0, 1.0, 2.0, 3.0]

    second = run_resumable_ensemble(_definition(), evaluator, tmp_path, workers=3)
    assert second.complete
    assert second.reused_member_count == 4
    assert second.newly_executed_count == 6
    assert len(calls) == 10
    assert [member.index for member in second.members] == list(range(10))
    assert second.failed_count == 1
    assert second.members[4].error == "FloatingPointError: declared synthetic failure"
    report_before = (tmp_path / "ensemble_summary.json").read_text(encoding="utf-8")

    third = run_resumable_ensemble(_definition(), evaluator, tmp_path, workers=2)
    assert third.newly_executed_count == 0
    assert third.reused_member_count == 10
    assert len(calls) == 10
    assert (tmp_path / "ensemble_summary.json").read_text(encoding="utf-8") == report_before


def test_worker_count_does_not_change_ordered_results_or_summaries(tmp_path) -> None:
    serial = run_resumable_ensemble(_definition(), _evaluator, tmp_path / "serial", workers=1)
    parallel = run_resumable_ensemble(_definition(), _evaluator, tmp_path / "parallel", workers=4)

    assert serial.members == parallel.members
    assert serial.metric_statistics == parallel.metric_statistics
    assert serial.correlations == parallel.correlations
    assert serial.worst_case_runs == parallel.worst_case_runs
    serial_payload = json.loads(
        (tmp_path / "serial" / "ensemble_summary.json").read_text(encoding="utf-8")
    )
    parallel_payload = json.loads(
        (tmp_path / "parallel" / "ensemble_summary.json").read_text(encoding="utf-8")
    )
    assert serial_payload == parallel_payload


def test_incompatible_manifest_is_rejected_and_corrupt_member_is_recomputed(tmp_path) -> None:
    run_resumable_ensemble(_definition(), _evaluator, tmp_path)
    corrupt_path = tmp_path / "members" / "00000003.json"
    corrupt_path.write_text("{broken", encoding="utf-8")
    calls: list[float] = []

    def evaluator(parameters: dict[str, float]) -> dict[str, float]:
        calls.append(parameters["x"])
        return _evaluator(parameters)

    resumed = run_resumable_ensemble(_definition(), evaluator, tmp_path)
    assert resumed.complete
    assert calls == [3.0]

    with pytest.raises(ValueError, match="incompatible"):
        run_resumable_ensemble(_definition(scale=2.0), _evaluator, tmp_path)
