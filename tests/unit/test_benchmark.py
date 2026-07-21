import json

import pytest

from aerognc.verification.benchmark import (
    BenchmarkBudget,
    benchmark_payload,
    run_benchmark,
    write_benchmark_report,
)


def test_benchmark_records_resources_throughput_environment_and_budgets(tmp_path) -> None:
    calls = 0

    def operation() -> int:
        nonlocal calls
        calls += 1
        return sum(value * value for value in range(2_000))

    result = run_benchmark(
        "deterministic arithmetic",
        operation,
        sample_count=2_000,
        step_count=1_999,
        repetitions=3,
        warmup=True,
        budget=BenchmarkBudget(
            maximum_wall_time_s=10.0,
            maximum_cpu_time_s=10.0,
            maximum_peak_traced_memory_mb=10.0,
            minimum_samples_per_second=1.0,
            minimum_steps_per_second=1.0,
        ),
    )

    assert calls == 4
    assert result.wall_time_s > 0.0
    assert result.cpu_time_s >= 0.0
    assert result.peak_traced_memory_bytes >= 0
    assert result.samples_per_second > 0.0
    assert result.passed
    assert not result.real_time_guarantee
    assert result.environment.python_version

    path = write_benchmark_report(result, tmp_path / "benchmark.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == benchmark_payload(result)
    assert payload["real_time_guarantee"] is False
    assert "not a hard real-time" in payload["scope_note"]


def test_benchmark_failed_budget_is_evidence_not_an_exception() -> None:
    result = run_benchmark(
        "budget failure",
        lambda: None,
        sample_count=1,
        step_count=1,
        repetitions=1,
        warmup=False,
        budget=BenchmarkBudget(minimum_samples_per_second=1.0e30),
    )
    assert not result.passed
    assert result.budget_results == {"minimum_samples_per_second": False}


def test_benchmark_configuration_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        BenchmarkBudget(maximum_wall_time_s=0.0)
    with pytest.raises(ValueError, match="sample_count"):
        run_benchmark("bad", lambda: None, sample_count=0, step_count=1)
