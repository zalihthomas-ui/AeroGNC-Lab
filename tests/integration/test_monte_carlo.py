from dataclasses import replace
from pathlib import Path

from aerognc.configuration import load_monte_carlo_configuration
from aerognc.simulation.monte_carlo import run_monte_carlo

PROJECT_ROOT = Path(__file__).parents[2]


def test_coupled_monte_carlo_is_reproducible_and_reports_evidence() -> None:
    configuration = load_monte_carlo_configuration(PROJECT_ROOT / "configs" / "monte_carlo.yaml")
    first = run_monte_carlo(configuration, sample_count=2, workers=1)
    second = run_monte_carlo(configuration, sample_count=2, workers=1)
    assert first.runs == second.runs
    assert first.statistics == second.statistics
    assert first.failed_count == 0
    assert "apogee_m" in first.statistics
    assert "overall" in first.requirement_pass_rates
    assert first.worst_case_runs
    different = run_monte_carlo(
        replace(configuration, master_seed=configuration.master_seed + 1),
        sample_count=2,
        workers=1,
    )
    assert first.runs[0].sample != different.runs[0].sample
