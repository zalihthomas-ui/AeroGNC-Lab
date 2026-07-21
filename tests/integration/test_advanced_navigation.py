import json
from pathlib import Path

from aerognc.configuration.advanced_navigation_loader import (
    load_advanced_navigation_configuration,
)
from aerognc.simulation.advanced_navigation import (
    run_navigation_consistency,
    simulate_advanced_navigation,
)
from aerognc.verification.advanced_navigation import (
    assess_advanced_navigation,
    write_advanced_navigation_results,
)
from aerognc.visualisation.advanced_navigation import plot_advanced_navigation

PROJECT_ROOT = Path(__file__).parents[2]


def test_advanced_navigation_meets_requirements_and_writes_evidence(tmp_path: Path) -> None:
    configuration = load_advanced_navigation_configuration(
        PROJECT_ROOT / "configs" / "advanced_navigation.yaml"
    )
    result = simulate_advanced_navigation(configuration)
    consistency = run_navigation_consistency(configuration, run_count=4)
    assessment = assess_advanced_navigation(result, consistency)
    paths = write_advanced_navigation_results(result, consistency, tmp_path)
    figure_path = plot_advanced_navigation(result, consistency, tmp_path)

    assert assessment.all_pass
    assert result.observability_rank == 15
    assert result.maximum_replayed_step_count == 18
    assert consistency.nees_inside_fraction >= 0.80
    assert all(path.stat().st_size > 500 for path in paths)
    assert figure_path.stat().st_size > 20_000
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert payload["requirements"]["all_pass"] is True
    assert payload["fixed_lag_and_integrity"]["sensors"]["gnss"]["rejected_count"] >= 5
