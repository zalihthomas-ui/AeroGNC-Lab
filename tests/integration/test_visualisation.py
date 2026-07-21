from pathlib import Path

from aerognc.configuration import load_three_dof_configuration
from aerognc.simulation.simulator import simulate_three_dof
from aerognc.visualisation import plot_three_dof_results


def test_three_dof_publication_figures_are_generated(tmp_path: Path) -> None:
    configuration = load_three_dof_configuration("configs/three_dof_nominal.yaml")
    result = simulate_three_dof(configuration)

    paths = plot_three_dof_results(result, tmp_path)

    assert {path.name for path in paths} == {
        "three_dof_kinematics.png",
        "three_dof_loads.png",
        "three_dof_trajectory.png",
    }
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in paths)
