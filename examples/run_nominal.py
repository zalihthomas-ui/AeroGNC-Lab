"""Minimal programmatic nominal 3-DOF example."""

from aerognc.configuration import load_three_dof_configuration
from aerognc.simulation.simulator import simulate_three_dof


def main() -> None:
    """Run the fictional nominal case and print key synthetic events."""
    configuration = load_three_dof_configuration("configs/three_dof_nominal.yaml")
    result = simulate_three_dof(configuration)
    for event in result.event_summary:
        print(f"{event['name']}: t={event['time_s']:.3f} s, altitude={event['altitude_m']:.1f} m")


if __name__ == "__main__":
    main()
