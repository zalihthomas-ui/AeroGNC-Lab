"""Publication-ready plots for the fictional coefficient-driven aircraft sandbox."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

from aerognc.simulation.aircraft_sandbox import AircraftSandboxSimulation
from aerognc.visualisation.style import BLUE, GREEN, NAVY, ORANGE, RED, engineering_style


def plot_aircraft_sandbox(
    simulation: AircraftSandboxSimulation, output_directory: str | Path
) -> tuple[Path, Path]:
    """Save a 3D path and coefficient/control diagnostic dashboard."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = simulation.result
    columns = result.columns
    with engineering_style():
        figure = plt.figure(figsize=(9.0, 6.8))
        axis = cast(Axes3D, figure.add_subplot(111, projection="3d"))
        axis.plot(
            columns["east_m"] / 1_000.0,
            columns["north_m"] / 1_000.0,
            columns["altitude_m"] / 1_000.0,
            color=BLUE,
            linewidth=2.0,
        )
        axis.scatter(
            [columns["east_m"][0] / 1_000.0],
            [columns["north_m"][0] / 1_000.0],
            [columns["altitude_m"][0] / 1_000.0],
            color=GREEN,
            s=35,
            label="Start",
        )
        axis.scatter(
            [columns["east_m"][-1] / 1_000.0],
            [columns["north_m"][-1] / 1_000.0],
            [columns["altitude_m"][-1] / 1_000.0],
            color=RED,
            s=35,
            label="End",
        )
        axis.set_title("Aquila-X1 coefficient-driven 6-DOF path")
        axis.set_xlabel("East from start [km]")
        axis.set_ylabel("North from start [km]")
        axis.set_zlabel("Altitude [km]")
        axis.legend()
        path_3d = output / "aircraft_trajectory_3d.png"
        figure.savefig(path_3d, dpi=220)
        plt.close(figure)

        figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
        time_s = result.time_s
        axes[0, 0].plot(time_s, columns["altitude_m"], color=BLUE)
        axes[0, 0].set_ylabel("Altitude [m]")
        speed_axis = axes[0, 0].twinx()
        speed_axis.plot(time_s, columns["true_airspeed_mps"], color=GREEN)
        speed_axis.plot(
            time_s,
            columns["stall_speed_mps"],
            color=RED,
            linestyle="--",
        )
        speed_axis.set_ylabel("Airspeed / stall speed [m/s]")
        axes[0, 0].set_title("Altitude (blue), airspeed (green), stall speed (red)")

        axes[0, 1].plot(time_s, columns["angle_of_attack_deg"], color=ORANGE, label="AoA")
        stall_angle = np.rad2deg(simulation.configuration.aerodynamics.stall_angle_rad)
        axes[0, 1].axhline(stall_angle, color=RED, linestyle="--", label="Stall onset")
        axes[0, 1].axhline(-stall_angle, color=RED, linestyle="--")
        axes[0, 1].fill_between(
            time_s,
            -stall_angle,
            stall_angle,
            color=GREEN,
            alpha=0.08,
        )
        axes[0, 1].set_ylabel("Angle [deg]")
        axes[0, 1].set_title("Angle of attack and configured stall boundary")
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].plot(time_s, columns["lift_coefficient"], color=BLUE, label="CL")
        axes[1, 0].plot(time_s, columns["drag_coefficient"], color=RED, label="CD")
        axes[1, 0].plot(
            time_s,
            columns["pitch_moment_coefficient"],
            color=ORANGE,
            label="Cm",
        )
        axes[1, 0].set_ylabel("Coefficient [-]")
        axes[1, 0].set_title("Coefficients used by the force/moment calculation")
        axes[1, 0].legend(ncol=3, fontsize=8)

        axes[1, 1].plot(time_s, columns["roll_deg"], color=BLUE, label="Roll")
        axes[1, 1].plot(time_s, columns["pitch_deg"], color=ORANGE, label="Pitch")
        axes[1, 1].plot(time_s, columns["turn_rate_degps"], color=GREEN, label="Turn rate")
        axes[1, 1].set_ylabel("Angle / rate [deg, deg/s]")
        axes[1, 1].set_title("Attitude and actual heading turn rate")
        axes[1, 1].legend(ncol=3, fontsize=8)

        axes[2, 0].plot(time_s, columns["aileron_deg"], color=BLUE, label="Aileron")
        axes[2, 0].plot(time_s, columns["elevator_deg"], color=ORANGE, label="Elevator")
        axes[2, 0].plot(time_s, columns["rudder_deg"], color=GREEN, label="Rudder")
        axes[2, 0].set_ylabel("Deflection [deg]")
        axes[2, 0].set_xlabel("Time [s]")
        axes[2, 0].set_title("Rate-limited actuator states")
        axes[2, 0].legend(ncol=3, fontsize=8)

        axes[2, 1].plot(time_s, columns["load_factor_g"], color=BLUE, label="Load factor")
        axes[2, 1].plot(time_s, columns["throttle"], color=GREEN, label="Throttle")
        axes[2, 1].plot(
            time_s,
            columns["mass_kg"] / simulation.configuration.mass.initial_mass_kg,
            color=ORANGE,
            label="Mass / initial",
        )
        axes[2, 1].set_ylabel("Normalized / g")
        axes[2, 1].set_xlabel("Time [s]")
        axes[2, 1].set_title("Loads, propulsion command, and mass")
        axes[2, 1].legend(ncol=3, fontsize=8)
        figure.suptitle(
            "Aquila-X1 nonlinear flight diagnostics | synthetic coefficients",
            color=NAVY,
            fontsize=13,
            fontweight="bold",
        )
        dashboard = output / "aircraft_flight_diagnostics.png"
        figure.savefig(dashboard, dpi=220)
        plt.close(figure)
    return path_3d, dashboard
