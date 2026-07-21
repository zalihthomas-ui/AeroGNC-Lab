"""Constrained ascent reference-versus-optimized engineering figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aerognc.simulation.guided_ascent import AscentGuidanceOptimizationResult
from aerognc.visualisation.style import engineering_style


def plot_ascent_guidance(
    optimization: AscentGuidanceOptimizationResult,
    output_directory: str | Path,
) -> Path:
    """Plot trajectory performance, constraints, and optimized commands."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    configuration = optimization.configuration
    reference = optimization.reference_run.result
    optimized = optimization.optimized_run.result
    burnout_time = configuration.base_scenario.vehicle.propulsion.burnout_time_s
    reference_powered = reference.time_s <= burnout_time
    optimized_powered = optimized.time_s <= burnout_time
    reference_alpha_active = reference_powered & (
        reference.columns["dynamic_pressure_pa"]
        >= configuration.minimum_alpha_constraint_dynamic_pressure_pa
    )
    optimized_alpha_active = optimized_powered & (
        optimized.columns["dynamic_pressure_pa"]
        >= configuration.minimum_alpha_constraint_dynamic_pressure_pa
    )
    with engineering_style():
        figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), constrained_layout=True)
        altitude_axis, trajectory_axis, q_axis, load_axis, alpha_axis, command_axis = axes.ravel()

        altitude_axis.plot(
            reference.time_s,
            reference.columns["altitude_m"],
            color="#A0A0A0",
            label="Reference only",
        )
        altitude_axis.plot(
            optimized.time_s,
            optimized.columns["altitude_m"],
            color="#2878B5",
            label="Optimized + governed",
        )
        altitude_axis.axhspan(
            configuration.desired_apogee_m - configuration.apogee_tolerance_m,
            configuration.desired_apogee_m + configuration.apogee_tolerance_m,
            color="#55A868",
            alpha=0.16,
            label="Apogee tolerance",
        )
        altitude_axis.set(title="Altitude performance", xlabel="Time [s]", ylabel="Altitude [m]")
        altitude_axis.legend(loc="best")

        trajectory_axis.plot(
            reference.columns["ground_range_m"],
            reference.columns["altitude_m"],
            color="#A0A0A0",
        )
        trajectory_axis.plot(
            optimized.columns["ground_range_m"],
            optimized.columns["altitude_m"],
            color="#2878B5",
        )
        trajectory_axis.set(
            title="Pitch-plane trajectory",
            xlabel="Ground range [m]",
            ylabel="Altitude [m]",
        )

        q_axis.plot(
            reference.time_s[reference_powered],
            reference.columns["dynamic_pressure_pa"][reference_powered] / 1000.0,
            color="#A0A0A0",
        )
        q_axis.plot(
            optimized.time_s[optimized_powered],
            optimized.columns["dynamic_pressure_pa"][optimized_powered] / 1000.0,
            color="#2878B5",
        )
        q_axis.axhline(
            configuration.maximum_dynamic_pressure_pa / 1000.0,
            color="#C44E52",
            linestyle="--",
            label="Limit",
        )
        q_axis.set(title="Powered-ascent max-Q", xlabel="Time [s]", ylabel="q [kPa]")
        q_axis.legend(loc="best")

        load_axis.plot(
            reference.time_s[reference_powered],
            reference.columns["proper_load_factor"][reference_powered],
            color="#A0A0A0",
        )
        load_axis.plot(
            optimized.time_s[optimized_powered],
            optimized.columns["proper_load_factor"][optimized_powered],
            color="#2878B5",
        )
        load_axis.axhline(
            configuration.maximum_proper_load_factor,
            color="#C44E52",
            linestyle="--",
            label="Limit",
        )
        load_axis.set(
            title="Powered-ascent proper load",
            xlabel="Time [s]",
            ylabel="Proper load [g0]",
        )
        load_axis.legend(loc="best")

        alpha_axis.plot(
            reference.time_s[reference_alpha_active],
            reference.columns["angle_of_attack_deg"][reference_alpha_active],
            color="#A0A0A0",
        )
        alpha_axis.plot(
            optimized.time_s[optimized_alpha_active],
            optimized.columns["angle_of_attack_deg"][optimized_alpha_active],
            color="#2878B5",
        )
        alpha_limit_deg = float(np.rad2deg(configuration.maximum_angle_of_attack_rad))
        alpha_axis.axhline(alpha_limit_deg, color="#C44E52", linestyle="--")
        alpha_axis.axhline(-alpha_limit_deg, color="#C44E52", linestyle="--")
        alpha_axis.set(
            title="Aerodynamically loaded ascent angle of attack",
            xlabel="Time [s]",
            ylabel="Angle of attack [deg]",
        )

        command_axis.plot(
            optimized.time_s[optimized_powered],
            optimized.columns["reference_throttle"][optimized_powered],
            color="#A0A0A0",
            linestyle="--",
            label="Reference throttle",
        )
        command_axis.plot(
            optimized.time_s[optimized_powered],
            optimized.columns["throttle"][optimized_powered],
            color="#E18727",
            label="Governed throttle",
        )
        command_axis.set(
            title="Online constraint governance",
            xlabel="Time [s]",
            ylabel="Throttle [-]",
            ylim=(-0.05, 1.05),
        )
        command_axis.legend(loc="best")

        for axis in axes.ravel():
            axis.grid(True, alpha=0.26)
        figure.suptitle(
            "Fictional civilian research-rocket constrained ascent guidance",
            fontweight="bold",
        )
        path = output / "constrained_ascent_guidance.png"
        figure.savefig(path, dpi=190)
        plt.close(figure)
    return path
