"""Professional flight-envelope and gain-schedule summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aerognc.gnc.flight_envelope import FlightEnvelopeResult
from aerognc.visualisation.style import engineering_style


def plot_flight_envelope(result: FlightEnvelopeResult, output_directory: str | Path) -> Path:
    """Plot trim, authority, gain evolution, and closed-loop pole migration."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    qbar_kpa = np.array(
        [item.operating_point.dynamic_pressure_pa / 1000.0 for item in result.analyses]
    )
    mach = np.array([item.operating_point.mach for item in result.analyses])
    alpha_deg = np.rad2deg([item.trim.decision[0] for item in result.analyses])
    command_deg = np.rad2deg([item.trim.decision[1] for item in result.analyses])
    authority_percent = np.array(
        [100.0 * item.control_authority_fraction for item in result.analyses]
    )
    angle_gain = np.array([item.lqr.gain[0, 0] for item in result.analyses])
    rate_gain = np.array([item.lqr.gain[0, 1] for item in result.analyses])
    poles = np.array(
        [mode.eigenvalue for item in result.analyses for mode in item.closed_loop_modes],
        dtype=np.complex128,
    )
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
        trim_axis, authority_axis, gain_axis, pole_axis = axes.ravel()
        alpha_scatter = trim_axis.scatter(
            qbar_kpa,
            alpha_deg,
            c=mach,
            cmap="viridis",
            edgecolor="white",
            linewidth=0.35,
            label="Angle of attack",
        )
        trim_axis.scatter(
            qbar_kpa,
            command_deg,
            c=mach,
            cmap="viridis",
            marker="x",
            label="Control command",
        )
        trim_axis.axhline(0.0, color="#65737E", linewidth=0.8)
        trim_axis.set(
            title="Nonlinear trim solutions",
            xlabel="Dynamic pressure [kPa]",
            ylabel="Angle [deg]",
        )
        trim_axis.legend(loc="best")
        figure.colorbar(alpha_scatter, ax=trim_axis, label="Mach number [-]")

        authority_scatter = authority_axis.scatter(
            qbar_kpa,
            authority_percent,
            c=mach,
            cmap="plasma",
            edgecolor="white",
            linewidth=0.35,
        )
        requirement_percent = 100.0 * result.configuration.minimum_control_authority_fraction
        authority_axis.axhline(
            requirement_percent,
            color="#C44E52",
            linestyle="--",
            label=f"Requirement ({requirement_percent:.0f}%)",
        )
        authority_axis.set(
            title="Remaining actuator authority",
            xlabel="Dynamic pressure [kPa]",
            ylabel="Unused position range [%]",
            ylim=(0.0, 105.0),
        )
        authority_axis.legend(loc="lower right")
        figure.colorbar(authority_scatter, ax=authority_axis, label="Mach number [-]")

        gain_axis.scatter(qbar_kpa, angle_gain, color="#2878B5", label="Angle gain")
        gain_axis.scatter(qbar_kpa, rate_gain, color="#E18727", label="Rate gain")
        gain_axis.set(
            title="Scheduled LQR gains",
            xlabel="Dynamic pressure [kPa]",
            ylabel="Feedback gain",
        )
        gain_axis.legend(loc="best")

        pole_axis.scatter(poles.real, poles.imag, marker="x", color="#C44E52", s=42)
        pole_axis.axvline(0.0, color="black", linewidth=0.9)
        pole_axis.set(
            title="Closed-loop pole migration",
            xlabel="Real [rad/s]",
            ylabel="Imaginary [rad/s]",
        )
        for axis in axes.ravel():
            axis.grid(True, alpha=0.26)
        figure.suptitle(
            "Asteria-SR1 synthetic ascent envelope and scheduled control",
            fontweight="bold",
        )
        path = output / "flight_envelope.png"
        figure.savefig(path, dpi=190)
        plt.close(figure)
    return path
