"""Publication-style overview of a fictional orbit-assisted planetary tour."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerognc.simulation.orbit_assisted_tour import OrbitTourSimulation
from aerognc.visualisation.style import BLUE, GREEN, GREY, NAVY, ORANGE, RED, engineering_style


def plot_orbit_assisted_tour(
    simulation: OrbitTourSimulation,
    output_directory: str | Path,
) -> Path:
    """Plot heliocentric legs, parking dwell, speed history, and burn/mass budget."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = simulation.result
    tour = simulation.tour
    columns = result.columns
    phase = columns["phase_code"].astype(np.int64)
    position_gm = (
        np.column_stack((columns["position_x_m"], columns["position_y_m"], columns["position_z_m"]))
        / 1.0e9
    )
    time_days = result.time_s / 86_400.0
    burn_names = [
        "Departure\ninjection",
        "Assist\ncapture",
        "Orbit\nalignment",
        "Assist\ndeparture",
        "Destination\ncapture",
    ]
    with engineering_style():
        figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
        axes[0, 0].plot(
            position_gm[phase == 0, 0],
            position_gm[phase == 0, 1],
            color=BLUE,
            label=f"{tour.departure_body.name} → {tour.assist_body.name}",
        )
        axes[0, 0].plot(
            position_gm[phase == 2, 0],
            position_gm[phase == 2, 1],
            color=ORANGE,
            label=f"{tour.assist_body.name} → {tour.destination_body.name}",
        )
        axes[0, 0].scatter(0.0, 0.0, s=90, color="#FDB813", edgecolor=NAVY, label="Helios")
        for body, marker, color in (
            (tour.departure_body, "o", BLUE),
            (tour.assist_body, "s", GREEN),
            (tour.destination_body, "D", ORANGE),
        ):
            body_position, _body_velocity = body.state_at_time(
                tour.departure_time_s
                if body is tour.departure_body
                else (
                    tour.assist_arrival_time_s
                    if body is tour.assist_body
                    else tour.destination_arrival_time_s
                ),
                simulation.configuration.catalog.primary.gravitational_parameter_m3_s2,
            )
            axes[0, 0].scatter(
                body_position[0] / 1.0e9,
                body_position[1] / 1.0e9,
                marker=marker,
                s=55,
                color=color,
            )
        axes[0, 0].set(
            xlabel="Inertial x [Gm]",
            ylabel="Inertial y [Gm]",
            title="Two Lambert legs and orbit-assist patch",
            aspect="equal",
        )
        axes[0, 0].legend(fontsize=8)

        orbit = phase == 1
        assist_x_m = columns[f"{tour.assist_body.name.casefold()}_position_x_m"][orbit]
        assist_y_m = columns[f"{tour.assist_body.name.casefold()}_position_y_m"][orbit]
        relative_x = (columns["position_x_m"][orbit] - assist_x_m) / tour.assist_parking_radius_m
        relative_y = (columns["position_y_m"][orbit] - assist_y_m) / tour.assist_parking_radius_m
        axes[0, 1].plot(relative_x, relative_y, color=GREEN, lw=1.8)
        axes[0, 1].scatter(0.0, 0.0, s=200, color=tour.assist_body.color, edgecolor=NAVY)
        axes[0, 1].scatter(relative_x[0], relative_y[0], color=BLUE, s=45, label="Capture")
        midpoint = relative_x.size // 2
        axes[0, 1].scatter(
            relative_x[midpoint],
            relative_y[midpoint],
            color=RED,
            marker="x",
            s=55,
            label="Alignment burn",
        )
        axes[0, 1].set(
            xlabel="Body-relative x / parking radius [-]",
            ylabel="Body-relative y / parking radius [-]",
            title=(
                f"{tour.dwell_revolutions} parking revolutions at "
                f"{simulation.configuration.assist_parking_altitude_m / 1_000.0:.0f} km"
            ),
            aspect="equal",
        )
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].plot(
            time_days,
            columns["heliocentric_speed_mps"] / 1_000.0,
            color=NAVY,
            label="Spacecraft speed",
        )
        for event in result.events:
            if "soi" not in event.name:
                axes[1, 0].axvline(event.time_s / 86_400.0, color=GREY, lw=0.55, alpha=0.5)
        axes[1, 0].axvspan(
            tour.assist_arrival_time_s / 86_400.0,
            tour.assist_departure_time_s / 86_400.0,
            color=GREEN,
            alpha=0.18,
            label="Parking-orbit dwell",
        )
        axes[1, 0].set(
            xlabel="Synthetic catalog epoch [day]",
            ylabel="Primary-relative speed [km/s]",
            title=(
                "Powered departure at periapsis: "
                f"Δε = {tour.departure_oberth_energy_gain_jpkg / 1.0e6:.1f} MJ/kg"
            ),
        )
        axes[1, 0].legend(fontsize=8)

        burn_delta_v = np.array([burn.delta_v_mps for burn in tour.burns]) / 1_000.0
        positions = np.arange(len(tour.burns))
        axes[1, 1].bar(positions, burn_delta_v, color=(BLUE, GREEN, RED, ORANGE, NAVY))
        for index, value in enumerate(burn_delta_v):
            axes[1, 1].text(index, value + 0.25, f"{value:.2f}", ha="center", fontsize=8)
        axes[1, 1].set_xticks(positions, burn_names)
        axes[1, 1].set(
            ylabel="Ideal impulsive Δv [km/s]",
            title=(
                f"Budget: {tour.total_delta_v_mps / 1_000.0:.2f} km/s; "
                f"final mass {tour.final_mass_kg / 1_000.0:.2f} t"
            ),
        )
        mass_axis = axes[1, 1].twinx()
        mass_axis.plot(
            positions,
            [burn.mass_after_kg / 1_000.0 for burn in tour.burns],
            color=NAVY,
            marker="o",
            ls="--",
        )
        mass_axis.set_ylabel("Mass after burn [t]", color=NAVY)
        figure.suptitle(
            "Fictional civilian capture-dwell-departure planetary tour",
            color=NAVY,
            fontweight="bold",
        )
        path = output / "orbit_assisted_tour.png"
        figure.savefig(path, dpi=220, pad_inches=0.12)
        plt.close(figure)
    return path
