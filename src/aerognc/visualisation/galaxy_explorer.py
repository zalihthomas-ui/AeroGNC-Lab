"""Interactive three-dimensional explorer for the observational planet snapshot."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import PickEvent
from matplotlib.figure import Figure

from aerognc.catalogs import ConfirmedExoplanet


@dataclass(frozen=True, slots=True)
class ExoplanetHostPoint:
    """One catalog host position and the selected planets reported around it."""

    host_name: str
    position_pc: tuple[float, float, float]
    planet_names: tuple[str, ...]


def build_exoplanet_host_points(
    selection: tuple[ConfirmedExoplanet, ...],
) -> tuple[ExoplanetHostPoint, ...]:
    """Collapse colocated planet rows into deterministic host-system points."""
    grouped: dict[str, list[ConfirmedExoplanet]] = {}
    for planet in selection:
        if planet.has_3d_position:
            grouped.setdefault(planet.host_name, []).append(planet)
    points: list[ExoplanetHostPoint] = []
    for host_name, planets in grouped.items():
        position = planets[0].galactic_position_pc()
        points.append(
            ExoplanetHostPoint(
                host_name=host_name,
                position_pc=(float(position[0]), float(position[1]), float(position[2])),
                planet_names=tuple(planet.name for planet in planets),
            )
        )
    return tuple(points)


class GalaxyExplorer3D:
    """Rotatable, pickable heliocentric view of selected confirmed exoplanet hosts."""

    def __init__(self, selection: tuple[ConfirmedExoplanet, ...]) -> None:
        self.points = build_exoplanet_host_points(selection)
        if not self.points:
            raise ValueError("the current selection has no rows with complete 3D coordinates")
        self.figure: Figure = plt.figure(figsize=(12.0, 8.0), facecolor="#07111F")
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.02, right=0.78, top=0.91, bottom=0.08)
        positions = np.asarray([point.position_pc for point in self.points], dtype=np.float64)
        sizes = np.asarray(
            [34.0 + 10.0 * min(len(point.planet_names), 7) for point in self.points],
            dtype=np.float64,
        )
        self.scatter = self.axes.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=np.linalg.norm(positions, axis=1),
            cmap="viridis_r",
            s=sizes,
            alpha=0.82,
            edgecolors="#D8F4FF",
            linewidths=0.35,
            depthshade=True,
            picker=True,
        )
        self.axes.scatter(
            [0.0],
            [0.0],
            [0.0],
            marker="*",
            s=260,
            c="#FFD166",
            edgecolors="#FFF4BF",
            linewidths=0.9,
            label="Sun",
        )
        self._style_axes(positions)
        self.info_text = self.figure.text(
            0.80,
            0.80,
            "SELECT A HOST\n\nClick a catalog point\nto inspect its system.",
            color="#D9E8F2",
            fontsize=10,
            va="top",
            family="DejaVu Sans Mono",
        )
        self.figure.text(
            0.80,
            0.48,
            "HOW TO USE\n\nDrag: rotate\nScroll: zoom\nClick: inspect\nToolbar: pan/save",
            color="#8FA6B8",
            fontsize=9,
            va="top",
        )
        self.figure.text(
            0.80,
            0.22,
            "OBSERVATIONAL CONTEXT\n\n"
            "This is a heliocentric catalog map,\n"
            "not a complete Milky Way model.\n"
            "Detection and reporting selection\n"
            "effects are substantial.",
            color="#5FD19A",
            fontsize=8.5,
            va="top",
            wrap=True,
        )
        self.figure.canvas.mpl_connect("pick_event", self._on_pick)

    def _style_axes(self, positions: np.ndarray) -> None:
        axes = self.axes
        axes.set_facecolor("#07111F")
        axes.set_xlabel("Galactic X toward centre [pc]", color="#BFD4E1", labelpad=10)
        axes.set_ylabel("Galactic Y [pc]", color="#BFD4E1", labelpad=10)
        axes.set_zlabel("Galactic Z [pc]", color="#BFD4E1", labelpad=10)
        axes.tick_params(colors="#8FA6B8", labelsize=8)
        axes.set_title(
            f"Confirmed-exoplanet host systems - {len(self.points):,} positioned hosts",
            color="#F1F7FA",
            fontsize=15,
            pad=18,
        )
        axes.legend(loc="upper left", facecolor="#102335", labelcolor="#FFFFFF")
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.pane.set_facecolor((0.05, 0.11, 0.16, 1.0))
            axis.pane.set_edgecolor((0.18, 0.35, 0.45, 0.7))
            axis._axinfo["grid"]["color"] = (0.35, 0.50, 0.58, 0.17)
        absolute_limit = float(np.max(np.abs(positions)))
        limit = max(1.0, absolute_limit * 1.06)
        axes.set_xlim(-limit, limit)
        axes.set_ylim(-limit, limit)
        axes.set_zlim(-limit, limit)
        axes.set_box_aspect((1.0, 1.0, 0.72))
        axes.view_init(elev=24.0, azim=-52.0)

    def _on_pick(self, event: PickEvent) -> None:
        indices: list[int] = getattr(event, "ind", [])
        if event.artist is not self.scatter or not indices:
            return
        point = self.points[int(indices[0])]
        distance = float(np.linalg.norm(point.position_pc))
        names = ", ".join(point.planet_names[:6])
        if len(point.planet_names) > 6:
            names += f", +{len(point.planet_names) - 6} more"
        self.info_text.set_text(
            "SELECTED HOST\n\n"
            f"{point.host_name}\n\n"
            f"Distance: {distance:.3f} pc\n"
            f"Selected planets: {len(point.planet_names)}\n\n"
            f"{names}"
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        """Open the explorer in a desktop Matplotlib window."""
        backend = str(matplotlib.get_backend()).lower()
        if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
            raise RuntimeError(
                f"Matplotlib backend {backend!r} cannot open the galaxy explorer; "
                "run it from a desktop session"
            )
        plt.show()

    def close(self) -> None:
        """Close the explorer figure."""
        plt.close(self.figure)


def explore_exoplanet_catalog(selection: tuple[ConfirmedExoplanet, ...]) -> None:
    """Open the selected observational catalog rows as a 3D host-system map."""
    GalaxyExplorer3D(selection).show()
