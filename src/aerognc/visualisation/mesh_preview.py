"""Preflight preview for imported visual-only aircraft meshes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerognc-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

from aerognc.visualisation.mesh import DEFAULT_LIVE_TRIANGLE_LIMIT, TriangleMesh
from aerognc.visualisation.style import BLUE, GREEN, NAVY, RED, engineering_style


def create_mesh_preview(
    mesh: TriangleMesh,
    *,
    live_triangle_limit: int = DEFAULT_LIVE_TRIANGLE_LIMIT,
) -> Figure:
    """Create a non-destructive axis/dimension/complexity preview figure."""
    inspection = mesh.inspection(live_triangle_limit)
    render_mesh = mesh.decimated(live_triangle_limit)
    with engineering_style():
        figure = plt.figure(figsize=(10.8, 6.8))
        scene = figure.add_subplot(1, 2, 1, projection="3d")
        if not isinstance(scene, Axes3D):
            raise RuntimeError("Matplotlib did not create a 3D mesh-preview axis")
        information = figure.add_subplot(1, 2, 2)
        collection = Poly3DCollection(
            render_mesh.triangles_body,
            facecolor="#38A6D8",
            edgecolor="#14384D",
            linewidth=0.35,
            alpha=0.92,
        )
        scene.add_collection3d(collection)
        minimum = np.min(render_mesh.vertices_body, axis=0)
        maximum = np.max(render_mesh.vertices_body, axis=0)
        centre = 0.5 * (minimum + maximum)
        span = max(float(np.max(maximum - minimum)), 1.0)
        for axis in range(3):
            low, high = centre[axis] - 0.62 * span, centre[axis] + 0.62 * span
            (scene.set_xlim, scene.set_ylim, scene.set_zlim)[axis](low, high)
        axis_length = 0.42 * span
        scene.quiver(0.0, 0.0, 0.0, axis_length, 0.0, 0.0, color=RED, linewidth=2.0)
        scene.quiver(0.0, 0.0, 0.0, 0.0, axis_length, 0.0, color=GREEN, linewidth=2.0)
        scene.quiver(0.0, 0.0, 0.0, 0.0, 0.0, axis_length, color=BLUE, linewidth=2.0)
        scene.text(axis_length, 0.0, 0.0, "+X NOSE", color=RED, fontsize=8)
        scene.text(0.0, axis_length, 0.0, "+Y RIGHT", color=GREEN, fontsize=8)
        scene.text(0.0, 0.0, axis_length, "+Z DOWN", color=BLUE, fontsize=8)
        scene.set_xlabel("Body X [visual units]")
        scene.set_ylabel("Body Y [visual units]")
        scene.set_zlabel("Body Z [visual units]")
        scene.set_title("Imported geometry and FRD axes")
        scene.view_init(elev=23.0, azim=-56.0)
        information.axis("off")
        dimensions = inspection.dimensions_body_m
        information.text(
            0.0,
            1.0,
            "MESH PREFLIGHT\n\n"
            f"Name              {inspection.name}\n"
            f"Vertices          {inspection.vertex_count:,}\n"
            f"Source triangles  {inspection.triangle_count:,}\n"
            f"Live triangles    {inspection.live_triangle_count:,}\n"
            f"Dimensions X/Y/Z  {dimensions[0]:.3g} / {dimensions[1]:.3g} / "
            f"{dimensions[2]:.3g}\n"
            f"Degenerate tris   {inspection.degenerate_triangle_count:,}\n\n"
            f"AXES\n{inspection.axis_statement}\n\n"
            f"ENGINEERING BOUNDARY\n{inspection.physics_boundary}\n\n"
            "Large meshes are deterministically decimated for live rendering. "
            "The source file is not modified.",
            transform=information.transAxes,
            va="top",
            color=NAVY,
            family="monospace",
            fontsize=9.2,
            wrap=True,
        )
        figure.suptitle("Aircraft visual model preview", color=NAVY, fontweight="bold")
        figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    return figure


def show_mesh_preview(mesh: TriangleMesh, *, block: bool = False) -> Figure:
    """Open a preflight preview while allowing the setup window to remain responsive."""
    figure = create_mesh_preview(mesh)
    plt.show(block=block)
    return figure
