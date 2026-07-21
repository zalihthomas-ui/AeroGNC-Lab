import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerognc.visualisation.mesh import (
    MeshTransform,
    load_triangle_mesh,
    transformed_triangles,
)
from aerognc.visualisation.mesh_preview import create_mesh_preview


def test_example_aircraft_obj_loads_and_scales_in_body_axes() -> None:
    mesh = load_triangle_mesh("assets/models/aquila_x1.obj", target_length=12.0)

    assert mesh.vertices_body.shape == (38, 3)
    assert mesh.triangle_indices.shape[0] >= 40
    assert np.ptp(mesh.vertices_body[:, 0]) == pytest.approx(12.0)
    assert np.max(mesh.vertices_body[:, 1]) > 5.0


def test_obj_polygon_is_triangulated_and_common_z_up_axes_are_converted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quad.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 1\nv 0 1 1\nf 1 2 3 4\n",
        encoding="utf-8",
    )
    mesh = load_triangle_mesh(path, axis_convention="x_forward_z_up")

    assert mesh.triangle_indices.shape == (2, 3)
    np.testing.assert_allclose(mesh.vertices_body[2], [1.0, -1.0, -1.0])


def test_ascii_and_binary_stl_are_supported(tmp_path: Path) -> None:
    ascii_path = tmp_path / "triangle_ascii.stl"
    ascii_path.write_text(
        "solid demo\nfacet normal 0 0 1\nouter loop\n"
        "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid demo\n",
        encoding="utf-8",
    )
    binary_path = tmp_path / "triangle_binary.stl"
    header = bytes(80) + struct.pack("<I", 1)
    triangle = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )
    binary_path.write_bytes(header + triangle)

    assert load_triangle_mesh(ascii_path).triangle_indices.shape == (1, 3)
    assert load_triangle_mesh(binary_path).triangle_indices.shape == (1, 3)


def test_mesh_transform_and_invalid_extension_fail_clearly(tmp_path: Path) -> None:
    mesh = load_triangle_mesh("assets/models/aquila_x1.obj")
    transformed = transformed_triangles(mesh, np.eye(3), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(transformed[0, 0], mesh.triangles_body[0, 0] + [1.0, 2.0, 3.0])

    unsupported = tmp_path / "mesh.ply"
    unsupported.write_text("ply", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.obj and \.stl"):
        load_triangle_mesh(unsupported)


def test_mesh_visual_transform_inspection_and_decimation_are_deterministic() -> None:
    source = load_triangle_mesh("assets/models/aquila_x1.obj")
    transformed = source.transformed(
        MeshTransform(rotation_deg_xyz=(0.0, 0.0, 90.0), flip_y=True, center_mode="bounds")
    )
    np.testing.assert_allclose(
        0.5
        * (
            np.min(transformed.vertices_body, axis=0)
            + np.max(transformed.vertices_body, axis=0)
        ),
        np.zeros(3),
        atol=1.0e-12,
    )
    first = transformed.decimated(5)
    repeated = transformed.decimated(5)
    assert first.triangle_indices.shape == (5, 3)
    np.testing.assert_array_equal(first.vertices_body, repeated.vertices_body)
    np.testing.assert_array_equal(first.triangle_indices, repeated.triangle_indices)
    inspection = source.inspection(5)
    assert inspection.live_triangle_count == 5
    assert inspection.physics_boundary.startswith("Visual geometry")


def test_mesh_preview_contains_geometry_and_engineering_boundary() -> None:
    mesh = load_triangle_mesh("assets/models/aquila_x1.obj")
    figure = create_mesh_preview(mesh, live_triangle_limit=10)
    try:
        assert len(figure.axes) == 2
        assert "MESH PREFLIGHT" in figure.axes[1].texts[0].get_text()
        assert "does not set CL" in figure.axes[1].texts[0].get_text()
    finally:
        plt.close(figure)
