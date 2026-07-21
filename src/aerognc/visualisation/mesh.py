"""Bounded OBJ/STL triangle-mesh import for local 3D playback.

Imported geometry is visual only: it never changes mass or aerodynamic properties.
Those engineering inputs remain explicit in the validated aircraft configuration.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

MeshAxisConvention = Literal[
    "body_frd",
    "x_forward_z_up",
    "y_forward_z_up",
]
MESH_AXIS_CONVENTIONS: tuple[MeshAxisConvention, ...] = (
    "body_frd",
    "x_forward_z_up",
    "y_forward_z_up",
)
MAXIMUM_MESH_BYTES = 20_000_000
MAXIMUM_MESH_VERTICES = 100_000
MAXIMUM_MESH_TRIANGLES = 200_000
DEFAULT_LIVE_TRIANGLE_LIMIT = 8_000
MeshCenterMode = Literal["centroid", "bounds", "none"]


@dataclass(frozen=True, slots=True)
class MeshTransform:
    """Explicit visual-only mesh transform in forward-right-down body axes."""

    rotation_deg_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    center_mode: MeshCenterMode = "centroid"

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_deg_xyz, dtype=np.float64)
        if rotation.shape != (3,) or not np.all(np.isfinite(rotation)):
            raise ValueError("mesh XYZ rotation must contain three finite angles")
        if self.center_mode not in ("centroid", "bounds", "none"):
            raise ValueError("mesh center mode must be centroid, bounds, or none")


@dataclass(frozen=True, slots=True)
class MeshInspection:
    """Human-readable facts shown before a visual mesh is launched."""

    name: str
    source_path: Path
    vertex_count: int
    triangle_count: int
    live_triangle_count: int
    dimensions_body_m: tuple[float, float, float]
    degenerate_triangle_count: int
    axis_statement: str
    physics_boundary: str


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    """Validated triangle mesh with vertices in forward-right-down body axes."""

    name: str
    vertices_body: FloatArray
    triangle_indices: npt.NDArray[np.int64]
    source_path: Path
    origin_is_explicit: bool = False

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_body, dtype=np.float64)
        triangles = np.asarray(self.triangle_indices)
        if not self.name.strip() or vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("mesh must have a name and N-by-3 vertices")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise ValueError("mesh triangle indices must be M-by-3")
        if not 3 <= vertices.shape[0] <= MAXIMUM_MESH_VERTICES:
            raise ValueError("mesh vertex count is outside the supported range")
        if not 1 <= triangles.shape[0] <= MAXIMUM_MESH_TRIANGLES:
            raise ValueError("mesh triangle count is outside the supported range")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("mesh contains non-finite vertices")
        if not np.issubdtype(triangles.dtype, np.integer):
            raise ValueError("mesh triangle indices must be integers")
        if np.min(triangles) < 0 or np.max(triangles) >= vertices.shape[0]:
            raise ValueError("mesh triangle index is outside the vertex array")
        triangle_vertices = vertices[triangles.astype(np.int64)]
        twice_areas = np.linalg.norm(
            np.cross(
                triangle_vertices[:, 1] - triangle_vertices[:, 0],
                triangle_vertices[:, 2] - triangle_vertices[:, 0],
            ),
            axis=1,
        )
        if not np.any(twice_areas > 1.0e-14):
            raise ValueError("mesh contains no non-degenerate triangle")
        object.__setattr__(self, "vertices_body", vertices.copy())
        object.__setattr__(self, "triangle_indices", triangles.astype(np.int64, copy=True))

    @property
    def triangles_body(self) -> FloatArray:
        """Return M-by-3-by-3 triangle coordinates for Matplotlib collections."""
        return np.asarray(self.vertices_body[self.triangle_indices], dtype=np.float64)

    def centered_and_scaled(self, target_length: float = 12.0) -> TriangleMesh:
        """Return a centroid-centred copy uniformly scaled to the requested x length."""
        centered = self.transformed(MeshTransform(center_mode="centroid"))
        return centered.scaled_to_length(target_length)

    def scaled_to_length(self, target_length: float = 12.0) -> TriangleMesh:
        """Uniformly scale about the declared mesh origin without recentering it."""
        if not np.isfinite(target_length) or target_length <= 0.0:
            raise ValueError("target_length must be positive and finite")
        x_extent = float(np.ptp(self.vertices_body[:, 0]))
        overall_extent = float(np.max(np.ptp(self.vertices_body, axis=0)))
        reference_extent = x_extent if x_extent > 1.0e-9 else overall_extent
        if reference_extent <= 1.0e-12:
            raise ValueError("mesh has zero spatial extent")
        return TriangleMesh(
            self.name,
            self.vertices_body * (target_length / reference_extent),
            self.triangle_indices,
            self.source_path,
            self.origin_is_explicit,
        )

    def transformed(self, transform: MeshTransform) -> TriangleMesh:
        """Return a visual-only rotated/flipped/centered mesh copy."""
        vertices = self.vertices_body.copy()
        if transform.center_mode == "centroid":
            vertices -= np.mean(vertices, axis=0)
        elif transform.center_mode == "bounds":
            vertices -= 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
        flip = np.array(
            [
                -1.0 if transform.flip_x else 1.0,
                -1.0 if transform.flip_y else 1.0,
                -1.0 if transform.flip_z else 1.0,
            ]
        )
        vertices *= flip
        x_rad, y_rad, z_rad = np.deg2rad(transform.rotation_deg_xyz)
        cx, sx = np.cos(x_rad), np.sin(x_rad)
        cy, sy = np.cos(y_rad), np.sin(y_rad)
        cz, sz = np.cos(z_rad), np.sin(z_rad)
        rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        rotation = rotation_z @ rotation_y @ rotation_x
        return TriangleMesh(
            self.name,
            np.asarray(vertices @ rotation.T, dtype=np.float64),
            self.triangle_indices,
            self.source_path,
            True,
        )

    def decimated(self, maximum_triangles: int = DEFAULT_LIVE_TRIANGLE_LIMIT) -> TriangleMesh:
        """Deterministically retain evenly distributed source triangles for live display."""
        if isinstance(maximum_triangles, bool) or not (
            1 <= maximum_triangles <= MAXIMUM_MESH_TRIANGLES
        ):
            raise ValueError("maximum live triangle count is outside the supported range")
        if self.triangle_indices.shape[0] <= maximum_triangles:
            return TriangleMesh(
                self.name,
                self.vertices_body,
                self.triangle_indices,
                self.source_path,
                self.origin_is_explicit,
            )
        selected_indices = np.linspace(
            0,
            self.triangle_indices.shape[0] - 1,
            maximum_triangles,
            dtype=np.int64,
        )
        selected_triangles = self.triangle_indices[selected_indices]
        retained_vertices, inverse = np.unique(selected_triangles, return_inverse=True)
        compact_triangles = inverse.reshape(-1, 3).astype(np.int64)
        return TriangleMesh(
            self.name,
            self.vertices_body[retained_vertices],
            compact_triangles,
            self.source_path,
            self.origin_is_explicit,
        )

    def inspection(self, live_triangle_limit: int = DEFAULT_LIVE_TRIANGLE_LIMIT) -> MeshInspection:
        """Return dimensions, complexity, axes, and the engineering-property boundary."""
        if isinstance(live_triangle_limit, bool) or live_triangle_limit <= 0:
            raise ValueError("live triangle limit must be a positive integer")
        triangles = self.triangles_body
        twice_areas = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        dimensions = np.ptp(self.vertices_body, axis=0)
        return MeshInspection(
            self.name,
            self.source_path,
            int(self.vertices_body.shape[0]),
            int(self.triangle_indices.shape[0]),
            min(int(self.triangle_indices.shape[0]), live_triangle_limit),
            (float(dimensions[0]), float(dimensions[1]), float(dimensions[2])),
            int(np.count_nonzero(twice_areas <= 1.0e-14)),
            "+X nose/forward, +Y right wing, +Z down (FRD)",
            "Visual geometry does not set CL, CD, Cm, mass, inertia, or propulsion.",
        )


def _axis_transform(vertices: FloatArray, convention: MeshAxisConvention) -> FloatArray:
    if convention == "body_frd":
        return vertices.copy()
    if convention == "x_forward_z_up":
        # Common modelling axes: +X forward, +Y left, +Z up.
        return np.asarray(vertices @ np.diag([1.0, -1.0, -1.0]), dtype=np.float64)
    # Common +Y-forward, +X-right, +Z-up source -> +X-forward, +Y-right, +Z-down.
    return np.asarray(vertices[:, [1, 0, 2]] * np.array([1.0, 1.0, -1.0]), dtype=np.float64)


def _obj_mesh(path: Path, text: str, convention: MeshAxisConvention) -> TriangleMesh:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"OBJ line {line_number}: vertex needs three coordinates")
            try:
                vertex = (float(fields[1]), float(fields[2]), float(fields[3]))
            except ValueError as error:
                raise ValueError(f"OBJ line {line_number}: invalid vertex") from error
            vertices.append(vertex)
            if len(vertices) > MAXIMUM_MESH_VERTICES:
                raise ValueError("OBJ exceeds the supported vertex limit")
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ValueError(f"OBJ line {line_number}: face needs at least three vertices")
            face: list[int] = []
            for token in fields[1:]:
                index_text = token.split("/", maxsplit=1)[0]
                try:
                    source_index = int(index_text)
                except ValueError as error:
                    raise ValueError(f"OBJ line {line_number}: invalid face index") from error
                if source_index == 0:
                    raise ValueError(f"OBJ line {line_number}: index zero is invalid")
                index = source_index - 1 if source_index > 0 else len(vertices) + source_index
                if index < 0 or index >= len(vertices):
                    raise ValueError(f"OBJ line {line_number}: face references a missing vertex")
                face.append(index)
            for offset in range(1, len(face) - 1):
                triangles.append((face[0], face[offset], face[offset + 1]))
                if len(triangles) > MAXIMUM_MESH_TRIANGLES:
                    raise ValueError("OBJ exceeds the supported triangle limit")
    if not vertices or not triangles:
        raise ValueError("OBJ must contain vertices and faces")
    vertex_array = _axis_transform(np.asarray(vertices, dtype=np.float64), convention)
    return TriangleMesh(path.stem, vertex_array, np.asarray(triangles, dtype=np.int64), path)


def _binary_stl_mesh(
    path: Path, payload: bytes, triangle_count: int, convention: MeshAxisConvention
) -> TriangleMesh:
    vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
    triangles = np.arange(triangle_count * 3, dtype=np.int64).reshape(-1, 3)
    offset = 84
    for triangle_index in range(triangle_count):
        unpacked = struct.unpack_from("<12fH", payload, offset)
        vertices[3 * triangle_index : 3 * triangle_index + 3] = np.asarray(
            unpacked[3:12], dtype=np.float64
        ).reshape(3, 3)
        offset += 50
    return TriangleMesh(
        path.stem,
        _axis_transform(vertices, convention),
        triangles,
        path,
    )


def _ascii_stl_mesh(path: Path, text: str, convention: MeshAxisConvention) -> TriangleMesh:
    vertices: list[tuple[float, float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fields = raw_line.strip().split()
        if not fields or fields[0].casefold() != "vertex":
            continue
        if len(fields) != 4:
            raise ValueError(f"STL line {line_number}: vertex needs three coordinates")
        try:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError as error:
            raise ValueError(f"STL line {line_number}: invalid vertex") from error
        if len(vertices) > 3 * MAXIMUM_MESH_TRIANGLES:
            raise ValueError("STL exceeds the supported triangle limit")
    if not vertices or len(vertices) % 3 != 0:
        raise ValueError("ASCII STL must contain complete vertex triplets")
    vertex_array = _axis_transform(np.asarray(vertices, dtype=np.float64), convention)
    triangles = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return TriangleMesh(path.stem, vertex_array, triangles, path)


@lru_cache(maxsize=16)
def _cached_triangle_mesh(
    source: Path,
    axis_convention: MeshAxisConvention,
    source_size: int,
    source_modified_ns: int,
) -> TriangleMesh:
    """Parse one immutable source revision; public callers receive defensive copies."""
    del source_size, source_modified_ns
    suffix = source.suffix.casefold()
    payload = source.read_bytes()
    if suffix == ".obj":
        try:
            mesh = _obj_mesh(source, payload.decode("utf-8"), axis_convention)
        except UnicodeDecodeError as error:
            raise ValueError("OBJ must be UTF-8 text") from error
    elif suffix == ".stl":
        triangle_count = struct.unpack_from("<I", payload, 80)[0] if len(payload) >= 84 else -1
        expected_size = 84 + 50 * triangle_count
        if 0 <= triangle_count <= MAXIMUM_MESH_TRIANGLES and expected_size == len(payload):
            mesh = _binary_stl_mesh(source, payload, triangle_count, axis_convention)
        else:
            try:
                mesh = _ascii_stl_mesh(source, payload.decode("utf-8"), axis_convention)
            except UnicodeDecodeError as error:
                raise ValueError("STL is neither valid bounded binary nor UTF-8 ASCII") from error
    else:
        raise ValueError("supported 3D mesh formats are .obj and .stl")
    return mesh


def load_triangle_mesh(
    path: str | Path,
    *,
    axis_convention: MeshAxisConvention = "body_frd",
    target_length: float | None = None,
    transform: MeshTransform | None = None,
    live_triangle_limit: int | None = None,
) -> TriangleMesh:
    """Load/cache a bounded OBJ/STL, transform it explicitly, and return a safe copy."""
    source = Path(path).resolve()
    if axis_convention not in MESH_AXIS_CONVENTIONS:
        raise ValueError(f"axis_convention must be one of {MESH_AXIS_CONVENTIONS}")
    if not source.is_file():
        raise FileNotFoundError(f"3D mesh file not found: {source}")
    stat = source.stat()
    size = stat.st_size
    if size <= 0 or size > MAXIMUM_MESH_BYTES:
        raise ValueError(f"mesh file size must lie in (0, {MAXIMUM_MESH_BYTES}] bytes")
    cached = _cached_triangle_mesh(source, axis_convention, size, stat.st_mtime_ns)
    mesh = TriangleMesh(
        cached.name,
        cached.vertices_body,
        cached.triangle_indices,
        cached.source_path,
        cached.origin_is_explicit,
    )
    if transform is not None:
        mesh = mesh.transformed(transform)
    if target_length is not None:
        mesh = mesh.centered_and_scaled(target_length)
    if live_triangle_limit is not None:
        mesh = mesh.decimated(live_triangle_limit)
    return mesh


def transformed_triangles(
    mesh: TriangleMesh,
    dcm_display_body: npt.ArrayLike,
    origin_display: npt.ArrayLike,
) -> FloatArray:
    """Rotate/translate mesh triangles for a 3D display coordinate system."""
    dcm = np.asarray(dcm_display_body, dtype=np.float64)
    origin = np.asarray(origin_display, dtype=np.float64)
    if dcm.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("mesh display transform requires a 3x3 DCM and three-vector origin")
    if not np.all(np.isfinite(dcm)) or not np.all(np.isfinite(origin)):
        raise ValueError("mesh display transform must be finite")
    return mesh.triangles_body @ dcm.T + origin
