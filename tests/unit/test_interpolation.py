import numpy as np
import pytest

from aerognc.mathematics.interpolation import (
    BilinearTable2D,
    LinearTable1D,
    RegularGridTableND,
)


def test_linear_table_policies() -> None:
    error_table = LinearTable1D([0.0, 1.0], [2.0, 4.0])
    assert error_table(0.25) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="outside"):
        error_table(2.0)
    assert LinearTable1D([0.0, 1.0], [2.0, 4.0], "clamp")(-1.0) == 2.0
    assert LinearTable1D([0.0, 1.0], [2.0, 4.0], "extrapolate")(2.0) == 6.0


def test_bilinear_table_exact_for_affine_plane() -> None:
    x = np.array([0.0, 1.0, 3.0])
    y = np.array([-2.0, 2.0])
    values = 3.0 * x[:, None] - 2.0 * y[None, :] + 5.0
    table = BilinearTable2D(x, y, values)
    assert table(0.4, -0.7) == pytest.approx(3.0 * 0.4 - 2.0 * -0.7 + 5.0)


def test_tables_reject_nonmonotonic_axes() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        LinearTable1D([0.0, 0.0], [1.0, 2.0])


def test_regular_grid_is_exact_for_multiaffine_function_and_gradient() -> None:
    x = np.array([-1.0, 0.5, 2.0])
    y = np.array([0.0, 4.0])
    z = np.array([-3.0, 1.0, 5.0])
    values = (
        2.0
        + 3.0 * x[:, None, None]
        - 2.0 * y[None, :, None]
        + 0.5 * z[None, None, :]
        + 0.2 * x[:, None, None] * y[None, :, None]
    )
    table = RegularGridTableND((x, y, z), values)
    query = (0.1, 2.5, -0.4)
    expected = 2.0 + 3.0 * query[0] - 2.0 * query[1] + 0.5 * query[2] + 0.2 * query[0] * query[1]
    value, gradient = table.value_and_gradient(*query)
    assert value == pytest.approx(expected)
    assert gradient == pytest.approx([3.0 + 0.2 * query[1], -2.0 + 0.2 * query[0], 0.5])


def test_regular_grid_boundary_policies_and_diagnostics() -> None:
    axes = (np.array([0.0, 1.0]), np.array([-1.0, 1.0]))
    values = np.array([[0.0, 2.0], [1.0, 3.0]])
    with pytest.raises(ValueError, match="outside"):
        RegularGridTableND(axes, values)(2.0, 0.0)
    clamped = RegularGridTableND(axes, values, "clamp")
    value, gradient = clamped.value_and_gradient(2.0, 0.0)
    assert value == pytest.approx(2.0)
    assert gradient == pytest.approx([0.0, 1.0])
    assert clamped.outside_axes(2.0, 0.0) == (0,)
    extrapolated = RegularGridTableND(axes, values, "extrapolate")
    assert extrapolated(2.0, 0.0) == pytest.approx(3.0)
