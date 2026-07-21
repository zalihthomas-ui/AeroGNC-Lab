import numpy as np
import pytest

from aerognc.mathematics.vectors import as_vector, skew_symmetric


def test_skew_symmetric_reproduces_cross_product() -> None:
    left = np.array([1.2, -3.4, 5.6])
    right = np.array([-0.3, 0.8, 2.1])
    np.testing.assert_allclose(skew_symmetric(left) @ right, np.cross(left, right))
    np.testing.assert_allclose(skew_symmetric(left).T, -skew_symmetric(left))


@pytest.mark.parametrize("value", [[1.0, 2.0], [[1.0, 2.0, 3.0]], [1.0, np.nan, 3.0]])
def test_vector_validation_rejects_wrong_shape_or_nonfinite(value: list[object]) -> None:
    with pytest.raises(ValueError):
        as_vector(value, 3, name="test")
