import numpy as np
import pytest

from aerognc.environment.gravity import GravityModel
from aerognc.environment.wind import OneCosineGust, WindModel, WindProfile


def test_gravity_modes_and_ned_sign() -> None:
    constant = GravityModel("constant", 9.8)
    inverse = GravityModel("inverse_square", 9.8)
    np.testing.assert_allclose(constant.acceleration_ned_mps2(50_000.0), [0.0, 0.0, 9.8])
    assert 0.0 < inverse.magnitude_mps2(50_000.0) < 9.8
    with pytest.raises(ValueError):
        GravityModel("flat")  # type: ignore[arg-type]


def test_wind_profile_interpolates_and_clamps() -> None:
    profile = WindProfile([0.0, 100.0], [[1.0, 0.0, 0.0], [3.0, 2.0, 0.0]])
    np.testing.assert_allclose(profile.velocity_ned_mps(50.0), [2.0, 1.0, 0.0])
    np.testing.assert_allclose(profile.velocity_ned_mps(-10.0), [1.0, 0.0, 0.0])


def test_seeded_stochastic_wind_is_query_order_independent_and_reproducible() -> None:
    profile = WindProfile.constant([2.0, -1.0, 0.0])
    first = WindModel(profile, gust_std_ned_mps=[1.0, 1.0, 0.2], seed=42, horizon_s=10.0)
    second = WindModel(profile, gust_std_ned_mps=[1.0, 1.0, 0.2], seed=42, horizon_s=10.0)
    at_five = first.velocity_ned_mps(5.0, 200.0)
    first.velocity_ned_mps(2.0, 100.0)
    np.testing.assert_allclose(first.velocity_ned_mps(5.0, 200.0), at_five)
    np.testing.assert_allclose(second.velocity_ned_mps(5.0, 200.0), at_five)


def test_one_cosine_gust_is_smooth_bounded_pulse() -> None:
    gust = OneCosineGust(2.0, 4.0, [0.0, 8.0, -2.0])
    np.testing.assert_array_equal(gust.velocity_ned_mps(1.0), np.zeros(3))
    np.testing.assert_allclose(gust.velocity_ned_mps(2.0), np.zeros(3), atol=1.0e-15)
    np.testing.assert_allclose(gust.velocity_ned_mps(4.0), [0.0, 8.0, -2.0])
    np.testing.assert_allclose(gust.velocity_ned_mps(6.0), np.zeros(3), atol=1.0e-15)
    np.testing.assert_array_equal(gust.velocity_ned_mps(7.0), np.zeros(3))
