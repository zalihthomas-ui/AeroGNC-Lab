import numpy as np
import pytest

from aerognc.astrodynamics.maneuvers import (
    FiniteBurn,
    ImpulsiveManeuver,
    apply_impulsive_maneuver,
    available_delta_v_mps,
    finite_burn_derivative,
    ideal_propellant_used_kg,
    rtn_to_inertial_matrix,
)


def test_rtn_transform_and_impulse_mass_follow_closed_forms() -> None:
    transform = rtn_to_inertial_matrix([7.0e6, 0.0, 0.0], [0.0, 7_500.0, 0.0])
    np.testing.assert_allclose(transform, np.eye(3), atol=1.0e-14)
    maneuver = ImpulsiveManeuver("raise orbit", 10.0, (0.0, 100.0, 0.0), "rtn", 300.0)
    initial = np.array([7.0e6, 0.0, 0.0, 0.0, 7_500.0, 0.0, 1_000.0])

    final = apply_impulsive_maneuver(initial, maneuver, dry_mass_kg=800.0)

    assert final[4] == pytest.approx(7_600.0)
    assert initial[6] - final[6] == pytest.approx(ideal_propellant_used_kg(1_000.0, 100.0, 300.0))
    assert available_delta_v_mps(1_000.0, final[6], 300.0) == pytest.approx(100.0)


def test_finite_burn_has_expected_acceleration_and_mass_flow_only_when_active() -> None:
    burn = FiniteBurn("cruise trim", 5.0, 20.0, 40.0, (1.0, 0.0, 0.0), "inertial", 250.0)
    state = np.array([7.0e6, 0.0, 0.0, 0.0, 7_500.0, 0.0, 1_000.0])

    before = finite_burn_derivative(4.0, state, burn, 800.0)
    active = finite_burn_derivative(10.0, state, burn, 800.0)
    after = finite_burn_derivative(25.0, state, burn, 800.0)

    np.testing.assert_array_equal(before[0], np.zeros(3))
    np.testing.assert_allclose(active[0], [0.04, 0.0, 0.0])
    assert active[1] == pytest.approx(-burn.mass_flow_rate_kg_s)
    np.testing.assert_array_equal(after[0], np.zeros(3))


def test_impulse_rejects_dry_mass_violation() -> None:
    maneuver = ImpulsiveManeuver("too large", 0.0, (5_000.0, 0.0, 0.0), "inertial", 100.0)
    state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0])
    with pytest.raises(FloatingPointError, match="more propellant"):
        apply_impulsive_maneuver(state, maneuver, dry_mass_kg=90.0)
