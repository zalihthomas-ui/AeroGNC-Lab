import numpy as np
import pytest

from aerognc.vehicle.propulsion import ThrustCurve


def test_thrust_interpolation_impulse_and_bounds() -> None:
    curve = ThrustCurve([0.0, 1.0, 2.0], [0.0, 100.0, 0.0], 2.0)
    assert curve.total_impulse_ns == pytest.approx(100.0)
    assert curve.thrust_at_time_n(0.5) == pytest.approx(50.0)
    assert curve.thrust_at_time_n(-0.1) == 0.0
    assert curve.thrust_at_time_n(2.1) == 0.0
    assert curve.delivered_impulse_ns(0.5) == pytest.approx(12.5)
    assert curve.delivered_impulse_ns(1.5) == pytest.approx(87.5)


def test_propellant_depletion_is_monotonic_and_exact_at_burnout() -> None:
    curve = ThrustCurve([0.0, 0.5, 2.0], [0.0, 120.0, 0.0], 5.0)
    remaining = np.array([curve.propellant_remaining_kg(t) for t in np.linspace(-1.0, 3.0, 101)])
    assert np.all(np.diff(remaining) <= 1.0e-12)
    assert remaining.min() >= 0.0
    assert curve.propellant_remaining_kg(curve.burnout_time_s) == pytest.approx(0.0, abs=1e-12)


def test_invalid_curve_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        ThrustCurve([0.0, 1.0], [1.0, -1.0], 1.0)
