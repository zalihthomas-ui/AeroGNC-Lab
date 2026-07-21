import numpy as np
import pytest

from aerognc.vehicle.mass_properties import MassPropertiesModel
from aerognc.vehicle.propulsion import ThrustCurve


def _model() -> MassPropertiesModel:
    propulsion = ThrustCurve([0.0, 1.0, 2.0], [0.0, 100.0, 0.0], 2.0)
    return MassPropertiesModel(
        8.0,
        propulsion,
        1.1,
        1.4,
        np.diag([0.2, 3.0, 3.0]),
        np.diag([0.3, 4.0, 4.0]),
    )


def test_mass_cg_and_inertia_endpoints() -> None:
    model = _model()
    wet = model.at_time(0.0)
    dry = model.at_time(2.0)
    assert wet.mass_kg == 10.0
    assert dry.mass_kg == model.dry_mass_kg
    assert wet.centre_of_gravity_from_nose_m == 1.4
    assert dry.centre_of_gravity_from_nose_m == 1.1
    np.testing.assert_allclose(wet.inertia_body_kgm2, np.diag([0.3, 4.0, 4.0]))
    np.testing.assert_allclose(dry.inertia_body_kgm2, np.diag([0.2, 3.0, 3.0]))


def test_inertia_stays_positive_and_mass_never_below_dry() -> None:
    model = _model()
    for time_s in np.linspace(-2.0, 5.0, 50):
        properties = model.at_time(float(time_s))
        assert properties.mass_kg >= model.dry_mass_kg
        assert np.all(np.linalg.eigvalsh(properties.inertia_body_kgm2) > 0.0)


def test_non_positive_definite_inertia_rejected() -> None:
    propulsion = ThrustCurve([0.0, 1.0], [1.0, 1.0], 1.0)
    with pytest.raises(ValueError, match="positive definite"):
        MassPropertiesModel(1.0, propulsion, 1.0, 1.0, np.diag([1.0, 0.0, 1.0]), np.eye(3))
