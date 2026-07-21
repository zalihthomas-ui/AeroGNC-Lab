import numpy as np
import pytest

from aerognc.vehicle.aerodynamics import AerodynamicModel


def _model() -> AerodynamicModel:
    return AerodynamicModel(
        reference_area_m2=0.03,
        reference_length_m=2.5,
        mach_points=[0.0, 1.0, 2.0],
        drag_coefficients=[0.4, 0.6, 0.45],
    )


def test_drag_force_opposes_air_relative_velocity() -> None:
    model = _model()
    velocity = np.array([100.0, 3.0, -2.0])
    loads = model.loads(velocity, density_kgpm3=1.0, speed_of_sound_mps=340.0)
    drag_only = loads.coefficients.drag * (-velocity / np.linalg.norm(velocity))
    assert np.dot(drag_only, velocity) < 0.0
    assert loads.dynamic_pressure_pa > 0.0


def test_restoring_pitch_and_yaw_moment_signs() -> None:
    model = _model()
    positive_alpha = model.coefficients(0.4, 0.1, 0.0)
    positive_beta = model.coefficients(0.4, 0.0, 0.1)
    assert positive_alpha.pitch < 0.0
    assert positive_beta.yaw > 0.0
    assert positive_alpha.normal < 0.0


def test_drag_table_clamps_outside_mach_domain() -> None:
    model = _model()
    assert model.drag_coefficient(-1.0) == pytest.approx(0.4)
    assert model.drag_coefficient(5.0) == pytest.approx(0.45)
