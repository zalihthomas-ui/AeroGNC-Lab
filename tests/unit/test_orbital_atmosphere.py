import numpy as np
import pytest

from aerognc.environment.orbital_atmosphere import ReferenceOrbitalAtmosphere


def test_reference_orbital_atmosphere_is_monotonic_above_100_km() -> None:
    atmosphere = ReferenceOrbitalAtmosphere()
    altitudes_m = np.array([100e3, 120e3, 200e3, 400e3, 800e3, 1_000e3])
    densities = np.array([atmosphere.density_kgpm3(value) for value in altitudes_m])

    assert np.all(densities > 0.0)
    assert np.all(np.diff(densities) < 0.0)
    assert atmosphere.density_kgpm3(200e3) == pytest.approx(2.789e-10)


def test_density_scale_and_bounded_tail_are_explicit() -> None:
    nominal = ReferenceOrbitalAtmosphere()
    doubled = ReferenceOrbitalAtmosphere(density_scale=2.0)

    assert doubled.density_kgpm3(300e3) == pytest.approx(2.0 * nominal.density_kgpm3(300e3))
    assert nominal.density_kgpm3(1_200e3) > 0.0
    assert nominal.density_kgpm3(1_600e3) == 0.0


def test_orbital_atmosphere_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="density_scale"):
        ReferenceOrbitalAtmosphere(density_scale=-1.0)
    with pytest.raises(ValueError, match="finite"):
        ReferenceOrbitalAtmosphere().properties(float("nan"))
