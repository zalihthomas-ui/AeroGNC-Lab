import pytest

from aerognc.environment.atmosphere import (
    StandardAtmosphere1976,
    dynamic_pressure_pa,
    mach_number,
)


def test_standard_sea_level_reference_values() -> None:
    state = StandardAtmosphere1976().properties(0.0)
    assert state.temperature_k == pytest.approx(288.15, rel=1.0e-12)
    assert state.pressure_pa == pytest.approx(101_325.0, rel=1.0e-12)
    assert state.density_kgpm3 == pytest.approx(1.2250, rel=1.0e-4)
    assert state.speed_of_sound_mps == pytest.approx(340.294, rel=1.0e-5)


def test_troposphere_and_stratosphere_values_are_physical() -> None:
    atmosphere = StandardAtmosphere1976()
    lower = atmosphere.properties(10_000.0)
    upper = atmosphere.properties(20_000.0)
    assert 220.0 < lower.temperature_k < 225.0
    assert 26_000.0 < lower.pressure_pa < 27_000.0
    assert upper.pressure_pa < lower.pressure_pa
    assert upper.density_kgpm3 < lower.density_kgpm3


def test_atmosphere_range_and_utility_validation() -> None:
    atmosphere = StandardAtmosphere1976(-100.0, 1_000.0)
    with pytest.raises(ValueError, match="outside configured"):
        atmosphere.properties(1_001.0)
    assert mach_number(170.0, 340.0) == 0.5
    assert dynamic_pressure_pa(1.2, 10.0) == 60.0
    with pytest.raises(ValueError):
        mach_number(-1.0, 340.0)
    with pytest.raises(ValueError):
        dynamic_pressure_pa(-1.0, 3.0)
