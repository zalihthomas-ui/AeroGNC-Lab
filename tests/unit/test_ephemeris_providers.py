import numpy as np
import pytest

from aerognc.astrodynamics.bodies import CircularOrbitBody
from aerognc.astrodynamics.ephemeris import EphemerisUnavailableError
from aerognc.astrodynamics.ephemeris_provider import (
    AnalyticalEphemerisProvider,
    EphemerisCoverage,
    EphemerisCoverageError,
    SpiceEphemerisProvider,
    TabulatedEphemerisProvider,
)


def _body() -> CircularOrbitBody:
    return CircularOrbitBody("Aster", "background", 1.0e10, 1.0e6, 1.0e12, 0.0)


def test_analytical_provider_enforces_body_epoch_frame_and_time_coverage() -> None:
    coverage = EphemerisCoverage(0.0, 1000.0, ("Aster",), "J2000", "Helios", "TDB")
    provider = AnalyticalEphemerisProvider({"Aster": _body()}, 1.0e16, coverage)
    state = provider.state("Aster", 50.0)

    assert state.frame == "J2000"
    assert state.center == "Helios"
    assert state.time_system == "TDB"
    assert state.position_m.shape == (3,)
    with pytest.raises(EphemerisCoverageError, match="outside"):
        provider.state("Aster", 1001.0)
    with pytest.raises(EphemerisCoverageError, match="body"):
        provider.state("Missing", 50.0)


def test_tabulated_provider_uses_position_velocity_consistent_hermite_interpolation() -> None:
    times = np.array([0.0, 2.0])
    states = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 4.0, 0.0, 0.0]])
    provider = TabulatedEphemerisProvider(
        times,
        {"body": states},
        frame="TEST_I",
        center="origin",
        time_system="TT",
        source="unit fixture",
    )
    midpoint = provider.state("body", 1.0)

    assert midpoint.position_m[0] == pytest.approx(1.0)
    assert midpoint.velocity_mps[0] == pytest.approx(2.0)
    with pytest.raises(EphemerisCoverageError):
        provider.state("body", -1.0)


def test_spice_provider_failure_does_not_substitute_analytical_state() -> None:
    with pytest.raises(EphemerisUnavailableError, match="SPICE"):
        SpiceEphemerisProvider(
            ["missing-public-kernel.bsp"],
            observer="SUN",
            body_names=("EARTH",),
            start_time_s=0.0,
            end_time_s=1.0,
        )


def test_provider_contract_rejects_mismatched_or_invalid_coverage() -> None:
    with pytest.raises(ValueError, match="positive duration"):
        EphemerisCoverage(1.0, 1.0, ("body",), "frame", "center", "TDB")
    coverage = EphemerisCoverage(0.0, 1.0, ("other",), "frame", "center", "TDB")
    with pytest.raises(ValueError, match="exactly match"):
        AnalyticalEphemerisProvider({"Aster": _body()}, 1.0e16, coverage)
