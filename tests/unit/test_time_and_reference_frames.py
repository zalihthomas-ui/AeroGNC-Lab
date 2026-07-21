from datetime import UTC, datetime

import numpy as np
import pytest

from aerognc.astrodynamics.reference_frames import (
    ecliptic_to_equatorial_j2000_matrix,
    transform_inertial_state,
)
from aerognc.astrodynamics.time_systems import tai_minus_utc_s, utc_to_time_scales


def test_current_bundled_leap_offset_and_tt_definition() -> None:
    epoch = utc_to_time_scales(datetime(2026, 7, 19, tzinfo=UTC))

    assert epoch.tai_minus_utc_s == 37
    assert (epoch.julian_date_tt - epoch.julian_date_utc) * 86_400.0 == pytest.approx(
        69.184, abs=2.0e-5
    )
    assert abs(epoch.tdb_minus_tt_s_approx) < 0.0017
    assert tai_minus_utc_s(datetime(2016, 12, 31, 12, tzinfo=UTC)) == 36


def test_julian_date_at_utc_j2000_calendar_noon() -> None:
    epoch = utc_to_time_scales(datetime(2000, 1, 1, 12, tzinfo=UTC))
    assert epoch.julian_date_utc == pytest.approx(2_451_545.0)
    assert epoch.tai_minus_utc_s == 32


def test_time_conversion_requires_aware_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_to_time_scales(datetime(2026, 1, 1))


def test_ecliptic_equatorial_rotation_is_orthonormal_and_round_trips() -> None:
    matrix = ecliptic_to_equatorial_j2000_matrix()
    state = np.array([1.0e11, 2.0e10, -3.0e9, -4_000.0, 25_000.0, 1_200.0])
    equatorial = transform_inertial_state(state, "HELIOS_ECLIPJ2000", "J2000")
    recovered = transform_inertial_state(equatorial, "J2000", "HELIOS_ECLIPJ2000")

    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1.0e-15)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    assert recovered == pytest.approx(state, rel=1.0e-15, abs=1.0e-6)
