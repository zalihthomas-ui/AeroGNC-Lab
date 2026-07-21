"""Explicit UTC, TAI, TT, and approximate TDB conversion for reproducible metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

SECONDS_PER_DAY = 86_400.0
JULIAN_DATE_UNIX_EPOCH = 2_440_587.5
JULIAN_DATE_J2000 = 2_451_545.0
TT_MINUS_TAI_S = 32.184
IERS_BULLETIN_C_NUMBER = 72
IERS_BULLETIN_C_DATE = "2026-07-06"
IERS_BULLETIN_C_URL = "https://datacenter.iers.org/data/html/bulletinc-072.html"
LEAP_SECOND_TABLE_VALID_UNTIL = "2026-12-31"

# Effective UTC date and TAI-UTC after the change. Dates follow the IERS history;
# Bulletin C 72 confirms that the latest value remains 37 s through December 2026.
_TAI_MINUS_UTC: tuple[tuple[datetime, int], ...] = tuple(
    (datetime.fromisoformat(date).replace(tzinfo=UTC), offset)
    for date, offset in (
        ("1972-01-01", 10),
        ("1972-07-01", 11),
        ("1973-01-01", 12),
        ("1974-01-01", 13),
        ("1975-01-01", 14),
        ("1976-01-01", 15),
        ("1977-01-01", 16),
        ("1978-01-01", 17),
        ("1979-01-01", 18),
        ("1980-01-01", 19),
        ("1981-07-01", 20),
        ("1982-07-01", 21),
        ("1983-07-01", 22),
        ("1985-07-01", 23),
        ("1988-01-01", 24),
        ("1990-01-01", 25),
        ("1991-01-01", 26),
        ("1992-07-01", 27),
        ("1993-07-01", 28),
        ("1994-07-01", 29),
        ("1996-01-01", 30),
        ("1997-07-01", 31),
        ("1999-01-01", 32),
        ("2006-01-01", 33),
        ("2009-01-01", 34),
        ("2012-07-01", 35),
        ("2015-07-01", 36),
        ("2017-01-01", 37),
    )
)


@dataclass(frozen=True, slots=True)
class TimeScaleEpoch:
    """One UTC instant represented on common atomic/dynamical time scales."""

    utc: datetime
    tai_minus_utc_s: int
    julian_date_utc: float
    julian_date_tai: float
    julian_date_tt: float
    julian_date_tdb_approx: float
    tdb_minus_tt_s_approx: float
    tdb_seconds_past_j2000_approx: float


def tai_minus_utc_s(utc_epoch: datetime) -> int:
    """Return the tabulated TAI-UTC step for an aware UTC date since 1972."""
    if utc_epoch.tzinfo is None or utc_epoch.utcoffset() is None:
        raise ValueError("UTC epoch must be timezone-aware")
    epoch = utc_epoch.astimezone(UTC)
    matches = [offset for effective, offset in _TAI_MINUS_UTC if epoch >= effective]
    if not matches:
        raise ValueError("bundled leap-second history supports UTC dates from 1972 onward")
    return matches[-1]


def utc_to_time_scales(utc_epoch: datetime) -> TimeScaleEpoch:
    """Convert aware UTC to Julian UTC/TAI/TT and a documented low-order TDB."""
    utc = utc_epoch.astimezone(UTC) if utc_epoch.tzinfo is not None else utc_epoch
    if utc.tzinfo is None or utc.utcoffset() is None:
        raise ValueError("UTC epoch must be timezone-aware")
    offset_s = tai_minus_utc_s(utc)
    julian_utc = JULIAN_DATE_UNIX_EPOCH + utc.timestamp() / SECONDS_PER_DAY
    julian_tai = julian_utc + offset_s / SECONDS_PER_DAY
    julian_tt = julian_tai + TT_MINUS_TAI_S / SECONDS_PER_DAY
    tt_seconds_from_j2000 = (julian_tt - JULIAN_DATE_J2000) * SECONDS_PER_DAY
    # NAIF's low-order DELTET form: E=M+EB*sin(M), TDB-TT=K*sin(E).
    mean_anomaly_rad = 6.239996 + 1.99096871e-7 * tt_seconds_from_j2000
    eccentric_anomaly_rad = mean_anomaly_rad + 1.671e-2 * np.sin(mean_anomaly_rad)
    tdb_minus_tt_s = float(1.657e-3 * np.sin(eccentric_anomaly_rad))
    julian_tdb = julian_tt + tdb_minus_tt_s / SECONDS_PER_DAY
    return TimeScaleEpoch(
        utc=utc,
        tai_minus_utc_s=offset_s,
        julian_date_utc=float(julian_utc),
        julian_date_tai=float(julian_tai),
        julian_date_tt=float(julian_tt),
        julian_date_tdb_approx=float(julian_tdb),
        tdb_minus_tt_s_approx=tdb_minus_tt_s,
        tdb_seconds_past_j2000_approx=float((julian_tdb - JULIAN_DATE_J2000) * SECONDS_PER_DAY),
    )
