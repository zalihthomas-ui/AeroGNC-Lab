from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from aerognc.interoperability.ccsds_oem import OemMetadata, parse_oem_kvn, write_oem_kvn


def test_oem_kvn_round_trip_preserves_si_state(tmp_path: Path) -> None:
    time_s = np.array([100.0, 160.0, 220.0])
    states_si = np.array(
        [
            [7.0e6, 0.0, 0.0, 0.0, 7_500.0, 0.0],
            [6.98e6, 4.5e5, 0.0, -480.0, 7_480.0, 0.0],
            [6.94e6, 8.95e5, 0.0, -960.0, 7_430.0, 0.0],
        ]
    )
    metadata = OemMetadata(
        datetime(2026, 7, 19),
        "AEROGNC-LAB",
        "TEST-001",
        "SyntheticSat",
        "FICTIONAL-001",
        "EARTH",
        "J2000",
        "TDB",
        datetime(2030, 1, 1),
    )
    path = write_oem_kvn(time_s, states_si, metadata, tmp_path / "test.oem")

    parsed_metadata, parsed_time_s, parsed_states_si = parse_oem_kvn(path)

    assert parsed_metadata["CCSDS_OEM_VERS"] == "3.0"
    assert parsed_metadata["TIME_SYSTEM"] == "TDB"
    assert parsed_time_s == pytest.approx([0.0, 60.0, 120.0])
    assert parsed_states_si == pytest.approx(states_si)
    assert "km and km/s" in path.read_text(encoding="utf-8")


def test_oem_rejects_nonmonotonic_samples(tmp_path: Path) -> None:
    metadata = OemMetadata(
        datetime(2026, 1, 1),
        "A",
        "M",
        "O",
        "I",
        "EARTH",
        "J2000",
        "UTC",
        datetime(2026, 1, 1),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        write_oem_kvn([0.0, 0.0], np.zeros((2, 6)), metadata, tmp_path / "bad.oem")
