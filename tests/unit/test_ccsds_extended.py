from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from aerognc.interoperability.ccsds_aem import AemMetadata, parse_aem_kvn, write_aem_kvn
from aerognc.interoperability.ccsds_opm import OpmMetadata, OpmState, parse_opm_kvn, write_opm_kvn
from aerognc.interoperability.ccsds_tdm import (
    TdmMetadata,
    TdmObservation,
    parse_tdm_kvn,
    write_tdm_kvn,
)


def test_aem_scalar_first_quaternion_round_trip_and_monotonic_validation(tmp_path: Path) -> None:
    metadata = AemMetadata(
        datetime(2026, 7, 20),
        "AEROGNC-LAB",
        "AEM-001",
        "FictionalSat",
        "FICTIONAL-001",
        "J2000",
        "SC_BODY_1",
        "TT",
        datetime(2035, 1, 1),
    )
    quaternion = np.array([[1.0, 0.0, 0.0, 0.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]])
    path = write_aem_kvn([10.0, 20.0], quaternion, metadata, tmp_path / "attitude.aem")
    parsed_metadata, time_s, parsed_quaternion = parse_aem_kvn(path)

    assert parsed_metadata["TIME_SYSTEM"] == "TT"
    np.testing.assert_allclose(time_s, [0.0, 10.0])
    np.testing.assert_allclose(parsed_quaternion, quaternion)
    with pytest.raises(ValueError, match="strictly increasing"):
        write_aem_kvn([0.0, 0.0], quaternion, metadata, tmp_path / "bad.aem")


def test_opm_cartesian_si_boundary_round_trip(tmp_path: Path) -> None:
    metadata = OpmMetadata(
        datetime(2026, 7, 20),
        "AEROGNC-LAB",
        "OPM-001",
        "FictionalSat",
        "FICTIONAL-001",
        "ORBis-A",
        "J2000",
        "TDB",
    )
    record = OpmState(
        datetime(2035, 1, 1),
        [7.0e6, 2.0e5, -1.0e5, -200.0, 7_500.0, 30.0],
        850.0,
    )
    path = write_opm_kvn(record, metadata, tmp_path / "state.opm")
    parsed_metadata, parsed = parse_opm_kvn(path)

    assert parsed_metadata["CCSDS_OPM_VERS"] == "3.0"
    np.testing.assert_allclose(parsed.state_si, record.state_si)
    assert parsed.mass_kg == pytest.approx(850.0)
    assert parsed.epoch == record.epoch


def test_tdm_range_angle_doppler_round_trip_and_epoch_validation(tmp_path: Path) -> None:
    metadata = TdmMetadata(
        datetime(2026, 7, 20),
        "AEROGNC-LAB",
        "TDM-001",
        "FICTIONAL_STATION",
        "FICTIONAL_SPACECRAFT",
        "UTC",
        datetime(2035, 1, 1),
    )
    observations = (
        TdmObservation(0.0, "RANGE", 1.2e6),
        TdmObservation(1.0, "ANGLE_1", np.deg2rad(12.0)),
        TdmObservation(2.0, "DOPPLER_INSTANTANEOUS", -1250.0),
    )
    path = write_tdm_kvn(observations, metadata, tmp_path / "tracking.tdm")
    parsed_metadata, parsed = parse_tdm_kvn(path)

    assert parsed_metadata["TIME_SYSTEM"] == "UTC"
    assert [item.observable for item in parsed] == [
        "RANGE",
        "ANGLE_1",
        "DOPPLER_INSTANTANEOUS",
    ]
    assert [item.value for item in parsed] == pytest.approx([1.2e6, np.deg2rad(12.0), -1250.0])

    nonmonotonic = (TdmObservation(1.0, "RANGE", 1.0), TdmObservation(0.0, "RANGE", 2.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        write_tdm_kvn(nonmonotonic, metadata, tmp_path / "bad.tdm")


def test_extended_ccsds_boundaries_reject_invalid_metadata_and_missing_fields(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="time system"):
        OpmMetadata(
            datetime(2026, 1, 1),
            "A",
            "M",
            "O",
            "I",
            "C",
            "F",
            "LOCAL",
        )
    path = tmp_path / "incomplete.opm"
    path.write_text("CCSDS_OPM_VERS = 3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing mandatory"):
        parse_opm_kvn(path)
