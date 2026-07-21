import csv
import json

import numpy as np
import pytest

from aerognc.data.telemetry import (
    ChannelMapping,
    QualityMapping,
    TelemetryMapping,
    TimestampMapping,
    import_telemetry_csv,
    load_telemetry_mapping,
    write_normalized_telemetry_csv,
    write_telemetry_provenance,
)


def _mapping() -> TelemetryMapping:
    return TelemetryMapping(
        "1.0",
        TimestampMapping("clock_ms", "ms", 0.001),
        QualityMapping("quality", ("OK",), "keep_nan"),
        (
            ChannelMapping("altitude_ft", "altitude_m", "ft", "m", 0.3048, 0.0, "keep_nan"),
            ChannelMapping("temperature_c", "temperature_k", "degC", "K", 1.0, 273.15),
        ),
    )


def test_versioned_csv_mapping_normalises_units_quality_and_provenance(tmp_path) -> None:
    source = tmp_path / "telemetry.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("clock_ms", "quality", "altitude_ft", "temperature_c"))
        writer.writerows(
            (
                (0, "OK", 100, 20),
                (100, "OK", "", 21),
                (200, "BAD", 102, 22),
                (300, "OK", 103, 23),
            )
        )

    record = import_telemetry_csv(source, _mapping())
    np.testing.assert_allclose(record.time_s, [0.0, 0.1, 0.2, 0.3])
    assert record.channels["altitude_m"][0] == pytest.approx(30.48)
    assert np.isnan(record.channels["altitude_m"][[1, 2]]).all()
    assert record.channels["temperature_k"][3] == pytest.approx(296.15)
    np.testing.assert_array_equal(record.quality_valid, [True, True, False, True])
    assert record.units == {"altitude_m": "m", "temperature_k": "K"}
    assert record.provenance.source_sha256
    assert record.provenance.mapping_sha256 == _mapping().sha256
    assert record.provenance.rows_read == 4

    normalized = write_normalized_telemetry_csv(record, tmp_path / "normalized.csv")
    provenance = write_telemetry_provenance(record, tmp_path / "provenance.json")
    assert normalized.read_text(encoding="utf-8").splitlines()[0] == (
        "time_s,altitude_m,temperature_k,quality_valid"
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert payload["units"] == {"altitude_m": "m", "temperature_k": "K"}


def test_yaml_mapping_is_strict_and_drop_policy_is_accounted(tmp_path) -> None:
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        """schema_version: '1.0'
timestamp:
  source_name: time_s
  source_unit: s
  scale_to_s: 1.0
  offset_s: 0.0
quality:
  source_name: q
  accepted_values: [VALID]
  invalid_policy: drop_row
channels:
  - source_name: speed
    destination_name: speed_mps
    source_unit: m/s
    destination_unit: m/s
    scale: 1.0
    offset: 0.0
    missing_policy: drop_row
""",
        encoding="utf-8",
    )
    mapping = load_telemetry_mapping(mapping_path)
    source = tmp_path / "data.csv"
    source.write_text("time_s,q,speed\n0,VALID,2\n1,BAD,3\n2,VALID,\n3,VALID,5\n", encoding="utf-8")

    record = import_telemetry_csv(source, mapping)
    np.testing.assert_allclose(record.time_s, [0.0, 3.0])
    np.testing.assert_allclose(record.channels["speed_mps"], [2.0, 5.0])
    assert record.provenance.rows_dropped == 2

    mapping_path.write_text(mapping_path.read_text(encoding="utf-8") + "unknown: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_telemetry_mapping(mapping_path)


def test_mapping_and_import_validation_fail_with_actionable_context(tmp_path) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        TelemetryMapping(
            "2.0",
            TimestampMapping("time", "s", 1.0),
            QualityMapping("quality", ("OK",)),
            (ChannelMapping("x", "x", "m", "m"),),
        )
    source = tmp_path / "bad.csv"
    source.write_text("clock_ms,quality,altitude_ft\n0,OK,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        import_telemetry_csv(source, _mapping())
