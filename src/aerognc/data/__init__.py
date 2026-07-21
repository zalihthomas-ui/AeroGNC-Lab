"""Versioned telemetry ingestion, normalisation, and stored-data analysis."""

from aerognc.data.analysis import (
    TelemetryComparison,
    TelemetryGap,
    TelemetryResidualStatistics,
    compare_telemetry_channels,
    estimate_marker_time_alignment,
    find_telemetry_gaps,
    telemetry_residual_statistics,
    write_aligned_comparison_csv,
    write_telemetry_comparison_report,
)
from aerognc.data.telemetry import (
    ChannelMapping,
    MissingValuePolicy,
    QualityMapping,
    TelemetryMapping,
    TelemetryProvenance,
    TelemetryRecord,
    TimestampMapping,
    import_telemetry_csv,
    load_telemetry_mapping,
    write_normalized_telemetry_csv,
    write_telemetry_provenance,
)

__all__ = [
    "ChannelMapping",
    "MissingValuePolicy",
    "QualityMapping",
    "TelemetryComparison",
    "TelemetryGap",
    "TelemetryMapping",
    "TelemetryProvenance",
    "TelemetryRecord",
    "TelemetryResidualStatistics",
    "TimestampMapping",
    "compare_telemetry_channels",
    "estimate_marker_time_alignment",
    "find_telemetry_gaps",
    "import_telemetry_csv",
    "load_telemetry_mapping",
    "telemetry_residual_statistics",
    "write_aligned_comparison_csv",
    "write_normalized_telemetry_csv",
    "write_telemetry_comparison_report",
    "write_telemetry_provenance",
]
