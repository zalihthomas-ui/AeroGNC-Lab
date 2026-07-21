"""Shared validation for the intentionally scoped CCSDS KVN boundaries."""

from __future__ import annotations

from datetime import datetime

SUPPORTED_TIME_SYSTEMS = frozenset({"UTC", "TAI", "TT", "TDB", "GPS"})


def validate_time_system(value: str) -> str:
    """Return a canonical supported time-system label."""
    canonical = value.strip().upper()
    if canonical not in SUPPORTED_TIME_SYSTEMS:
        raise ValueError(
            f"unsupported CCSDS time system {value!r}; expected {sorted(SUPPORTED_TIME_SYSTEMS)}"
        )
    return canonical


def require_text(**values: str) -> None:
    """Reject empty mandatory metadata with its field name."""
    for name, value in values.items():
        if not value.strip():
            raise ValueError(f"CCSDS metadata {name} cannot be empty")


def epoch_text(epoch: datetime) -> str:
    """Write deterministic extended ISO text without an implicit time conversion."""
    if epoch.tzinfo is not None:
        epoch = epoch.replace(tzinfo=None)
    return epoch.isoformat(timespec="microseconds")


def parse_epoch(value: str, *, context: str) -> datetime:
    """Parse one KVN epoch with a contextual error."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError as error:
        raise ValueError(f"invalid {context} epoch {value!r}") from error
