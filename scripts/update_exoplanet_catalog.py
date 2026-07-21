"""Refresh the compact confirmed-exoplanet snapshot from NASA's TAP service."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

SOURCE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SOURCE_TABLE = "pscomppars"
FIELDS = (
    "pl_name",
    "hostname",
    "sy_pnum",
    "discoverymethod",
    "disc_year",
    "pl_orbper",
    "pl_orbsmax",
    "pl_rade",
    "pl_bmasse",
    "sy_dist",
    "ra",
    "dec",
    "st_spectype",
    "st_teff",
    "st_mass",
    "st_rad",
)
QUERY = f"select {','.join(FIELDS)} from {SOURCE_TABLE} order by pl_name"


def _parse_payload(payload: str) -> list[dict[str, str]]:
    """Validate, normalize, and sort one NASA TAP CSV response."""
    reader = csv.DictReader(io.StringIO(payload))
    if reader.fieldnames != list(FIELDS):
        raise RuntimeError(f"unexpected NASA TAP schema: {reader.fieldnames!r}")
    rows = [dict(row) for row in reader]
    if len(rows) < 1_000:
        raise RuntimeError("NASA TAP response is implausibly small; refusing to replace snapshot")
    names = [row["pl_name"].strip() for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise RuntimeError("NASA TAP response has blank or duplicate planet names")
    return sorted(rows, key=lambda row: row["pl_name"].casefold())


def _download(timeout_s: float) -> list[dict[str, str]]:
    encoded = urllib.parse.urlencode({"query": QUERY, "format": "csv"})
    request = urllib.request.Request(
        f"{SOURCE_URL}?{encoded}",
        headers={"User-Agent": "AeroGNC-Lab catalog snapshot updater"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8-sig")
    return _parse_payload(payload)


def _write_snapshot(output_directory: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "nasa_confirmed_exoplanets.csv"
    temporary_path = csv_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "source_name": "NASA Exoplanet Archive",
        "source_url": SOURCE_URL,
        "source_table": SOURCE_TABLE,
        "query": QUERY,
        "retrieved_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "row_count": len(rows),
        "sha256": digest,
        "fields": list(FIELDS),
        "scope_note": (
            "Point-in-time confirmed-planet snapshot. Blank values mean not reported in the "
            "selected composite fields. This is not a census of all Milky Way planets and "
            "does not provide complete dynamical ephemerides."
        ),
        "documentation_url": ("https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html"),
    }
    metadata_path = output_directory / "nasa_confirmed_exoplanets.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "catalogs",
    )
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="normalize a previously downloaded TAP CSV instead of using the network",
    )
    arguments = parser.parse_args()
    if arguments.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")
    rows = (
        _parse_payload(arguments.input_csv.read_text(encoding="utf-8-sig"))
        if arguments.input_csv is not None
        else _download(arguments.timeout_s)
    )
    csv_path, metadata_path = _write_snapshot(arguments.output, rows)
    print(f"Wrote {len(rows)} confirmed planets to {csv_path}")
    print(f"Wrote provenance and SHA-256 to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
