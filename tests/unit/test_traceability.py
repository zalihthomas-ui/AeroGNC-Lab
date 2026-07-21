import csv
import re
from pathlib import Path


def test_every_system_requirement_has_one_complete_traceability_row() -> None:
    requirement_text = Path("requirements/system_requirements.md").read_text(encoding="utf-8")
    identifiers = re.findall(r"\*\*(SYS-[A-Z]+-\d{3})", requirement_text)
    with Path("requirements/traceability_matrix.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    traced_identifiers = [row["Requirement"] for row in rows]
    assert len(identifiers) == len(set(identifiers))
    assert len(traced_identifiers) == len(set(traced_identifiers))
    assert set(traced_identifiers) == set(identifiers)
    assert all(all(value.strip() for value in row.values()) for row in rows)
    assert {row["Status"] for row in rows} <= {"Planned", "Implemented", "Verified"}
