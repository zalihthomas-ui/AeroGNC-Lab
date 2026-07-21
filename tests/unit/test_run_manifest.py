import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from aerognc.project.manifest import (
    ArtifactRecord,
    RequirementOutcome,
    file_sha256,
    input_fingerprint,
    load_manifest,
    manifest_as_mapping,
    manifest_from_mapping,
    new_run_manifest,
    with_artifacts,
    write_manifest,
)


def _manifest(tmp_path):
    configuration = tmp_path / "case.yaml"
    configuration.write_text("step_s: 0.1\n", encoding="utf-8")
    return new_run_manifest(
        project_name="Project",
        scenario_name="case-a",
        workflow="three-dof",
        safety_scope="Fictional civilian vehicle with synthetic parameters.",
        configuration_path="configs/case.yaml",
        configuration_sha256=file_sha256(configuration),
        seed=17,
        solver_settings={"method": "rk4", "step_s": 0.1},
        parameters={"scale": 1.0},
        status="completed",
        execution_time_s=0.25,
        requirements=(RequirementOutcome("REQ-ALT-001", True, 100.0, 90.0, 10.0, "m"),),
        events=({"name": "apogee", "time_s": 4.0},),
        maxima={"altitude": {"value": 100.0, "unit": "m", "time_s": 4.0}},
        created=datetime(2026, 7, 20, 12, 30, tzinfo=UTC),
    )


def test_manifest_fingerprint_is_stable_and_sensitive(tmp_path) -> None:
    configuration = tmp_path / "case.yaml"
    configuration.write_text("value: 1\n", encoding="utf-8")
    digest = file_sha256(configuration)
    base = dict(
        project_name="Project",
        scenario_name="case-a",
        workflow="three-dof",
        configuration_sha256=digest,
        seed=3,
        solver_settings={"step_s": 0.1},
        parameters={},
        safety_scope="Fictional civilian synthetic case",
    )
    assert input_fingerprint(**base) == input_fingerprint(**base)
    assert input_fingerprint(**base) != input_fingerprint(**{**base, "seed": 4})


def test_manifest_round_trip_and_artifact_inventory(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    artifact_path = tmp_path / "trajectory.csv"
    artifact_path.write_text("time_s,value_m\n0,1\n", encoding="utf-8")
    artifact = ArtifactRecord(
        role="trajectory-csv",
        relative_path="trajectory.csv",
        sha256=file_sha256(artifact_path),
        media_type="text/csv",
        size_bytes=artifact_path.stat().st_size,
    )
    manifest = with_artifacts(manifest, (artifact,))
    path = write_manifest(manifest, tmp_path / "manifest.json")

    loaded = load_manifest(path)

    assert loaded == manifest
    assert loaded.run_id.startswith("case-a-20260720T123000.000000Z-")
    assert loaded.artifacts[0].sha256 == file_sha256(artifact_path)
    assert loaded.requirements[0].passed
    assert loaded.events[0]["name"] == "apogee"


def test_manifest_rejects_inconsistent_status_and_unknown_fields(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="failure_reason"):
        replace(manifest, status="failed")
    with pytest.raises(ValueError, match="completed"):
        replace(manifest, failure_reason="not allowed")

    data = manifest_as_mapping(manifest)
    data["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        manifest_from_mapping(json.loads(json.dumps(data)))
