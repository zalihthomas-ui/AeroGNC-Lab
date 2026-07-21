from datetime import UTC, datetime

import numpy as np
import pytest

from aerognc.project.manifest import file_sha256, new_run_manifest
from aerognc.project.result_store import (
    ResultDataset,
    ResultIntegrityError,
    ResultStore,
    infer_channel_unit,
)


def _dataset() -> ResultDataset:
    return ResultDataset(
        scenario_name="case-a",
        time_s=np.array([0.0, 0.5, 1.0]),
        channels={
            "altitude_m": np.array([0.0, 2.0, 3.0]),
            "velocity_up_mps": np.array([5.0, 3.0, 1.0]),
            "mach": np.array([0.0, 0.02, 0.03]),
        },
        units={"altitude_m": "m", "velocity_up_mps": "m/s", "mach": "1"},
        events=({"name": "burnout", "time_s": 0.5},),
        maxima={"altitude": {"value": 3.0, "time_s": 1.0, "unit": "m"}},
        metadata={"frame": "NED"},
    )


def _manifest(tmp_path, *, status="completed", failure_reason=None):
    configuration = tmp_path / "case.yaml"
    configuration.write_text("case: true\n", encoding="utf-8")
    return new_run_manifest(
        project_name="Project",
        scenario_name="case-a",
        workflow="three-dof",
        safety_scope="Fictional civilian synthetic case.",
        configuration_path="configs/case.yaml",
        configuration_sha256=file_sha256(configuration),
        seed=1,
        solver_settings={"method": "rk4", "step_s": 0.5},
        parameters={},
        status=status,
        execution_time_s=0.01,
        failure_reason=failure_reason,
        events=({"name": "burnout", "time_s": 0.5},),
        maxima={"altitude": {"value": 3.0, "time_s": 1.0, "unit": "m"}},
        created=datetime(2026, 7, 20, 15, 0, 0, 123456, tzinfo=UTC),
    )


def test_result_store_atomic_commit_index_reload_and_rebuild(tmp_path) -> None:
    store = ResultStore(tmp_path / "runs")
    stored = store.commit(_manifest(tmp_path), _dataset())

    assert stored.directory.is_dir()
    assert {item.role for item in stored.manifest.artifacts} == {
        "trajectory-npz",
        "trajectory-csv",
        "dataset-metadata",
    }
    records = store.list_runs(project_name="Project")
    assert len(records) == 1
    assert records[0].run_id == stored.manifest.run_id

    reloaded = store.load(stored.manifest.run_id)
    assert reloaded.dataset is not None
    np.testing.assert_array_equal(reloaded.dataset.time_s, _dataset().time_s)
    np.testing.assert_array_equal(
        reloaded.dataset.channels["altitude_m"], _dataset().channels["altitude_m"]
    )
    assert reloaded.dataset.units["altitude_m"] == "m"
    assert store.rebuild_index() == 1
    assert store.list_runs()[0].input_fingerprint == stored.manifest.input_fingerprint


def test_result_store_detects_tampering_and_never_overwrites(tmp_path) -> None:
    store = ResultStore(tmp_path / "runs")
    stored = store.commit(_manifest(tmp_path), _dataset())
    with pytest.raises(FileExistsError, match="already exists"):
        store.commit(_manifest(tmp_path), _dataset())

    trajectory = stored.directory / "trajectory.csv"
    trajectory.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ResultIntegrityError, match="hash mismatch"):
        store.load(stored.manifest.run_id)


def test_result_store_records_failed_run_without_dataset(tmp_path) -> None:
    store = ResultStore(tmp_path / "runs")
    manifest = _manifest(tmp_path, status="failed", failure_reason="solver did not converge")
    stored = store.commit(manifest, None)

    assert stored.dataset is None
    assert store.load(manifest.run_id).manifest.failure_reason == "solver did not converge"
    assert store.list_runs(status="failed")[0].status == "failed"


def test_dataset_guards_units_and_immutability() -> None:
    dataset = _dataset()
    assert infer_channel_unit("dynamic_pressure_pa") == "Pa"
    assert infer_channel_unit("quaternion_q2") == "1"
    with pytest.raises(ValueError, match="cannot infer"):
        infer_channel_unit("ambiguous")
    with pytest.raises(ValueError, match="read-only"):
        dataset.time_s[0] = 1.0
    with pytest.raises(ValueError, match="exactly match"):
        ResultDataset(
            "bad",
            np.array([0.0, 1.0]),
            {"altitude_m": np.array([0.0, 1.0])},
            {},
        )
