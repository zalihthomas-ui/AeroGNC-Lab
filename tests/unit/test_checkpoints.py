import json
from pathlib import Path

import numpy as np
import pytest

from aerognc.mathematics.adaptive_integrators import AdaptiveOptions, integrate_adaptive
from aerognc.simulation.checkpoints import (
    CheckpointIntegrityError,
    IntegratorCheckpoint,
    checkpoint_from_result,
    load_checkpoint,
    write_checkpoint,
)


def _options(initial_step_s: float) -> AdaptiveOptions:
    return AdaptiveOptions(
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
        initial_step_s=initial_step_s,
        minimum_step_s=1.0e-12,
        maximum_step_s=0.25,
    )


def test_checkpoint_round_trip_and_resumed_propagation(tmp_path: Path) -> None:
    def derivative(_time_s: float, state: np.ndarray) -> np.ndarray:
        return -0.3 * state

    first = integrate_adaptive(derivative, [2.0, -1.0], (0.0, 1.25), options=_options(0.1))
    checkpoint = checkpoint_from_result(
        first,
        epoch="2035-01-01T00:00:00Z",
        metadata={"scenario": "decay", "seed": 17},
    )
    path = write_checkpoint(checkpoint, tmp_path / "restart.json")
    loaded = load_checkpoint(path)
    resumed = integrate_adaptive(
        derivative,
        loaded.state,
        (loaded.time_s, 3.0),
        options=_options(loaded.next_step_s),
    )
    continuous = integrate_adaptive(derivative, [2.0, -1.0], (0.0, 3.0), options=_options(0.1))

    assert loaded.epoch == checkpoint.epoch
    assert dict(loaded.metadata) == {"scenario": "decay", "seed": 17}
    np.testing.assert_array_equal(loaded.state, checkpoint.state)
    np.testing.assert_allclose(resumed.state[-1], continuous.state[-1], rtol=2.0e-10, atol=1.0e-12)


def test_checkpoint_detects_descriptor_and_payload_tampering(tmp_path: Path) -> None:
    checkpoint = IntegratorCheckpoint("epoch", 1.0, [1.0, 2.0], 0.1, {"case": "unit"})
    path = write_checkpoint(checkpoint, tmp_path / "restart.json")
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor["time_s"] = 2.0
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="content hash"):
        load_checkpoint(path)

    path = write_checkpoint(checkpoint, tmp_path / "restart.json")
    path.with_suffix(".npz").write_bytes(b"not an npz")
    with pytest.raises(CheckpointIntegrityError, match="payload hash"):
        load_checkpoint(path)


def test_checkpoint_rejects_invalid_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epoch"):
        IntegratorCheckpoint("", 0.0, [1.0], 0.1, {})
    with pytest.raises(ValueError, match="next_step"):
        IntegratorCheckpoint("epoch", 0.0, [1.0], 0.0, {})
    with pytest.raises(ValueError, match="JSON"):
        IntegratorCheckpoint("epoch", 0.0, [1.0], 0.1, {"bad": np.nan})
    checkpoint = IntegratorCheckpoint("epoch", 0.0, [1.0], 0.1, {})
    with pytest.raises(ValueError, match=r"\.json"):
        write_checkpoint(checkpoint, tmp_path / "restart.bin")
