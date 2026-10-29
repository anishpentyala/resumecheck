import torch

from resumecheck.recorder import Trajectory, TrajectoryRecorder, fingerprint_tensor


def test_fingerprint_stable_and_sensitive():
    t = torch.arange(8, dtype=torch.float32)
    assert fingerprint_tensor(t) == fingerprint_tensor(t.clone())
    assert fingerprint_tensor(t) != fingerprint_tensor(t + 1)


def test_fingerprint_nested_and_scalar():
    b = {"x": torch.ones(2), "y": [3, "a"]}
    assert fingerprint_tensor(b) == fingerprint_tensor({"x": torch.ones(2), "y": [3, "a"]})
    assert fingerprint_tensor(5) != fingerprint_tensor(6)


def test_save_load_roundtrip(tmp_path):
    rec = TrajectoryRecorder("run", seed=7)
    rec.record(0, loss=1.5, batch=torch.tensor([1.0]), grad_norm=0.3, lr=1e-3)
    rec.record(1, loss=1.2)
    p = tmp_path / "t.json"
    rec.trajectory.save(p)
    loaded = Trajectory.load(p)
    assert loaded.name == "run" and loaded.meta["seed"] == 7
    assert [s.step for s in loaded.steps] == [0, 1]
    assert loaded.steps[0].loss == 1.5
    assert loaded.steps[0].extras["lr"] == 1e-3
    assert loaded.steps[1].batch_fingerprint is None
