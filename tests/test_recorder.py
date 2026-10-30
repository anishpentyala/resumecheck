import numpy as np
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


class TestFingerprintStructure:
    """Fingerprint must distinguish structure, dtype, and shape - not just
    raw byte content (audit 2026-10-30)."""

    def test_join_separator_ambiguity(self):
        # b"\x01|\x02" as one tensor vs two tensors b"\x01", b"\x02"
        a = np.frombuffer(b"\x01|\x02", dtype=np.uint8)
        b = [np.frombuffer(b"\x01", dtype=np.uint8),
             np.frombuffer(b"\x02", dtype=np.uint8)]
        assert fingerprint_tensor(a) != fingerprint_tensor(b)

    def test_dtype_distinguished(self):
        f32 = np.array([1.0], dtype=np.float32)
        i32 = np.array([0x3F800000], dtype=np.int32)  # same bytes
        assert fingerprint_tensor(f32) != fingerprint_tensor(i32)

    def test_shape_distinguished(self):
        assert fingerprint_tensor(np.zeros((2, 2))) != fingerprint_tensor(np.zeros(4))

    def test_str_vs_bytes_distinguished(self):
        assert fingerprint_tensor("ab") != fingerprint_tensor(b"ab")

    def test_same_content_stable(self):
        t1 = np.arange(6, dtype=np.float64).reshape(2, 3)
        t2 = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert fingerprint_tensor(t1) == fingerprint_tensor(t2)
