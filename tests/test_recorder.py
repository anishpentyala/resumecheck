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


def test_fingerprint_nested_container_boundaries():
    assert fingerprint_tensor([[1], 2]) != fingerprint_tensor([[1, 2]])
    assert fingerprint_tensor({"a": {}, "b": 1}) != fingerprint_tensor({"a": {"b": 1}})
    assert fingerprint_tensor([[], 1]) != fingerprint_tensor([[1]])
    assert fingerprint_tensor([1, 2]) != fingerprint_tensor((1, 2))
    assert fingerprint_tensor([1, 2]).startswith("v2:")


class TestLowPrecisionFingerprint:
    def test_bfloat16_exact_bits_and_shape(self):
        t = torch.tensor([1., 2.], dtype=torch.bfloat16)
        assert fingerprint_tensor(t) == fingerprint_tensor(t.clone())
        assert fingerprint_tensor(t) != fingerprint_tensor(torch.tensor([1., 3.], dtype=t.dtype))
        assert fingerprint_tensor(t) != fingerprint_tensor(t.reshape(1, 2))
        assert fingerprint_tensor(t) != fingerprint_tensor(t.float())

    def test_bfloat16_views_exclude_unrelated_storage(self):
        t = torch.tensor([99., 1., 2., 88.], dtype=torch.bfloat16)
        assert fingerprint_tensor(t[1:3]) == fingerprint_tensor(torch.tensor([1., 2.], dtype=t.dtype))
        t[0], t[3] = 5., 7.
        assert fingerprint_tensor(t[1:3]) == fingerprint_tensor(torch.tensor([1., 2.], dtype=t.dtype))

    def test_noncontiguous_scalar_and_empty_bfloat16(self):
        t = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).T
        assert fingerprint_tensor(t) == fingerprint_tensor(t.contiguous())
        assert fingerprint_tensor(torch.tensor(1., dtype=t.dtype)) != fingerprint_tensor(torch.ones(1, dtype=t.dtype))
        assert fingerprint_tensor(torch.empty(0, dtype=t.dtype)) != fingerprint_tensor(torch.empty(0, 1, dtype=t.dtype))

    def test_bfloat16_preserves_signed_zero_bits(self):
        assert fingerprint_tensor(torch.tensor([0.], dtype=torch.bfloat16)) != fingerprint_tensor(torch.tensor([-0.], dtype=torch.bfloat16))

    def test_float8_exact_bits_and_dtype(self):
        a = torch.tensor([1., 2.], dtype=torch.float8_e4m3fn)
        b = torch.tensor([1., 2.], dtype=torch.float8_e5m2)
        assert fingerprint_tensor(a) == fingerprint_tensor(a.clone())
        assert fingerprint_tensor(a) != fingerprint_tensor(b)
        assert fingerprint_tensor(a) != fingerprint_tensor(torch.tensor([1., 3.], dtype=a.dtype))

    def test_float32_fingerprint_is_unchanged(self):
        assert fingerprint_tensor(torch.tensor([1., 2.], dtype=torch.float32)) == 'v2:f97f2a809e253731'

    def test_low_precision_trajectory_roundtrip_and_divergence(self, tmp_path):
        from resumecheck.compare import compare_trajectories
        for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2,
                      torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
            a, b = TrajectoryRecorder('a'), TrajectoryRecorder('b')
            a.record(1, loss=1., batch={'x': torch.tensor([1., 2.], dtype=dtype)})
            b.record(1, loss=1., batch={'x': torch.tensor([1., 3.], dtype=dtype)})
            path = tmp_path / 'trajectory.json'
            a.trajectory.save(path)
            loaded = Trajectory.load(path)
            assert compare_trajectories(a.trajectory, loaded).passed
            assert not compare_trajectories(loaded, b.trajectory).passed


class TestObjectArrayFingerprint:
    def test_mutable_object_array_is_rejected(self):
        import pytest
        payload = {"token": 1}
        array = np.empty(1, dtype=object)
        array[0] = payload
        with pytest.raises(TypeError, match="object-containing array"):
            fingerprint_tensor(array)
        payload["token"] = 2
        with pytest.raises(TypeError, match="object-containing array"):
            fingerprint_tensor(array)

    def test_nested_structured_object_field_is_rejected(self):
        import pytest
        array = np.zeros(2, dtype=[("nested", [("value", object)])])
        with pytest.raises(TypeError, match="object-containing array"):
            fingerprint_tensor({"batch": array})

    def test_zero_dimensional_and_empty_object_arrays_are_rejected(self):
        import pytest
        for array in (np.array({"token": 1}, dtype=object), np.empty(0, dtype=object)):
            with pytest.raises(TypeError, match="object-containing array"):
                fingerprint_tensor(array)

    def test_recorder_does_not_append_unverified_object_batch(self):
        import pytest
        recorder = TrajectoryRecorder()
        with pytest.raises(TypeError, match="object-containing array"):
            recorder.record(0, loss=1., batch=np.array(["a"], dtype=object))
        assert recorder.trajectory.steps == []

    def test_numeric_structured_arrays_remain_supported(self):
        array = np.array([(1, 2.)], dtype=[("token", "i4"), ("value", "f4")])
        assert fingerprint_tensor(array) == fingerprint_tensor(array.copy())
        changed = array.copy()
        changed["token"] = 2
        assert fingerprint_tensor(array) != fingerprint_tensor(changed)

    def test_strings_and_nested_native_containers_remain_supported(self):
        assert fingerprint_tensor(np.array(["a", "b"])) == fingerprint_tensor(np.array(["a", "b"]))
        assert fingerprint_tensor([{"token": 1}]) != fingerprint_tensor([{"token": 2}])


class TestOpaqueBatchFingerprint:
    def test_constant_repr_mutable_batch_is_rejected(self):
        import pytest
        class Batch:
            def __init__(self): self.tokens = [1]
            def __repr__(self): return "Batch()"
        batch = Batch()
        rec = TrajectoryRecorder()
        with pytest.raises(TypeError, match="unsupported batch value type Batch"):
            rec.record(0, batch=batch)
        batch.tokens[0] = 2
        with pytest.raises(TypeError):
            rec.record(0, batch=batch)
        assert rec.trajectory.steps == []

    def test_opaque_nested_batch_is_rejected(self):
        import pytest
        with pytest.raises(TypeError, match="unsupported batch value type object"):
            fingerprint_tensor({"tokens": [1, object()]})

    def test_broken_array_conversion_cannot_fall_back_to_repr(self):
        import pytest
        class BrokenArray:
            def tobytes(self): raise ValueError("cannot encode")
            def __repr__(self): return "constant"
        with pytest.raises(TypeError, match="unsupported batch value type BrokenArray"):
            fingerprint_tensor(BrokenArray())

    def test_repr_spoof_cannot_alias_builtin_scalar(self):
        import pytest
        class PretendInt:
            def __repr__(self): return "1"
        class StatefulInt(int):
            pass
        for value in (PretendInt(), StatefulInt(1)):
            with pytest.raises(TypeError, match="unsupported batch value type"):
                fingerprint_tensor(value)

    def test_supported_scalars_keep_existing_fingerprints(self):
        import hashlib
        for value in (None, False, True, -1, 0, 1, 2.5, complex(1, 2)):
            payload = repr(value).encode()
            expected = "v2:" + hashlib.sha256(b"repr" + str(len(payload)).encode() + b":" + payload).hexdigest()[:16]
            assert fingerprint_tensor(value) == expected

    def test_supported_nested_batch_remains_content_sensitive(self):
        first = {"tokens": [1, 2], "mask": (True, False), "score": 1.5, "label": "a"}
        second = {"tokens": [1, 3], "mask": (True, False), "score": 1.5, "label": "a"}
        assert fingerprint_tensor(first) != fingerprint_tensor(second)


class TestMaskedArrayRejection:
    def test_masked_array_collides_with_plain_filled_bytes(self):
        import pytest
        masked = np.ma.array([1., 2.], mask=[False, True])
        plain = masked.filled()
        assert masked.tobytes() == plain.tobytes()
        assert fingerprint_tensor(plain).startswith("v2:")
        with pytest.raises(TypeError, match="masked arrays"):
            fingerprint_tensor(masked)

    def test_hidden_payload_is_not_erased(self):
        import pytest
        for hidden in [2., 777.]:
            masked = np.ma.array([1., hidden], mask=[False, True])
            with pytest.raises(TypeError, match="masked arrays"):
                fingerprint_tensor(masked)

    def test_all_mask_forms_fail_closed(self):
        import pytest
        for value in [np.ma.array([1, 2]), np.ma.array([], mask=[]),
                      np.ma.array(3, mask=True), np.ma.masked,
                      np.ma.array([(1, 2.)], dtype=[("a", "i8"), ("b", "f8")],
                                  mask=[(False, True)])]:
            with pytest.raises(TypeError, match="masked arrays"):
                fingerprint_tensor(value)

    def test_nested_masked_value_is_not_accepted(self):
        import pytest
        with pytest.raises(TypeError, match="masked arrays"):
            fingerprint_tensor({"x": [np.ma.array([1], mask=[True])]})

    def test_explicit_mask_container_preserves_semantics(self):
        masked = np.ma.array([1., 2.], mask=[False, True])
        encoded = {"data": masked.data, "mask": np.ma.getmaskarray(masked),
                   "fill_value": float(masked.fill_value)}
        changed = dict(encoded, mask=np.array([False, False]))
        assert fingerprint_tensor(encoded) != fingerprint_tensor(changed)
        changed = dict(encoded, data=np.array([1., 777.]))
        assert fingerprint_tensor(encoded) != fingerprint_tensor(changed)
        changed = dict(encoded, fill_value=0.)
        assert fingerprint_tensor(encoded) != fingerprint_tensor(changed)
