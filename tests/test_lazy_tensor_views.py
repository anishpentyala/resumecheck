"""Fingerprint logical values of lazy conjugate/negative tensor views."""

import pytest

torch = pytest.importorskip("torch")
from resumecheck.recorder import fingerprint_tensor, TrajectoryRecorder


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("route", ["contiguous", "transpose", "slice"])
def test_lazy_conjugate_matches_materialized_values(dtype, route):
    base = torch.tensor([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=dtype)
    if route == "transpose":
        base = base.T
    elif route == "slice":
        base = base[:, 1:]
    view = base.conj()
    saved = base.clone()
    assert view.is_conj()
    assert fingerprint_tensor(view) == fingerprint_tensor(view.resolve_conj())
    assert fingerprint_tensor(view) != fingerprint_tensor(base)
    torch.testing.assert_close(base, saved, rtol=0, atol=0)
    assert view.is_conj()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_lazy_negative_matches_materialized_values(dtype):
    base = torch.tensor([1.0, -2.0, 0.0], dtype=dtype)
    view = torch._neg_view(base)
    assert view.is_neg()
    assert fingerprint_tensor(view) == fingerprint_tensor(view.resolve_neg())
    assert fingerprint_tensor(view) != fingerprint_tensor(base)
    assert view.is_neg()


def test_nested_recorder_supports_lazy_views():
    base = torch.tensor([1 + 2j])
    view = torch._neg_view(base.conj())
    rec = TrajectoryRecorder("views")
    rec.record(1, loss=1.0, batch={"view": view})
    assert rec.trajectory.steps[0].batch_fingerprint == fingerprint_tensor(
        {"view": view.resolve_conj().resolve_neg()}
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_negative_view_preserves_raw_dtype_path(dtype):
    base = torch.tensor([1.0, -2.0], dtype=dtype)
    view = torch._neg_view(base)
    assert fingerprint_tensor(view) == fingerprint_tensor(view.resolve_neg())
    assert fingerprint_tensor(view) != fingerprint_tensor(base)


def test_resolving_lazy_bits_does_not_consume_rng_or_mutate_storage():
    base = torch.tensor([1 + 2j, 3 + 4j])
    view = torch._neg_view(base.conj())
    state = torch.get_rng_state().clone()
    before = base.clone()
    fingerprint_tensor(view)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(base, before)
    assert view.is_conj() and view.is_neg()
