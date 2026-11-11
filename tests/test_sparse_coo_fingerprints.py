"""Exact stored COO representation, not dense mathematical equivalence."""
import pytest

torch = pytest.importorskip('torch')
from resumecheck import TrajectoryRecorder, compare_trajectories
from resumecheck.recorder import fingerprint_tensor


def sparse(indices=None, values=None, size=(4,), **kwargs):
    return torch.sparse_coo_tensor(
        [[0, 2]] if indices is None else indices,
        [1., 2.] if values is None else values,
        size, check_invariants=True, **kwargs)


@pytest.mark.parametrize('coalesced', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.int64,
                                   torch.bfloat16, torch.complex64])
def test_sparse_clone_preserves_fingerprint_and_input(coalesced, dtype):
    x = sparse(dtype=dtype)
    if coalesced:
        x = x.coalesce()
    indices, values = x._indices().clone(), x._values().clone()
    assert fingerprint_tensor(x) == fingerprint_tensor(x.clone())
    assert torch.equal(x._indices(), indices)
    assert torch.equal(x._values(), values)
    assert x.is_coalesced() == coalesced


@pytest.mark.parametrize('other', [
    lambda: sparse(values=[1., 3.]),
    lambda: sparse(indices=[[0, 3]]),
    lambda: sparse(size=(5,)),
    lambda: sparse(dtype=torch.float64),
    lambda: sparse().to_dense(),
    lambda: sparse().coalesce(),
])
def test_sparse_representation_changes_are_observed(other):
    assert fingerprint_tensor(sparse()) != fingerprint_tensor(other())


def test_duplicate_entries_and_order_are_not_silently_normalized():
    x = sparse(indices=[[1, 1]], values=[3., 4.])
    y = sparse(indices=[[1, 1]], values=[4., 3.])
    assert torch.equal(x.to_dense(), y.to_dense())
    assert torch.equal(x.to_dense(), x.coalesce().to_dense())
    assert len({fingerprint_tensor(x), fingerprint_tensor(y),
                fingerprint_tensor(x.coalesce())}) == 3


def test_hybrid_sparse_dense_dimensions_and_nested_containers():
    x = sparse(indices=[[0, 2]], values=[[1., 2.], [3., 4.]], size=(4, 2))
    y = sparse(indices=[[0, 2]], values=[[1., 2.], [3., 5.]], size=(4, 2))
    assert fingerprint_tensor({'state': [x]}) == fingerprint_tensor({'state': [x.clone()]})
    assert fingerprint_tensor({'state': [x]}) != fingerprint_tensor({'state': [y]})


@pytest.mark.parametrize('shape', [(0,), (5,), (10**12,)])
def test_empty_and_huge_extent_do_not_densify_or_coalesce(shape, monkeypatch):
    x = sparse(indices=torch.empty((1, 0), dtype=torch.int64), values=[], size=shape)
    def forbidden(*args, **kwargs):
        pytest.fail('fingerprinting must not densify/coalesce')
    monkeypatch.setattr(torch.Tensor, 'to_dense', forbidden)
    monkeypatch.setattr(torch.Tensor, 'coalesce', forbidden)
    assert fingerprint_tensor(x).startswith('v2:')


def test_nonempty_huge_extent_does_not_densify_or_coalesce(monkeypatch):
    x = sparse(size=(10**12,))
    def forbidden(*args, **kwargs):
        pytest.fail('fingerprinting must not densify/coalesce')
    monkeypatch.setattr(torch.Tensor, 'to_dense', forbidden)
    monkeypatch.setattr(torch.Tensor, 'coalesce', forbidden)
    assert fingerprint_tensor(x) == fingerprint_tensor(x.clone())


@pytest.mark.parametrize('field', ['batch', 'model', 'optimizer'])
@pytest.mark.parametrize('changed', [False, True])
def test_sparse_content_reaches_comparison(field, changed):
    a, b = TrajectoryRecorder('a'), TrajectoryRecorder('b')
    x, y = sparse(), sparse(values=[1., 3.] if changed else [1., 2.])
    if field == 'batch':
        a.record(0, loss=1, batch=x); b.record(0, loss=1, batch=y)
    else:
        a.record(0, loss=1); b.record(0, loss=1)
        name = field + '_fingerprint'
        setattr(a.trajectory.steps[0], name, fingerprint_tensor({'state': x}))
        setattr(b.trajectory.steps[0], name, fingerprint_tensor({'state': y}))
    report = compare_trajectories(a.trajectory, b.trajectory)
    assert report.passed is not changed
    if changed:
        assert report.first_divergence.field == ('batch' if field == 'batch' else field+'_state')


def test_requires_grad_supported_without_mutating_source():
    x = sparse(requires_grad=True)
    assert fingerprint_tensor(x) == fingerprint_tensor(x.detach())
    assert x.requires_grad


def test_compressed_sparse_layout_still_fails_atomically():
    x = torch.tensor([[1., 0.], [0., 2.]]).to_sparse_csr()
    rec = TrajectoryRecorder()
    rec.record(0, loss=1)
    with pytest.raises((RuntimeError, TypeError)):
        rec.record(1, loss=1, batch=x)
    assert len(rec.trajectory.steps) == 1


def test_dense_known_fingerprint_remains_unchanged():
    assert fingerprint_tensor(torch.tensor([1., 2.], dtype=torch.float32)) == 'v2:f97f2a809e253731'


@pytest.mark.parametrize('restore_optimizer', [True, False])
def test_actual_sparse_sgd_momentum_checkpoint_replay(tmp_path, restore_optimizer):
    """Zero LR isolates missing sparse momentum from equal losses/weights."""
    from resumecheck.artifacts import save_comparison_artifacts, replay_comparison_artifacts
    batches = [torch.tensor([0, 1, 0]), torch.tensor([2, 1]),
               torch.tensor([0, 3]), torch.tensor([1, 2, 1])]
    def fresh():
        model = torch.nn.Embedding(4, 2, sparse=True)
        with torch.no_grad():
            model.weight.copy_(torch.arange(8).reshape(4, 2).float()/10)
        return model, torch.optim.SGD(model.parameters(), lr=0, momentum=.9)
    def update(model, opt, rec, i):
        opt.zero_grad()
        loss = model(batches[i]).square().sum()
        loss.backward()
        opt.step()
        momentum = opt.state[model.weight]['momentum_buffer']
        assert momentum.layout == torch.sparse_coo
        rec.record(i, loss=loss.item(), batch=batches[i])
        rec.trajectory.steps[-1].model_fingerprint = fingerprint_tensor(model.state_dict())
        rec.trajectory.steps[-1].optimizer_fingerprint = fingerprint_tensor(opt.state_dict())
    full, resumed = TrajectoryRecorder('full'), TrajectoryRecorder('resumed')
    model, opt = fresh()
    for i in range(4):
        update(model, opt, full, i)
    model, opt = fresh()
    for i in range(2):
        update(model, opt, resumed, i)
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    model, opt = fresh()
    model.load_state_dict(state['model'])
    if restore_optimizer:
        opt.load_state_dict(state['optimizer'])
    for i in range(2, 4):
        update(model, opt, resumed, i)
    report = compare_trajectories(full.trajectory, resumed.trajectory, rtol=0, atol=0)
    assert report.passed is restore_optimizer
    if not restore_optimizer:
        assert report.first_divergence.step == 2
        assert report.first_divergence.field == 'optimizer_state'
        assert compare_trajectories(full.trajectory, resumed.trajectory,
                                    check_optimizer_state=False, rtol=0, atol=0).passed
    destination = tmp_path/'evidence'
    save_comparison_artifacts(destination, full.trajectory, resumed.trajectory,
                              report, rtol=0, atol=0)
    assert replay_comparison_artifacts(destination).passed is restore_optimizer
