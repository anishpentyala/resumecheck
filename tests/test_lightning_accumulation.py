"""Record optimizer windows without dropping intermediate microbatches."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('lightning.pytorch')
from resumecheck.lightning import _make_callback_cls
from resumecheck.recorder import TrajectoryRecorder
from resumecheck.compare import compare_trajectories


def trajectory(batches, losses, advances, start=0):
    recorder = TrajectoryRecorder('window')
    cb = _make_callback_cls()(recorder)
    trainer = SimpleNamespace(global_step=start)
    module = torch.nn.Linear(1, 1)
    cb.on_train_start(trainer, module)
    for i, (batch, loss, step) in enumerate(zip(batches, losses, advances)):
        batch = torch.tensor([batch])
        cb.on_train_batch_start(trainer, module, batch, i)
        trainer.global_step = step
        cb.on_train_batch_end(trainer, module, {'loss': torch.tensor(loss)}, batch, i)
    return recorder.trajectory


def test_accumulation_records_once_per_optimizer_step():
    t = trajectory([1, 2, 3, 4, 5], [1., 3., 5., 7., 9.], [0, 1, 1, 2, 3])
    assert [r.step for r in t.steps] == [1, 2, 3]
    assert [r.loss for r in t.steps] == [2., 6., 9.]
    assert compare_trajectories(t, t).passed


def test_nonfinal_microbatch_changes_are_not_dropped():
    t = trajectory([1, 2], [1., 3.], [0, 1])
    changed = trajectory([9, 2], [1., 3.], [0, 1])
    assert len(t.steps) == len(changed.steps) == 1
    report = compare_trajectories(t, changed)
    assert not report.passed
    assert report.first_divergence.field == 'batch'


def test_nonfinal_microbatch_loss_is_compared():
    t = trajectory([1, 2], [1., 3.], [0, 1])
    changed = trajectory([1, 2], [7., 3.], [0, 1])
    assert len(t.steps) == len(changed.steps) == 1
    report = compare_trajectories(t, changed)
    assert not report.passed
    assert report.first_divergence.field == 'loss'


def test_resume_starts_after_restored_optimizer_step():
    t = trajectory([1, 2], [1., 3.], [3, 4], start=3)
    assert [r.step for r in t.steps] == [4]


def test_unknown_window_loss_does_not_become_a_number():
    recorder = TrajectoryRecorder()
    cb = _make_callback_cls()(recorder)
    trainer = SimpleNamespace(global_step=0)
    module = torch.nn.Linear(1, 1)
    cb.on_train_start(trainer, module)
    for i in range(2):
        b = torch.tensor([i])
        cb.on_train_batch_start(trainer, module, b, i)
        trainer.global_step = i
        cb.on_train_batch_end(trainer, module, None if i == 0 else {'loss': torch.tensor(2.)}, b, i)
    assert recorder.trajectory.steps[0].loss is None


def test_multiple_optimizer_steps_per_batch_are_rejected():
    with pytest.raises(ValueError, match='one optimizer step'):
        trajectory([1], [1.], [2])


@pytest.mark.parametrize('accumulation', [2, 3])
def test_actual_accumulated_training_resume_parity(accumulation):
    from test_lightning_parity import TinyModel, StatefulDM
    from resumecheck.lightning import check_lightning
    report = check_lightning(
        TinyModel, StatefulDM, max_steps=6, resume_at=3, seed=42,
        trainer_kwargs={'accumulate_grad_batches': accumulation},
    )
    assert report.passed, report.summary
    assert report.steps_compared == 6


def test_incomplete_optimizer_window_is_rejected():
    recorder = TrajectoryRecorder()
    cb = _make_callback_cls()(recorder)
    trainer = SimpleNamespace(global_step=0)
    module = torch.nn.Linear(1, 1)
    cb.on_train_start(trainer, module)
    b = torch.tensor([1])
    cb.on_train_batch_start(trainer, module, b, 0)
    cb.on_train_batch_end(trainer, module, {'loss': torch.tensor(1.)}, b, 0)
    with pytest.raises(ValueError, match='incomplete optimizer window'):
        cb.on_train_end(trainer, module)
