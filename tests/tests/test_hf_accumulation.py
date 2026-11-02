"""Actual Trainer optimizer windows, including incomplete final epoch windows."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import TrainingArguments
from resumecheck.hf import check_hf, _make_trainer_cls
import resumecheck.hf as hf
from resumecheck.recorder import TrajectoryRecorder, fingerprint_tensor
from test_hf_parity import TinyDataset, TinyModel


@pytest.mark.parametrize('accumulation', [1, 2, 3, 4])
@pytest.mark.parametrize('size', [8, 14])
@pytest.mark.parametrize('resume_at', [2, 3])
def test_actual_hf_accumulated_resume(accumulation, size, resume_at):
    report = check_hf(TinyModel, TinyDataset(size), max_steps=6, resume_at=resume_at, seed=42,
                      training_args_kwargs={'gradient_accumulation_steps': accumulation})
    assert report.passed, report.summary
    assert report.steps_compared == 6


def trainer_and_recorder(tmp_path):
    rec = TrajectoryRecorder()
    trainer = _make_trainer_cls(rec)(model=TinyModel(), args=TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=True))
    return trainer, rec


def flush(trainer, step):
    trainer.callback_handler.callbacks[-1].on_step_end(trainer.args, SimpleNamespace(global_step=step), trainer.control)


def test_window_retains_every_microbatch_and_mean_loss(tmp_path):
    trainer, rec = trainer_and_recorder(tmp_path)
    trainer._rc_recording = True
    batches = [{'x': torch.tensor([[float(i)]]), 'y': torch.zeros(1, 1)} for i in [1, 2, 3]]
    losses = [float(trainer.compute_loss(trainer.model, b).detach()) for b in batches]
    assert rec.trajectory.steps == []
    flush(trainer, 7)
    row = rec.trajectory.steps[0]
    assert row.step == 7
    assert row.loss == pytest.approx(sum(losses) / 3)
    assert row.batch_fingerprint == fingerprint_tensor([fingerprint_tensor(b) for b in batches])
    assert trainer._rc_fingerprints == trainer._rc_losses == []


def test_evaluation_does_not_enter_training_window(tmp_path):
    trainer, rec = trainer_and_recorder(tmp_path)
    trainer.compute_loss(trainer.model, {'x': torch.ones(1, 1), 'y': torch.zeros(1, 1)})
    assert rec.trajectory.steps == []
    assert getattr(trainer, '_rc_fingerprints', []) == []


def test_single_microbatch_fingerprint_stays_compatible(tmp_path):
    trainer, rec = trainer_and_recorder(tmp_path)
    trainer._rc_recording = True
    batch = {'x': torch.ones(1, 1), 'y': torch.zeros(1, 1)}
    trainer.compute_loss(trainer.model, batch)
    flush(trainer, 1)
    assert rec.trajectory.steps[0].batch_fingerprint == fingerprint_tensor(batch)


def test_incomplete_window_fails_closed(tmp_path):
    trainer, rec = trainer_and_recorder(tmp_path)
    trainer._rc_recording = True
    trainer.compute_loss(trainer.model, {'x': torch.ones(1, 1), 'y': torch.zeros(1, 1)})
    with pytest.raises(ValueError, match='incomplete optimizer window'):
        trainer.callback_handler.on_train_end(trainer.args, trainer.state, trainer.control)
    assert not rec.trajectory.steps


def test_optimizer_step_without_training_inputs_fails_closed(tmp_path):
    trainer, rec = trainer_and_recorder(tmp_path)
    with pytest.raises(ValueError, match='no recorded training microbatches'):
        flush(trainer, 1)


@pytest.mark.parametrize('observed', [[], [1], [1, 2], [1, 2, 4], [1, 2, 3, 4]])
def test_truncated_or_wrong_phase_cannot_verify(observed):
    rec = TrajectoryRecorder()
    for step in observed:
        rec.record(step, loss=1.)
    with pytest.raises(ValueError, match='requested optimizer steps'):
        hf._require_phase_steps(rec.trajectory, 1, 3, 'uninterrupted')


@pytest.mark.parametrize('mutate', [False, True])
@pytest.mark.parametrize('changing', [False, True])
def test_accumulated_nonfinal_microbatch_is_not_lost(mutate, changing):
    class Data(torch.utils.data.Dataset):
        calls = 0
        def __len__(self): return 24
        def __getitem__(self, i):
            type(self).calls += 1
            # Alter only the first microbatch in each three-microbatch window.
            change = changing and ((type(self).calls - 1) // 2) % 3 == 0
            return {'x': torch.tensor([float(type(self).calls if change else 1)])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, x):
            if mutate: x.zero_()
            return {'loss': self.weight.square()}
    report = check_hf(Model, Data(), max_steps=4, resume_at=2,
                      training_args_kwargs={'gradient_accumulation_steps': 3})
    assert report.passed == (not changing), report.summary
    if changing:
        assert report.first_divergence.field == 'batch'
        assert report.first_divergence.step == 1
    else:
        assert report.steps_compared == 4


def test_training_exception_cannot_leave_recording_enabled(tmp_path):
    trainer, rec = trainer_and_recorder(tmp_path)
    class Broken(torch.nn.Module):
        def forward(self, **kwargs): raise RuntimeError('deliberate')
    with pytest.raises(RuntimeError, match='deliberate'):
        trainer.training_step(Broken(), {'x': torch.ones(1, 1), 'y': torch.zeros(1, 1)})
    assert not trainer._rc_recording
    assert not rec.trajectory.steps


def test_actual_evaluation_does_not_pollute_training_windows(tmp_path):
    rec = TrajectoryRecorder()
    trainer = _make_trainer_cls(rec)(model=TinyModel(),train_dataset=TinyDataset(8),eval_dataset=TinyDataset(8),
        args=TrainingArguments(output_dir=str(tmp_path),max_steps=4,gradient_accumulation_steps=3,
            per_device_train_batch_size=2,use_cpu=True,report_to=[],disable_tqdm=True,
            save_strategy='no',eval_strategy='steps',eval_steps=1))
    trainer.train()
    assert [row.step for row in rec.trajectory.steps] == [1, 2, 3, 4]
    assert not trainer._rc_fingerprints


@pytest.mark.parametrize('phase', ['uninterrupted', 'pre-crash', 'post-resume'])
def test_actual_phase_completion_guard(monkeypatch, phase):
    original = hf._make_trainer_cls
    calls = 0
    def factory(rec):
        nonlocal calls
        calls += 1
        cls = original(rec)
        if calls == {'uninterrupted':1,'pre-crash':2,'post-resume':3}[phase]:
            class Stopped(cls):
                def train(self, *args, **kwargs): return None
            return Stopped
        return cls
    monkeypatch.setattr(hf, '_make_trainer_cls', factory)
    with pytest.raises(ValueError, match=f'HF {phase} phase'):
        check_hf(TinyModel, TinyDataset(8),max_steps=4,resume_at=2)
