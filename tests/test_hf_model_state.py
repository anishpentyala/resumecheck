"""Real Trainer checks: equal losses may hide changed model state."""
import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import Trainer

from resumecheck.hf import check_hf
from resumecheck.artifacts import replay_comparison_artifacts
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import StepRecord, Trajectory, fingerprint_tensor


class Data(torch.utils.data.Dataset):
    def __len__(self):return 16
    def __getitem__(self, index):return {'x': torch.tensor([float(index)])}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -1.]))
        self.active = torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer('running', torch.tensor(1.))
    def forward(self, x):
        return {'loss': self.weight.sum().square() + self.active.square()}


@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('perturb', [None, 'parameter', 'buffer'])
@pytest.mark.parametrize('accumulation', [1, 3])
def test_actual_checkpoint_perturbation(tmp_path, monkeypatch, record, perturb, accumulation):
    load = Trainer._load_from_checkpoint
    def changed(self, *args, **kwargs):
        result = load(self, *args, **kwargs)
        if perturb:
            with torch.no_grad():
                if perturb == 'parameter':self.model.weight.add_(torch.tensor([1., -1.]))
                else:self.model.running.add_(1.)
        return result
    monkeypatch.setattr(Trainer, '_load_from_checkpoint', changed)
    path = tmp_path / 'evidence'
    report = check_hf(Model, Data(), max_steps=4, resume_at=2, seed=42,
                      record_model_state=record, artifacts_dir=path,
                      training_args_kwargs={'gradient_accumulation_steps': accumulation})
    assert report.passed == (not record or perturb is None)
    replay = replay_comparison_artifacts(path)
    assert asdict(replay) == asdict(report)
    if record and perturb:
        assert report.first_divergence.field == 'model_state'
        assert report.first_divergence.step == 3
    saved = json.loads((path / 'resumed.json').read_text())
    assert all((step.get('model_fingerprint') is not None) == record for step in saved['steps'])


@pytest.mark.parametrize('left,right', [(None, 'v2:abc'), ('v2:abc', None)])
def test_one_sided_state_is_unverified(left, right):
    a = Trajectory('a', [StepRecord(1, loss=1., model_fingerprint=left)])
    b = Trajectory('b', [StepRecord(1, loss=1., model_fingerprint=right)])
    assert not compare_trajectories(a,b).passed


def test_state_only_and_exact_bits():
    a = fingerprint_tensor({'weight': torch.tensor([1., -1.])})
    b = fingerprint_tensor({'weight': torch.tensor([-1., 1.])})
    assert a != b
    left = Trajectory('a', [StepRecord(1, model_fingerprint=a)])
    right = Trajectory('b', [StepRecord(1, model_fingerprint=b)])
    result = compare_trajectories(left,right)
    assert not result.passed and result.first_divergence.field == 'model_state'
    assert compare_trajectories(left,left).passed


def test_default_json_remains_legacy_readable(tmp_path):
    from resumecheck.recorder import TrajectoryRecorder
    rec = TrajectoryRecorder('legacy')
    rec.record(1, loss=1., batch=[1])
    path = tmp_path / 'legacy.json'
    rec.trajectory.save(path)
    assert 'model_fingerprint' not in json.loads(path.read_text())['steps'][0]


def test_custom_unsupported_model_state_fails_instead_of_ignoring():
    class CustomState(Model):
        def get_extra_state(self):
            return object()
    with pytest.raises(TypeError, match='unsupported batch value'):
        check_hf(CustomState, Data(), 4, 2, record_model_state=True)


def test_fingerprinting_does_not_mutate_parameters_or_rng():
    model = Model()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    fingerprint_tensor(model.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    for key,value in model.state_dict().items():
        assert torch.equal(value,state[key])


@pytest.mark.parametrize('value', [1, None, 'yes'])
def test_invalid_record_model_state_rejected(value):
    with pytest.raises(TypeError, match='must be boolean'):
        check_hf(Model, Data(), 4, 2, record_model_state=value)


def test_state_evidence_cannot_disappear_in_both_runs_at_a_step():
    records = [StepRecord(1, loss=1., model_fingerprint='v2:abc'),
               StepRecord(2, loss=1.)]
    report = compare_trajectories(Trajectory('a', records), Trajectory('b', records))
    assert not report.passed
    assert 'missing in both runs at step 2' in report.summary


def test_explicit_model_state_opt_out_limits_comparison_scope():
    a = Trajectory('a', [StepRecord(1, loss=1., model_fingerprint='v2:abc')])
    b = Trajectory('b', [StepRecord(1, loss=1., model_fingerprint='v2:def')])
    assert not compare_trajectories(a,b).passed
    assert compare_trajectories(a,b,check_model_state=False).passed
