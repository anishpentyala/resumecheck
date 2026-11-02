"""Both real Accelerate checkpoint paths expose otherwise invisible state changes."""
import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('accelerate')
pytest.importorskip('torchdata')
from accelerate import Accelerator

from resumecheck.accelerate import check_accelerate, check_accelerate_skip_path
from resumecheck.artifacts import replay_comparison_artifacts


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


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('perturb', [None, 'parameter', 'buffer'])
def test_checkpoint_hidden_state_perturbation(tmp_path, monkeypatch, skip, record, perturb):
    load = Accelerator.load_state
    def changed(self, *args, **kwargs):
        result = load(self, *args, **kwargs)
        if perturb:
            model = self.unwrap_model(self._models[0])
            with torch.no_grad():
                if perturb == 'parameter':model.weight.add_(torch.tensor([1., -1.]))
                else:model.running.add_(1.)
        return result
    monkeypatch.setattr(Accelerator, 'load_state', changed)
    path = tmp_path / 'evidence'
    check = check_accelerate_skip_path if skip else check_accelerate
    report = check(Model, Data(), 6, 3, seed=42, record_model_state=record, artifacts_dir=path)
    assert report.passed == (not record or perturb is None), report.summary
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['schema_version'] == (2 if record else 1)
    for name in ('uninterrupted', 'resumed'):
        steps = json.loads((path / (name + '.json')).read_text())['steps']
        assert len(steps) == 6
        assert [s['step'] for s in steps] == list(range(1, 7))
        assert all((s.get('model_fingerprint') is not None) == record for s in steps)
    if record and perturb:
        assert report.first_divergence.field == 'model_state'
        assert report.first_divergence.step == 4


@pytest.mark.parametrize('check', [check_accelerate, check_accelerate_skip_path])
@pytest.mark.parametrize('value', [1, None, 'yes'])
def test_invalid_flag_before_training(check, value, tmp_path):
    def forbidden():pytest.fail('must not train')
    path = tmp_path / 'evidence'
    with pytest.raises(TypeError, match='must be boolean'):
        check(forbidden, Data(), 6, 3, record_model_state=value, artifacts_dir=path)
    assert not path.exists()


@pytest.mark.parametrize('check', [check_accelerate, check_accelerate_skip_path])
def test_unsupported_state_fails_without_bundle(check, tmp_path):
    class CustomState(Model):
        def get_extra_state(self):return object()
    path = tmp_path / 'evidence'
    with pytest.raises(TypeError, match='unsupported batch value'):
        check(CustomState, Data(), 6, 3, record_model_state=True, artifacts_dir=path)
    assert not path.exists()


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('record', [False, True])
def test_multi_epoch_workaround_preserves_state(tmp_path, skip, record):
    check = check_accelerate_skip_path if skip else check_accelerate
    kwargs = {'skip_path_set_epoch_workaround': True} if skip else {'boundary_set_epoch_workaround': True}
    report = check(Model, Data(), 19, 3 if skip else 8, seed=42,
                   artifacts_dir=tmp_path / 'evidence', record_model_state=record, **kwargs)
    assert report.passed, report.summary
    assert asdict(replay_comparison_artifacts(tmp_path / 'evidence')) == asdict(report)


def test_opt_in_fingerprinting_does_not_change_training(tmp_path):
    for check in (check_accelerate, check_accelerate_skip_path):
        saved = []
        for flag in (False, True):
            path = tmp_path / (check.__name__ + str(flag))
            assert check(Model, Data(), 6, 3, seed=123, record_model_state=flag, artifacts_dir=path).passed
            steps = json.loads((path / 'uninterrupted.json').read_text())['steps']
            saved.append([{k:v for k,v in s.items() if k != 'model_fingerprint'} for s in steps])
        assert saved[0] == saved[1]


@pytest.mark.parametrize('check', [check_accelerate, check_accelerate_skip_path])
def test_model_state_hashing_does_not_change_rng_or_parameters(check, monkeypatch):
    import resumecheck.accelerate as adapter
    fingerprint = adapter.fingerprint_tensor
    calls = []
    def inspected(value):
        if isinstance(value, dict) and 'active' in value:
            rng = torch.get_rng_state().clone()
            snapshot = {key:item.clone() for key,item in value.items()}
            result = fingerprint(value)
            assert torch.equal(rng, torch.get_rng_state())
            assert all(torch.equal(snapshot[key], item) for key,item in value.items())
            calls.append(result)
            return result
        return fingerprint(value)
    monkeypatch.setattr(adapter, 'fingerprint_tensor', inspected)
    assert check(Model, Data(), 6, 3, seed=42, record_model_state=True).passed
    assert len(calls) == 12


@pytest.mark.parametrize('check', [check_accelerate, check_accelerate_skip_path])
def test_default_never_fingerprints_model_state(check, monkeypatch):
    import resumecheck.accelerate as adapter
    fingerprint = adapter.fingerprint_tensor
    def checked(value):
        if isinstance(value, dict) and 'active' in value:
            pytest.fail('default must not copy or hash model state')
        return fingerprint(value)
    monkeypatch.setattr(adapter, 'fingerprint_tensor', checked)
    assert check(Model, Data(), 6, 3, seed=42).passed
