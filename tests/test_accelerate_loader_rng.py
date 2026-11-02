"""Isolate DataLoader iterator seeds from model dropout on both own adapters."""
from dataclasses import asdict
import json
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('accelerate')
pytest.importorskip('torchdata')
from resumecheck.accelerate import check_accelerate, check_accelerate_skip_path
from resumecheck.artifacts import replay_comparison_artifacts

class Data(torch.utils.data.Dataset):
    def __len__(self):return 16
    def __getitem__(self, index):return {'x':torch.tensor([float(index)/16]),'y':torch.tensor([float(index)/8+1])}

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(1,8),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(8,1))
    def forward(self, x, y):return {'loss':(self.net(x)-y).square().mean()}

@pytest.mark.parametrize('seed', [42,123])
@pytest.mark.parametrize('kind', ['mid','boundary','skip'])
@pytest.mark.parametrize('isolate', [False,True])
def test_dropout_multi_epoch_checkpoint(tmp_path, seed, kind, isolate):
    check = check_accelerate_skip_path if kind=='skip' else check_accelerate
    resume = 8 if kind=='boundary' else 3
    kwargs = {'boundary_set_epoch_workaround':True} if kind=='boundary' else {'skip_path_set_epoch_workaround':True} if kind=='skip' else {}
    path = tmp_path/'evidence'
    report = check(Model, Data(), 19, resume, seed=seed, isolate_dataloader_rng=isolate,
                   record_model_state=True, artifacts_dir=path, **kwargs)
    assert report.passed == isolate, report.summary
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
    saved = json.loads((path/'resumed.json').read_text())
    assert len(saved['steps']) == 19
    assert saved['meta'].get('isolate_dataloader_rng',False) == isolate
    full = json.loads((path/'uninterrupted.json').read_text())
    assert full['meta'].get('isolate_dataloader_rng',False) == isolate
    if isolate:assert report.steps_compared == 19
    else:assert report.first_divergence.step >= resume+1

@pytest.mark.parametrize('check', [check_accelerate,check_accelerate_skip_path])
@pytest.mark.parametrize('value', [1,None,'yes'])
def test_invalid_isolation_flag_precedes_training(check, value):
    def forbidden():pytest.fail('must not train')
    with pytest.raises(TypeError,match='must be boolean'):
        check(forbidden,Data(),6,3,isolate_dataloader_rng=value)

@pytest.mark.parametrize('kind',['boundary','skip'])
def test_rng_isolation_does_not_hide_sampler_defect(kind):
    check = check_accelerate_skip_path if kind=='skip' else check_accelerate
    report = check(Model,Data(),19,3 if kind=='skip' else 8,seed=42,
                   isolate_dataloader_rng=True,record_model_state=True)
    assert not report.passed
    assert report.first_divergence.field == 'batch'
    assert report.first_divergence.step == 9

@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_explicit_false_preserves_default_trajectory(check,tmp_path):
    values=[]
    for name,kwargs in [('default',{}),('false',{'isolate_dataloader_rng':False})]:
        path=tmp_path/name
        report=check(Model,Data(),6,3,seed=42,artifacts_dir=path,**kwargs)
        values.append((asdict(report),json.loads((path/'uninterrupted.json').read_text()),
                       json.loads((path/'resumed.json').read_text())))
    assert values[0]==values[1]

@pytest.mark.parametrize('kind',['mid','boundary','skip'])
def test_private_loader_generator_is_used_without_changing_batch_content(tmp_path,kind):
    check = check_accelerate_skip_path if kind=='skip' else check_accelerate
    resume=8 if kind=='boundary' else 3
    kwargs={'boundary_set_epoch_workaround':True} if kind=='boundary' else {'skip_path_set_epoch_workaround':True} if kind=='skip' else {}
    values=[]
    for flag in (False,True):
        path=tmp_path/str(flag)
        check(Model,Data(),19,resume,seed=42,artifacts_dir=path,isolate_dataloader_rng=flag,**kwargs)
        values.append([[s['batch_fingerprint'] for s in json.loads((path/(name+'.json')).read_text())['steps']]
                       for name in ('uninterrupted','resumed')])
    assert values[0] == values[1]
