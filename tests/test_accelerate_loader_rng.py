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


class MixedRNGModel(torch.nn.Module):
    def __init__(self, source='both'):
        import random
        import numpy as np
        super().__init__()
        value=(random.random() if source in ('python','both') else 0.)
        value+=(float(np.random.random()) if source in ('numpy','both') else 0.)
        self.weight=torch.nn.Parameter(torch.tensor([value]))
    def forward(self,x,y):return {'loss':(x*self.weight-y).square().mean()}


@pytest.mark.parametrize('kind',['mid','boundary','skip'])
@pytest.mark.parametrize('source',['python','numpy','both'])
def test_phase_initialization_seeds_all_global_rngs(kind,source,tmp_path):
    import random
    import numpy as np
    random.seed(987);np.random.seed(654)
    check=check_accelerate_skip_path if kind=='skip' else check_accelerate
    resume=8 if kind=='boundary' else 3
    kwargs={'boundary_set_epoch_workaround':True} if kind=='boundary' else {'skip_path_set_epoch_workaround':True} if kind=='skip' else {}
    report=check(lambda:MixedRNGModel(source),Data(),19,resume,seed=42,
                 isolate_dataloader_rng=True,record_model_state=True,
                 artifacts_dir=tmp_path/'evidence',**kwargs)
    assert report.passed,report.summary
    assert report.steps_compared==19
    assert asdict(replay_comparison_artifacts(tmp_path/'evidence'))==asdict(report)


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_same_seed_ignores_ambient_python_numpy_state(check,tmp_path):
    import random
    import numpy as np
    results=[]
    for label,ambient in [('first',12),('second',765)]:
        random.seed(ambient);np.random.seed(ambient)
        path=tmp_path/label
        report=check(MixedRNGModel,Data(),6,3,seed=7,isolate_dataloader_rng=True,
                     record_model_state=True,artifacts_dir=path)
        assert report.passed,report.summary
        results.append(json.loads((path/'uninterrupted.json').read_text()))
    assert results[0]==results[1]


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_global_seed_still_changes_experiment(check,tmp_path):
    initial=[]
    for seed in [7,42]:
        path=tmp_path/str(seed)
        report=check(MixedRNGModel,Data(),6,3,seed=seed,isolate_dataloader_rng=True,
                     record_model_state=True,artifacts_dir=path)
        assert report.passed,report.summary
        initial.append(json.loads((path/'uninterrupted.json').read_text())['steps'][0]['model_fingerprint'])
    assert initial[0]!=initial[1]


class RandomTransformData(Data):
    def __getitem__(self,index):
        import random
        import numpy as np
        row=super().__getitem__(index)
        row['x']+=random.random()/10+float(np.random.random())/10
        return row


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_random_transform_resume_limit_stays_visible(check,tmp_path):
    report=check(MixedRNGModel,RandomTransformData(),6,3,seed=42,
                 isolate_dataloader_rng=True,record_model_state=True,
                 artifacts_dir=tmp_path/'evidence')
    # Loader prefetch/skip may consume extra main-process transform RNG draws.
    # Global phase initialization is fixed, not arbitrary transform resume.
    assert not report.passed
    assert report.first_divergence.step==4
    assert not replay_comparison_artifacts(tmp_path/'evidence').passed


class ForwardRNGModel(MixedRNGModel):
    def forward(self,x,y):
        import random
        import numpy as np
        factor=random.random()+float(np.random.random())
        return {'loss':(x*self.weight*factor-y).square().mean()}


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_forward_python_numpy_rng_is_restored(check):
    report=check(ForwardRNGModel,Data(),6,3,seed=42,
                 isolate_dataloader_rng=True,record_model_state=True)
    assert report.passed,report.summary


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_broken_restored_python_numpy_rng_still_diverges(check,monkeypatch):
    import random
    import numpy as np
    from accelerate import Accelerator
    original=Accelerator.load_state
    def corrupt(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        random.seed(999);np.random.seed(999)
        return result
    monkeypatch.setattr(Accelerator,'load_state',corrupt)
    report=check(ForwardRNGModel,Data(),6,3,seed=42,
                 isolate_dataloader_rng=True,record_model_state=True)
    assert not report.passed
    assert report.first_divergence.step==4
