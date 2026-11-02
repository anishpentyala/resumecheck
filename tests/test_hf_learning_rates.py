"""Actual Trainer scheduler restore can be wrong while loss stays identical."""
import json
from dataclasses import asdict
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import Trainer
from resumecheck.hf import check_hf, _optimizer_rates
from resumecheck.artifacts import replay_comparison_artifacts


class Data(torch.utils.data.Dataset):
    def __len__(self): return 24
    def __getitem__(self,i): return {'x':torch.tensor([float(i)])}


class ConstantLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x): return {'loss':self.weight*0+1}


@pytest.mark.parametrize('record',[False,True])
@pytest.mark.parametrize('omit',[False,True])
@pytest.mark.parametrize('accumulation',[1,3])
def test_real_scheduler_restore_with_constant_loss(tmp_path,monkeypatch,record,omit,accumulation):
    load=Trainer._load_optimizer_and_scheduler
    def restore(self,checkpoint):
        if not omit or not checkpoint:
            return load(self,checkpoint)
        restore_scheduler = self.lr_scheduler.load_state_dict
        self.lr_scheduler.load_state_dict = lambda state: None
        try:
            return load(self,checkpoint)  # optimizer restoration still runs
        finally:
            self.lr_scheduler.load_state_dict = restore_scheduler
    monkeypatch.setattr(Trainer,'_load_optimizer_and_scheduler',restore)
    path=tmp_path/'evidence'
    report=check_hf(ConstantLoss,Data(),max_steps=6,resume_at=3,
                    record_learning_rates=record,artifacts_dir=path,
                    training_args_kwargs={'gradient_accumulation_steps':accumulation})
    assert report.passed==(not record or not omit),report.summary
    assert asdict(replay_comparison_artifacts(path))==asdict(report)
    if record and omit:
        assert report.first_divergence.field.startswith('extras.lr.group')
        assert report.first_divergence.step==4
    manifest=json.loads((path/'manifest.json').read_text())
    rows=json.loads((path/'uninterrupted.json').read_text())['steps']
    if record:
        assert manifest['schema_version']==3
        assert manifest['comparison']['check_extras']==['lr.group0','lr.group1']
        assert len(rows)==6
        assert all(row['loss']==1 for row in rows)
        assert rows[0]['extras']['lr.group0']>rows[1]['extras']['lr.group0']
    else:
        assert manifest['schema_version']==1
        assert all(row['extras']=={} for row in rows)


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_flag_strict_boolean(value):
    with pytest.raises(TypeError,match='must be boolean'):
        check_hf(ConstantLoss,Data(),4,2,record_learning_rates=value)


@pytest.mark.parametrize('value',[None,True,'1',float('nan'),float('inf'),-1,torch.tensor([.1]),torch.tensor(True)])
def test_invalid_group_rate_fails(value):
    with pytest.raises(ValueError,match='finite nonnegative scalar'):
        _optimizer_rates(SimpleNamespace(param_groups=[{'lr':value}]))


def test_multiple_scalar_rates_do_not_mutate_optimizer_or_rng():
    value=torch.tensor(.2)
    optimizer=SimpleNamespace(param_groups=[{'lr':.1},{'lr':value},{'lr':0}])
    rng=torch.get_rng_state().clone()
    rates=_optimizer_rates(optimizer)
    assert rates=={'lr.group0':.1,'lr.group1':pytest.approx(.2),'lr.group2':0.}
    assert optimizer.param_groups[1]['lr'] is value
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('groups',[None,[],{}])
def test_missing_groups_cannot_silently_verify(groups):
    with pytest.raises(ValueError,match='no learning-rate parameter groups'):
        _optimizer_rates(SimpleNamespace(param_groups=groups))


@pytest.mark.parametrize('record_state',[False,True])
def test_rate_scope_combines_with_optional_model_state(tmp_path,record_state):
    path=tmp_path/'combined'
    report=check_hf(ConstantLoss,Data(),4,2,record_learning_rates=True,
                    record_model_state=record_state,artifacts_dir=path)
    assert report.passed
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['comparison']['check_model_state']==record_state
    assert asdict(replay_comparison_artifacts(path))==asdict(report)


@pytest.mark.parametrize('drop',[None,4])
def test_requested_rate_capture_cannot_disappear(monkeypatch,drop):
    import resumecheck.hf as hf
    real=hf._optimizer_rates
    calls=0
    def capture(optimizer):
        nonlocal calls
        calls+=1
        return {} if drop is None or calls==drop else real(optimizer)
    monkeypatch.setattr(hf,'_optimizer_rates',capture)
    if drop is None:
        with pytest.raises(ValueError,match='no optimizer learning rates'):
            check_hf(ConstantLoss,Data(),4,2,record_learning_rates=True)
    else:
        report=check_hf(ConstantLoss,Data(),4,2,record_learning_rates=True)
        assert not report.passed
        assert 'missing' in report.summary


@pytest.mark.parametrize('accumulation',[1,3])
def test_selected_rate_comparison_is_the_scheduler_failure_witness(monkeypatch,accumulation):
    load=Trainer._load_optimizer_and_scheduler
    def restore(self,checkpoint):
        if not checkpoint:
            return load(self,checkpoint)
        restore_scheduler=self.lr_scheduler.load_state_dict
        self.lr_scheduler.load_state_dict=lambda state: None
        try:
            return load(self,checkpoint)
        finally:
            self.lr_scheduler.load_state_dict=restore_scheduler
    monkeypatch.setattr(Trainer,'_load_optimizer_and_scheduler',restore)
    report=check_hf(ConstantLoss,Data(),6,3,record_learning_rates=True,
                    training_args_kwargs={'gradient_accumulation_steps':accumulation})
    assert not report.passed, 'scheduler-state omission must diverge despite constant loss'
    assert report.first_divergence.step==4
    assert report.first_divergence.field.startswith('extras.lr.group')
