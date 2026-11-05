"""Requested learning-rate evidence catches scheduler errors invisible to loss."""
import json
from dataclasses import asdict
import pytest

torch = pytest.importorskip('torch')
pl = pytest.importorskip('lightning.pytorch')
from resumecheck.lightning import check_lightning
from resumecheck.artifacts import replay_comparison_artifacts


class Data(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.ones(48,1), batch_size=2, shuffle=False)


class ConstantLoss(pl.LightningModule):
    corrupt = False
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
    def training_step(self,batch,batch_idx):
        return {'loss':self.weight*0+1}
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.5)
        return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}
    def on_load_checkpoint(self,checkpoint):
        if self.corrupt:
            checkpoint['lr_schedulers'][0]['last_epoch'] = 0


@pytest.mark.parametrize('record',[False,True])
@pytest.mark.parametrize('corrupt',[False,True])
@pytest.mark.parametrize('accumulation',[1,3])
def test_real_checkpoint_scheduler_phase_with_constant_loss(tmp_path,record,corrupt,accumulation):
    class Model(ConstantLoss): pass
    Model.corrupt = corrupt
    path = tmp_path/'evidence'
    report = check_lightning(Model,Data,6,3,record_learning_rates=record,
                             artifacts_dir=path,trainer_kwargs={'accumulate_grad_batches':accumulation})
    assert report.passed == (not record or not corrupt), report.summary
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
    manifest = json.loads((path/'manifest.json').read_text())
    rows = json.loads((path/'uninterrupted.json').read_text())['steps']
    if record:
        assert manifest['schema_version'] == 3
        assert manifest['comparison']['check_extras'] == ['optimizer0.lr.group0']
        assert len(rows) == 6
        assert rows[0]['extras']['optimizer0.lr.group0'] > rows[1]['extras']['optimizer0.lr.group0']
        if corrupt:
            assert report.first_divergence.field == 'extras.optimizer0.lr.group0'
            assert report.first_divergence.step == 4
    else:
        assert manifest['schema_version'] == 1
        assert all(r['extras']=={} for r in rows)


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_flag_is_strict_boolean(value):
    with pytest.raises(TypeError,match='must be boolean'):
        check_lightning(ConstantLoss,Data,4,2,record_learning_rates=value)


@pytest.mark.parametrize('value',[None,True,'1',float('nan'),float('inf'),-1,torch.tensor([.1]),torch.tensor(True)])
def test_invalid_rate_cannot_verify(value):
    from types import SimpleNamespace
    from resumecheck.lightning import _optimizer_rates
    with pytest.raises(ValueError,match='finite nonnegative scalar'):
        _optimizer_rates(SimpleNamespace(param_groups=[{'lr':value}]))


@pytest.mark.parametrize('groups',[None,[],{}])
def test_missing_groups_cannot_verify(groups):
    from types import SimpleNamespace
    from resumecheck.lightning import _optimizer_rates
    with pytest.raises(ValueError,match='no learning-rate parameter groups'):
        _optimizer_rates(SimpleNamespace(param_groups=groups))


@pytest.mark.parametrize('record_state',[False,True])
def test_multiple_groups_and_model_state_scope(tmp_path,record_state):
    class TwoGroups(ConstantLoss):
        def __init__(self):
            super().__init__()
            self.other = torch.nn.Parameter(torch.tensor(2.))
        def configure_optimizers(self):
            return torch.optim.SGD([{'params':[self.weight],'lr':.1},
                                    {'params':[self.other],'lr':.2}],lr=.1)
    path=tmp_path/'evidence'
    report=check_lightning(TwoGroups,Data,4,2,record_learning_rates=True,
                           record_model_state=record_state,artifacts_dir=path)
    assert report.passed
    assert asdict(replay_comparison_artifacts(path))==asdict(report)
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['comparison']['check_model_state'] is record_state
    assert manifest['comparison']['check_extras']==['optimizer0.lr.group0','optimizer0.lr.group1']


@pytest.mark.parametrize('drop',[None,4])
def test_rate_capture_cannot_silently_disappear(monkeypatch,drop):
    import resumecheck.lightning as adapter
    capture=adapter._optimizer_rates
    calls=0
    def rates(optimizer):
        nonlocal calls
        calls+=1
        return {} if drop is None or calls==drop else capture(optimizer)
    monkeypatch.setattr(adapter,'_optimizer_rates',rates)
    with pytest.raises(ValueError,match='no optimizer learning rates'):
        check_lightning(ConstantLoss,Data,4,2,record_learning_rates=True)


def test_multiple_scalar_rates_do_not_mutate_optimizer_or_rng():
    from types import SimpleNamespace
    from resumecheck.lightning import _optimizer_rates
    tensor=torch.tensor(.2)
    optimizer=SimpleNamespace(param_groups=[{'lr':.1},{'lr':tensor},{'lr':0}])
    rng=torch.get_rng_state().clone()
    rates=_optimizer_rates(optimizer)
    assert rates=={'lr.group0':.1,'lr.group1':pytest.approx(.2),'lr.group2':0.}
    assert optimizer.param_groups[1]['lr'] is tensor
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('accumulation',[1,3])
def test_scheduler_phase_error_has_observable_negative_control(accumulation):
    import inspect
    class Corrupt(ConstantLoss):
        corrupt=True
    optional = ({'record_learning_rates':True}
                if 'record_learning_rates' in inspect.signature(check_lightning).parameters else {})
    report=check_lightning(Corrupt,Data,6,3,trainer_kwargs={'accumulate_grad_batches':accumulation},**optional)
    assert not report.passed, 'constant-loss scheduler counter corruption must be visible when rate evidence is supported'
    assert report.first_divergence.step==4
    assert report.first_divergence.field=='extras.optimizer0.lr.group0'


def test_epoch_scheduler_samples_are_scoped_and_replayable(tmp_path):
    class EpochScheduler(ConstantLoss):
        def configure_optimizers(self):
            optimizer=torch.optim.SGD(self.parameters(),lr=.1)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.5)
            return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'epoch'}}
    path=tmp_path/'epoch'
    report=check_lightning(EpochScheduler,Data,4,2,record_learning_rates=True,artifacts_dir=path)
    assert report.passed
    rows=json.loads((path/'uninterrupted.json').read_text())['steps']
    assert all(r['extras']['optimizer0.lr.group0']==.1 for r in rows)
    assert asdict(replay_comparison_artifacts(path))==asdict(report)
