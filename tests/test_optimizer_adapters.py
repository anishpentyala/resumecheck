"""Real CPU checkpoint reloads: corrupt moments with model/loss/LR fixed."""
import json
from dataclasses import asdict
import pytest

torch = pytest.importorskip('torch')
from resumecheck.artifacts import replay_comparison_artifacts


class Dataset(torch.utils.data.Dataset):
    def __len__(self): return 48
    def __getitem__(self, index): return {'x': torch.tensor([float(index % 3)])}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, x): return {'loss': (self.weight - x - 2).square().mean()}


def corrupt(optimizer):
    state = optimizer.state_dict()
    assert state['state'], 'test must corrupt actual populated optimizer state'
    for values in state['state'].values():
        if 'exp_avg' in values:
            values['exp_avg'].add_(5)
        elif 'momentum_buffer' in values:
            values['momentum_buffer'].add_(5)
        else:
            pytest.fail('no expected moment in state')
    optimizer.load_state_dict(state)


def inspect(report, path, record, perturb):
    assert report.passed == (not record or not perturb), report.summary
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
    assert json.loads((path/'manifest.json').read_text())['schema_version'] == (4 if record else 2)
    steps = json.loads((path/'resumed.json').read_text())['steps']
    assert all(('optimizer_fingerprint' in s) == record for s in steps)
    if record and perturb:
        assert report.first_divergence.field == 'optimizer_state'
        assert report.first_divergence.step == 3


@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('perturb', [False, True])
@pytest.mark.parametrize('accumulation', [1, 3])
def test_hf_adamw(tmp_path, monkeypatch, record, perturb, accumulation):
    transformers = pytest.importorskip('transformers')
    from resumecheck.hf import check_hf
    original = transformers.Trainer._load_optimizer_and_scheduler
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if perturb: corrupt(self.optimizer)
        return result
    monkeypatch.setattr(transformers.Trainer, '_load_optimizer_and_scheduler', load)
    path = tmp_path/'evidence'
    report = check_hf(Model, Dataset(), 4, 2, seed=42,
                      record_model_state=True, record_optimizer_state=record,
                      artifacts_dir=path, training_args_kwargs={
                          'learning_rate': 0.0, 'gradient_accumulation_steps': accumulation})
    inspect(report,path,record,perturb)


@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('perturb', [False, True])
@pytest.mark.parametrize('accumulation', [1, 3])
def test_lightning_adamw(tmp_path, record, perturb, accumulation):
    pl = pytest.importorskip('lightning.pytorch')
    from resumecheck.lightning import check_lightning
    class Data(pl.LightningDataModule):
        def train_dataloader(self):
            return torch.utils.data.DataLoader(torch.ones(48,1), batch_size=2)
    class Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
        def training_step(self,batch,batch_idx): return (self.weight-batch-2).square().mean()
        def configure_optimizers(self): return torch.optim.AdamW(self.parameters(),lr=0.0)
        def on_load_checkpoint(self, checkpoint):
            if perturb:
                state=checkpoint['optimizer_states'][0]['state']
                assert state
                for values in state.values(): values['exp_avg'].add_(5)
    path = tmp_path/'evidence'
    report = check_lightning(Module,Data,4,2,seed=42,
                             record_model_state=True,record_optimizer_state=record,
                             artifacts_dir=path,trainer_kwargs={'accumulate_grad_batches':accumulation})
    inspect(report,path,record,perturb)


@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('perturb', [False, True])
@pytest.mark.parametrize('skip', [False, True])
def test_accelerate_momentum(tmp_path, monkeypatch, record, perturb, skip):
    accelerate=pytest.importorskip('accelerate'); pytest.importorskip('torchdata')
    from resumecheck.accelerate import check_accelerate, check_accelerate_skip_path
    original_sgd=torch.optim.SGD
    # The adapters deliberately use SGD, so supply a real zero-LR momentum SGD
    # in this experiment without changing adapter training semantics.
    def sgd(parameters, lr): return original_sgd(parameters,lr=0.0,momentum=0.9)
    monkeypatch.setattr(torch.optim,'SGD',sgd)
    original=accelerate.Accelerator.load_state
    def load(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if perturb: corrupt(self._optimizers[0])
        return result
    monkeypatch.setattr(accelerate.Accelerator,'load_state',load)
    path=tmp_path/'evidence'
    check=check_accelerate_skip_path if skip else check_accelerate
    report=check(Model,Dataset(),4,2,seed=42,record_model_state=True,
                 record_optimizer_state=record,artifacts_dir=path)
    inspect(report,path,record,perturb)


@pytest.mark.parametrize('adapter', ['hf','lightning','accelerate','accelerate_skip'])
@pytest.mark.parametrize('value', [1,None,'yes'])
def test_invalid_flags_before_factory(adapter,value):
    from resumecheck.hf import check_hf
    from resumecheck.lightning import check_lightning
    from resumecheck.accelerate import check_accelerate,check_accelerate_skip_path
    check={'hf':check_hf,'lightning':check_lightning,'accelerate':check_accelerate,'accelerate_skip':check_accelerate_skip_path}[adapter]
    def forbidden(): pytest.fail('must validate before model construction')
    with pytest.raises(TypeError,match='record_optimizer_state'):
        check(forbidden,forbidden if adapter=='lightning' else Dataset(),4,2,record_optimizer_state=value)


@pytest.mark.parametrize('adapter', ['hf','lightning','accelerate','accelerate_skip'])
@pytest.mark.parametrize('record', [False, True])
def test_default_no_optimizer_hash_and_enabled_is_observational(adapter,record,monkeypatch,tmp_path):
    pytest.importorskip({'hf':'transformers','lightning':'lightning.pytorch','accelerate':'accelerate','accelerate_skip':'accelerate'}[adapter])
    if adapter.startswith('accelerate'): pytest.importorskip('torchdata')
    import copy
    import resumecheck.recorder as recorder
    module=__import__('resumecheck.'+('accelerate' if adapter.startswith('accelerate') else adapter),fromlist=['unused'])
    original=recorder.fingerprint_tensor
    calls=[]
    def inspected(value):
        state=value[0] if isinstance(value,list) and len(value)==1 and isinstance(value[0],dict) else value
        if isinstance(state,dict) and 'state' in state and 'param_groups' in state:
            assert record, 'default must not copy/hash optimizer state'
            before=copy.deepcopy(value); rng=torch.get_rng_state().clone()
            result=original(value)
            assert result==original(before)
            assert torch.equal(rng,torch.get_rng_state())
            calls.append(result)
            return result
        return original(value)
    if hasattr(module,'fingerprint_tensor'):
        monkeypatch.setattr(module,'fingerprint_tensor',inspected)
    # Lightning imports the recorder function locally in its callback.
    monkeypatch.setattr(recorder,'fingerprint_tensor',inspected)
    path=tmp_path/'evidence'
    if adapter=='lightning':
        pl=pytest.importorskip('lightning.pytorch')
        class Data(pl.LightningDataModule):
            def train_dataloader(self): return torch.utils.data.DataLoader(torch.ones(32,1),batch_size=2)
        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__(); self.a=torch.nn.Parameter(torch.tensor(1.)); self.b=torch.nn.Parameter(torch.tensor(2.))
            def training_step(self,batch,batch_idx): return (self.a*batch+self.b).square().mean()
            def configure_optimizers(self):
                return torch.optim.AdamW([{'params':[self.a],'lr':.01},{'params':[self.b],'lr':.02}])
        report=module.check_lightning(Module,Data,4,2,seed=42,record_optimizer_state=record,artifacts_dir=path)
    else:
        if adapter=='hf':
            check=module.check_hf
        else:
            check=module.check_accelerate_skip_path if adapter.endswith('skip') else module.check_accelerate
        report=check(Model,Dataset(),4,2,seed=42,record_optimizer_state=record,artifacts_dir=path)
    assert report.passed,report.summary
    assert len(calls)==(8 if record else 0)
    assert asdict(replay_comparison_artifacts(path))==asdict(report)


def test_actual_adamw_warmup_blind_spot_then_next_update_diverges():
    """No checkpoint framework mocked: only optimizer restoration is omitted."""
    import copy
    from resumecheck.recorder import TrajectoryRecorder,fingerprint_tensor
    from resumecheck.compare import compare_trajectories
    torch.manual_seed(83)
    model=torch.nn.Linear(2,1,bias=False).double()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0)
    x=torch.tensor([[1.,2.]],dtype=torch.float64)
    (model(x)-5.).square().mean().backward(); optimizer.step(); optimizer.zero_grad()
    clone=torch.nn.Linear(2,1,bias=False).double(); clone.load_state_dict(copy.deepcopy(model.state_dict()))
    fresh=torch.optim.AdamW(clone.parameters(),lr=0.0)
    a,b=TrajectoryRecorder('full'),TrajectoryRecorder('broken-resume')
    for m,o,r in ((model,optimizer,a),(clone,fresh,b)):
        r.record(1,loss=(m(x)-.5).square().mean().item(),batch=x,lr=o.param_groups[0]['lr'])
        r.trajectory.steps[-1].model_fingerprint=fingerprint_tensor(m.state_dict())
        r.trajectory.steps[-1].optimizer_fingerprint=fingerprint_tensor(o.state_dict())
    assert compare_trajectories(a.trajectory,b.trajectory,check_optimizer_state=False,check_extras=['lr'],rtol=0,atol=0).passed
    report=compare_trajectories(a.trajectory,b.trajectory,check_extras=['lr'],rtol=0,atol=0)
    assert not report.passed and report.first_divergence.field=='optimizer_state'
    for m,o in ((model,optimizer),(clone,fresh)):
        o.param_groups[0]['lr']=.1
        (m(x)-.5).square().mean().backward();o.step();o.zero_grad()
    assert (model.weight-clone.weight).abs().max().item()>.1


@pytest.mark.parametrize('adapter',['hf','lightning','accelerate','accelerate_skip'])
def test_unsupported_custom_optimizer_state_raises_without_bundle(adapter,monkeypatch,tmp_path):
    pytest.importorskip({'hf':'transformers','lightning':'lightning.pytorch','accelerate':'accelerate','accelerate_skip':'accelerate'}[adapter])
    if adapter.startswith('accelerate'): pytest.importorskip('torchdata')
    import resumecheck.hf as hf
    import resumecheck.lightning as lightning
    import resumecheck.accelerate as accelerate
    adam,sgd=torch.optim.AdamW,torch.optim.SGD
    class CustomAdam(adam):
        def state_dict(self):
            result=super().state_dict();result['opaque_custom_state']=object();return result
    class CustomSGD(sgd):
        def state_dict(self):
            result=super().state_dict();result['opaque_custom_state']=object();return result
    monkeypatch.setattr(torch.optim,'AdamW',CustomAdam)
    monkeypatch.setattr(torch.optim,'SGD',CustomSGD)
    path=tmp_path/'evidence'
    if adapter=='lightning':
        pl=pytest.importorskip('lightning.pytorch')
        class Data(pl.LightningDataModule):
            def train_dataloader(self):return torch.utils.data.DataLoader(torch.ones(32,1),batch_size=2)
        class Module(pl.LightningModule):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
            def training_step(self,batch,batch_idx):return (self.weight*batch).square().mean()
            def configure_optimizers(self):return CustomAdam(self.parameters(),lr=.01)
        check=lightning.check_lightning;args=(Module,Data)
    else:
        check={'hf':hf.check_hf,'accelerate':accelerate.check_accelerate,'accelerate_skip':accelerate.check_accelerate_skip_path}[adapter]
        args=(Model,Dataset())
    with pytest.raises(TypeError,match='unsupported batch value'):
        check(*args,4,2,record_optimizer_state=True,artifacts_dir=path)
    assert not path.exists()


@pytest.mark.parametrize('adapter',['hf','lightning'])
def test_optimizer_and_learning_rate_observations_coexist(adapter,tmp_path):
    pytest.importorskip('transformers' if adapter=='hf' else 'lightning.pytorch')
    if adapter=='hf':
        from resumecheck.hf import check_hf
        check=check_hf; args=(Model,Dataset())
    else:
        import lightning.pytorch as pl
        from resumecheck.lightning import check_lightning
        class Data(pl.LightningDataModule):
            def train_dataloader(self):return torch.utils.data.DataLoader(torch.ones(32,1),batch_size=2)
        class Module(pl.LightningModule):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
            def training_step(self,batch,batch_idx):return (self.weight*batch).square().mean()
            def configure_optimizers(self):return torch.optim.AdamW(self.parameters(),lr=.01)
        check=check_lightning;args=(Module,Data)
    path=tmp_path/'evidence'
    report=check(*args,4,2,seed=42,record_optimizer_state=True,record_learning_rates=True,artifacts_dir=path)
    assert report.passed,report.summary
    assert asdict(replay_comparison_artifacts(path))==asdict(report)
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['schema_version']==4
    assert manifest['comparison']['check_extras']
    assert manifest['comparison']['check_model_state'] is False
