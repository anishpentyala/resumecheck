"""Owned default training directories are removed; caller-selected ones survive."""
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from resumecheck.lightning import check_lightning
from resumecheck.hf import check_hf

class Data(torch.utils.data.Dataset):
    def __len__(self):return 16
    def __getitem__(self,index):return {'x':torch.tensor([1.])}
class HFModel(torch.nn.Module):
    def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x):return {'loss':(self.w*x).square().mean()}
@pytest.mark.parametrize('adapter',['lightning','hf'])
@pytest.mark.parametrize('broken',[False,True])
@pytest.mark.parametrize('custom',[False,True])
@pytest.mark.parametrize('export',[False,True])
def test_owned_output_cleanup_on_success_and_error(tmp_path,monkeypatch,adapter,broken,custom,export):
    artifacts=tmp_path/"evidence"
    extra={"artifacts_dir":artifacts} if export else {}
    roots=[]
    sentinel=tmp_path/'caller';sentinel.mkdir();(sentinel/'keep').write_text('owned')
    if adapter=='lightning':
        pl=pytest.importorskip('lightning.pytorch')
        class PLModel(pl.LightningModule):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
            def training_step(self,batch,batch_idx):return (self.w*batch['x']).square().mean()
            def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.1)
        class DM(pl.LightningDataModule):
            def train_dataloader(self):return torch.utils.data.DataLoader(Data(),batch_size=2)

        original=pl.Trainer.fit
        def fit(self,*args,**kwargs):
            root=Path(self.default_root_dir);roots.append(root);(root/'probe').write_text('made')
            if broken:raise RuntimeError('intentional training failure')
            return original(self,*args,**kwargs)
        monkeypatch.setattr(pl.Trainer,'fit',fit)
        def run():return check_lightning(PLModel,DM,4,2,trainer_kwargs={'default_root_dir':str(sentinel)} if custom else {},**extra)
    else:
        pytest.importorskip("transformers")
        from transformers import Trainer
        original=Trainer.train
        def train(self,*args,**kwargs):
            root=Path(self.args.output_dir);roots.append(root);root.mkdir(exist_ok=True);(root/'probe').write_text('made')
            if broken:raise RuntimeError('intentional training failure')
            return original(self,*args,**kwargs)
        monkeypatch.setattr(Trainer,'train',train)
        def run():return check_hf(HFModel,Data(),4,2,training_args_kwargs={'output_dir':str(sentinel)} if custom else {},**extra)
    if broken:
        with pytest.raises(RuntimeError,match='intentional training'):run()
    else:assert run().passed
    assert roots
    if custom:
        # HF pre-crash checkpoint path is internally owned even with a caller output root.
        assert (sentinel/'keep').read_text()=='owned'
        assert (sentinel/'probe').exists()
        assert all(root.exists() for root in roots if root==sentinel)
    else:assert all(not root.exists() for root in roots)

    if export and not broken:
        from resumecheck.artifacts import replay_comparison_artifacts
        assert replay_comparison_artifacts(artifacts).passed
    else:assert not artifacts.exists()
