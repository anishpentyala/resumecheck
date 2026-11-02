"""Adapter limits never silently round or accept boolean experiment boundaries."""
import importlib
import pytest

def validate_step_limits(*args, **kwargs):
    from resumecheck.validation import validate_step_limits as validate
    return validate(*args, **kwargs)


class Index:
    def __init__(self, value): self.value = value
    def __index__(self): return self.value


@pytest.mark.parametrize("value", [True, False, 2.5, 3.0, "3", None, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["max_steps", "resume_at"])
def test_shared_limits_reject_coercions(value, field):
    kwargs = dict(max_steps=6, resume_at=2)
    kwargs[field] = value
    with pytest.raises(TypeError, match=field):
        validate_step_limits(**kwargs)


@pytest.mark.parametrize("max_steps,resume_at", [(0, 0), (-1, 0), (6, 0), (6, 6), (6, 7), (6, -1)])
def test_shared_limits_reject_invalid_ranges(max_steps, resume_at):
    with pytest.raises(ValueError, match="resume_at"):
        validate_step_limits(max_steps, resume_at)


def test_integer_index_values_normalized():
    assert validate_step_limits(Index(6), Index(2)) == (6, 2)
    assert all(type(value) is int for value in validate_step_limits(Index(6), Index(2)))


@pytest.mark.parametrize("adapter", ["lightning", "hf", "accelerate", "skip"])
@pytest.mark.parametrize("field,value", [("max_steps", 6.5), ("max_steps", 6.0), ("max_steps", True), ("resume_at", True), ("resume_at", 2.5), ("resume_at", "2")])
def test_invalid_limits_precede_factory_and_artifacts(tmp_path, monkeypatch, adapter, field, value):
    name = "accelerate" if adapter == "skip" else adapter
    module = importlib.import_module("resumecheck." + name)
    check = getattr(module, "check_accelerate_skip_path" if adapter == "skip" else "check_" + name)
    monkeypatch.setattr(module, "_require_" + name, lambda: None)
    def forbidden():
        pytest.fail("invalid experiment must not start training")
    kwargs = dict(max_steps=6, resume_at=2, artifacts_dir=tmp_path / "evidence")
    kwargs[field] = value
    with pytest.raises(TypeError, match=field):
        check(forbidden, None, **kwargs)
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("adapter", ["lightning", "hf", "accelerate", "skip"])
def test_actual_integer_index_experiment_matches_plain_int(adapter):
    torch = pytest.importorskip("torch")
    name = "accelerate" if adapter == "skip" else adapter
    module = importlib.import_module("resumecheck." + name)
    class Data(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, i): return {"x": torch.ones(1), "y": torch.ones(1)}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
        def forward(self, x, y):
            return {"loss": (self.layer(x) - y).square().mean()}
    if adapter == "lightning":
        pl = pytest.importorskip("lightning.pytorch")
        class LightningModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.net = Model()
            def training_step(self, batch, batch_idx): return self.net(**batch)
            def configure_optimizers(self): return torch.optim.SGD(self.parameters(), lr=0.1)
        class LightningData(pl.LightningDataModule):
            def train_dataloader(self): return torch.utils.data.DataLoader(Data(), batch_size=2)
        check = module.check_lightning
        factories = (LightningModel, LightningData)
    else:
        pytest.importorskip("transformers" if adapter == "hf" else "accelerate")
        if adapter in ("accelerate", "skip"): pytest.importorskip("torchdata")
        check = getattr(module, "check_accelerate_skip_path" if adapter == "skip" else "check_" + name)
        factories = (Model, Data())
    plain = check(*factories, max_steps=6, resume_at=2, seed=42)
    indexed = check(*factories, max_steps=Index(6), resume_at=Index(2), seed=42)
    from dataclasses import asdict
    assert plain.passed and indexed.passed
    assert plain.steps_compared == indexed.steps_compared == 6
    assert asdict(plain) == asdict(indexed)
