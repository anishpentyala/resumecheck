"""Actual Lightning loss-nullspace corruption must be observable on request."""

import json
from dataclasses import asdict
import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")
from resumecheck.lightning import check_lightning
from resumecheck.artifacts import replay_comparison_artifacts


class Data(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.ones(48, 1), batch_size=2, shuffle=False
        )


class Model(pl.LightningModule):
    def __init__(self, perturb=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        self.active = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("running", torch.tensor(1.0))
        self.perturb = perturb

    def training_step(self, batch, batch_idx):
        return {"loss": self.weight.sum().square() + self.active.square()}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)

    def on_load_checkpoint(self, checkpoint):
        if self.perturb == "parameter":
            checkpoint["state_dict"]["weight"] += torch.tensor([1.0, -1.0])
        elif self.perturb == "buffer":
            checkpoint["state_dict"]["running"] += 1.0


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("perturb", [None, "parameter", "buffer"])
@pytest.mark.parametrize("accumulation", [1, 3])
def test_loaded_nullspace_corruption(tmp_path, record, perturb, accumulation):
    path = tmp_path / "evidence"
    report = check_lightning(
        lambda: Model(perturb),
        Data,
        4,
        2,
        seed=42,
        record_model_state=record,
        artifacts_dir=path,
        trainer_kwargs={"accumulate_grad_batches": accumulation},
    )
    assert report.passed == (not record or perturb is None), report.summary
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
    steps = json.loads((path / "resumed.json").read_text())["steps"]
    assert all(("model_fingerprint" in step) == record for step in steps)
    if record and perturb:
        assert report.first_divergence.field == "model_state"
        assert report.first_divergence.step == 3


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_invalid_state_flag_rejected_before_factory(value):
    def forbidden():
        pytest.fail("must validate before constructing model")

    with pytest.raises(TypeError, match="must be boolean"):
        check_lightning(forbidden, Data, 4, 2, record_model_state=value)


def test_unsupported_extra_state_fails_without_bundle(tmp_path):
    class CustomState(Model):
        def get_extra_state(self):
            return object()

    path = tmp_path / "evidence"
    with pytest.raises(TypeError, match="unsupported batch value"):
        check_lightning(
            CustomState, Data, 4, 2, record_model_state=True, artifacts_dir=path
        )
    assert not path.exists()


@pytest.mark.parametrize("perturb", ["parameter", "buffer"])
def test_default_loss_observables_cannot_detect_nullspace_corruption(perturb):
    report = check_lightning(lambda: Model(perturb), Data, 4, 2, seed=42)
    assert report.passed, report.summary


@pytest.mark.parametrize("accumulation", [1, 2, 3, 4])
def test_real_deterministic_training_state_matches_after_checkpoint(
    tmp_path, accumulation
):
    class TrainingModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
            )

        def training_step(self, batch, batch_idx):
            return {"loss": (self.net(batch) - 2.0).square().mean()}

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)

    path = tmp_path / "evidence"
    report = check_lightning(
        TrainingModel,
        Data,
        4,
        2,
        seed=42,
        record_model_state=True,
        artifacts_dir=path,
        trainer_kwargs={"accumulate_grad_batches": accumulation},
    )
    assert report.passed, report.summary
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
