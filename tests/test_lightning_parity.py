"""End-to-end: deterministic lightning training resumed mid-run must match."""
import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")

from torch.utils.data import DataLoader, Dataset

from resumecheck.lightning import check_lightning


class TinyDataset(Dataset):
    def __init__(self, n=8):
        self.x = torch.arange(n, dtype=torch.float32).reshape(n, 1) / n
        self.y = 2 * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TinyDM(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(TinyDataset(), batch_size=2, shuffle=False)


class StatefulLoader(DataLoader):
    """A DataLoader that tracks consumed batches and, after load_state_dict,
    skips them once on the next iterator - lightning's `_Stateful` protocol.
    The skip applies to one iterator only, so later epochs run normally."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consumed = 0
        self._pending_skip = 0

    def __iter__(self):
        skip, self._pending_skip = self._pending_skip, 0
        for i, batch in enumerate(super().__iter__()):
            if i < skip:
                continue
            self._consumed += 1
            yield batch

    def state_dict(self):
        return {"consumed": self._consumed}

    def load_state_dict(self, state_dict):
        self._consumed = state_dict["consumed"]
        self._pending_skip = self._consumed % (len(self.dataset) // self.batch_size)


class StatefulDM(pl.LightningDataModule):
    def train_dataloader(self):
        return StatefulLoader(TinyDataset(), batch_size=2, shuffle=False)


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)


def test_resume_parity_stateful_loader_mid_epoch():
    # The supported path: a stateful dataloader restores its position, so a
    # mid-epoch resume must match uninterrupted training step for step.
    report = check_lightning(
        module_factory=TinyModel,
        datamodule_factory=StatefulDM,
        max_steps=6,
        resume_at=3,
        seed=42,
    )
    assert report.passed, report.summary
    assert report.steps_compared == 6


def test_mid_epoch_resume_with_plain_dataloader_is_flagged():
    # Known lightning behavior (it emits a PossibleUserWarning): a plain
    # DataLoader has no state to restore, so a mid-epoch resume replays the
    # epoch start. resumecheck surfaces this as a divergence.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = check_lightning(
            module_factory=TinyModel,
            datamodule_factory=TinyDM,
            max_steps=6,
            resume_at=3,
            seed=42,
        )
    assert not report.passed
    assert report.first_divergence.field == "batch"
    assert report.first_divergence.step == 4


def test_epoch_boundary_max_steps_resume_regression_lightning_22077():
    # Lightning bug: a checkpoint saved when max_steps coincides with the end
    # of an epoch resumes with the next epoch's first batch replayed and its
    # last batch dropped (no warning). Tracked upstream:
    # https://github.com/Lightning-AI/pytorch-lightning/issues/22077
    # resumecheck must flag it; with shuffle=False the duplicate IS the same
    # data as the expected batch, so the mismatch surfaces one step later.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = check_lightning(
            module_factory=TinyModel,
            datamodule_factory=TinyDM,
            max_steps=8,
            resume_at=4,
            seed=42,
        )
    assert not report.passed, "lightning#22077 appears fixed - update this spec"
    assert report.first_divergence.field == "batch"


def test_simulated_buggy_resume_diverges():
    """If resume replays already-seen batches (the classic lost-loader-state
    bug), the compare must fail at the first resumed step."""
    from resumecheck.compare import compare_trajectories
    from resumecheck.recorder import StepRecord, Trajectory

    def fake_run(order):
        return Trajectory(
            name="x",
            steps=[
                StepRecord(step=i, loss=1.0 - 0.01 * i, batch_fingerprint=f"batch{b}")
                for i, b in enumerate(order)
            ],
        )

    uninterrupted = fake_run([0, 1, 2, 3])
    buggy_resumed = fake_run([0, 1, 0, 1])  # loader restarted from scratch
    report = compare_trajectories(uninterrupted, buggy_resumed)
    assert not report.passed
    assert report.first_divergence.step == 2
    assert report.first_divergence.field == "batch"
