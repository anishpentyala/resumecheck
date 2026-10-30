"""End-to-end: deterministic HF Trainer training resumed mid-run must match."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch.utils.data import Dataset

from resumecheck.hf import check_hf


class TinyDataset(Dataset):
    def __init__(self, n=16):
        self.x = torch.arange(n, dtype=torch.float32).reshape(n, 1) / n
        self.y = 2 * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x, y):
        loss = torch.nn.functional.mse_loss(self.layer(x), y)
        return {"loss": loss}


def test_hf_resume_parity_matches_uninterrupted():
    report = check_hf(
        model_factory=TinyModel,
        train_dataset=TinyDataset(),
        max_steps=6,
        resume_at=3,
        seed=42,
    )
    assert report.passed, report.summary
    assert report.steps_compared == 6
def test_hf_epoch_boundary_resume_parity():
    """Resume exactly at an epoch boundary: n=8, batch 2 -> 4 steps/epoch,
    resume_at=4 starts the resumed phase at the first step of epoch 2.

    This is the HF analog of the Lightning epoch-boundary regression
    (#22077): the Trainer must restore sampler position AND the per-epoch
    shuffle so epoch 2 draws batches in the same order as the
    uninterrupted run."""
    report = check_hf(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=8,
        resume_at=4,
        seed=42,
    )
    assert report.passed, report.summary
    assert report.steps_compared == 8
