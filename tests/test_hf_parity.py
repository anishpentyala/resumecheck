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


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("changing", [False, True])
def test_batch_fingerprint_preserves_preforward_inputs(mutate, changing):
    class ChangingDataset(Dataset):
        calls = 0

        def __len__(self):
            return 16

        def __getitem__(self, index):
            type(self).calls += 1
            value = float(type(self).calls if changing else index)
            return {"x": torch.tensor([value]), "y": torch.tensor([0.])}

    class InputMutatingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, x, y):
            # Batch-independent loss makes the content fingerprint the witness.
            # These inputs are not needed by autograd, so mutation is legal.
            if mutate:
                x.zero_()
                y.zero_()
            return {"loss": self.weight.square()}

    report = check_hf(InputMutatingModel, ChangingDataset(), max_steps=6, resume_at=3, seed=42)
    assert report.passed == (not changing), report.summary
    if changing:
        assert report.first_divergence.field == "batch"
        assert report.first_divergence.step == 1
    else:
        assert report.steps_compared == 6
