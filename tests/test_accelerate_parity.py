"""End-to-end: deterministic Accelerate training resumed mid-run must match.

Includes the epoch-boundary regression spec for the verified divergence on
accelerate==1.15.0 + torchdata==0.11.0: a save_state taken exactly at an
epoch boundary restores into a full replay of the completed epoch, because
SeedableRandomSampler advances its epoch counter only when its iterator is
fully exhausted (accelerate/data_loader.py) and the stateful restore
fast-forwards by draw-and-discard without exhausting it, so the epoch
counter never advances on restore.

Verified affected: accelerate 1.15.0 + torchdata 0.11.0 (CPU, seedable
sampler + stateful dataloader, the documented resume path).
Verified correct: mid-epoch resume on the same stack.
Not claimed: other versions (untested), the skip_first_batches path, and
torch-generator samplers (see torchdata#1561).
"""
import importlib.metadata as _importlib_metadata

import pytest

torch = pytest.importorskip("torch")
_accelerate = pytest.importorskip("accelerate")
_torchdata = pytest.importorskip("torchdata")

from torch.utils.data import Dataset

from resumecheck.accelerate import check_accelerate

_ACCELERATE_BOUNDARY_REPLAY = (
    "accelerate stateful-dataloader boundary resume replays the completed "
    "epoch (SeedableRandomSampler epoch counter is not advanced by "
    "torchdata's non-exhausting restore fast-forward)"
)

_VERIFIED_AFFECTED = (
    _importlib_metadata.version("accelerate") == "1.15.0"
    and _importlib_metadata.version("torchdata").startswith("0.11.0")
)


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


def test_accelerate_resume_parity_matches_uninterrupted():
    report = check_accelerate(
        model_factory=TinyModel,
        train_dataset=TinyDataset(),
        max_steps=6,
        resume_at=3,
        seed=42,
    )
    assert report.passed, report.summary
    assert report.steps_compared == 6


def _epoch_boundary_report():
    # n=8, batch 2 -> exactly 4 steps/epoch; resume_at=4 is the boundary
    return check_accelerate(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=8,
        resume_at=4,
        seed=42,
    )


@pytest.mark.xfail(
    _VERIFIED_AFFECTED,
    strict=True,
    raises=AssertionError,
    reason=_ACCELERATE_BOUNDARY_REPLAY,
)
def test_accelerate_epoch_boundary_resume_parity():
    """Positive check: epoch-boundary resume SHOULD reproduce the
    uninterrupted run. Strict-xfail only on the verified-affected versions;
    everywhere else it asserts compatibility normally - a real pass on
    fixed upstream, a loud failure on any other broken version."""
    report = _epoch_boundary_report()
    assert report.passed, report.summary


@pytest.mark.skipif(
    not _VERIFIED_AFFECTED,
    reason="epoch-replay reproduction is pinned to the verified-affected "
    "accelerate==1.15.0 + torchdata==0.11.0; other versions are covered by "
    "the positive parity test asserting compatibility normally",
)
def test_accelerate_epoch_boundary_resume_regression():
    # Negative control: resumecheck must FLAG the epoch replay on the
    # verified-affected versions.
    report = _epoch_boundary_report()
    assert not report.passed, (
        "accelerate boundary replay appears fixed - update this spec"
    )
    assert report.first_divergence.field == "batch"
