"""End-to-end: deterministic Accelerate training resumed mid-run must match.

Includes the epoch-boundary regression spec for the verified divergence on
accelerate==1.15.0 + torchdata==0.11.0: a save_state taken exactly at an
epoch boundary restores into a full replay of the completed epoch.

Verified mechanism (accelerate/data_loader.py + runtime probes):
DataLoaderShard.__iter__ calls self.set_epoch(self.iteration) at the start
of every iteration, i.e. the sampler's shuffle permutation is driven by the
shard's own iteration counter. That counter is NOT part of the checkpointed
state (the saved state_dict has no iteration field) and a crash-break at
the boundary stops the loop before __iter__ reaches its self.iteration += 1
line, so after load_state the next epoch reseeds the sampler with epoch 0
and replays the completed epoch's permutation. Mid-epoch resume
self-corrects because the resumed iteration completes the partial epoch
and the counter increments normally.

Verified caller workaround (2026-10-30): after load_state, call
train_dataloader.set_epoch(next_epoch_index) before continuing - parity
holds with it, fails without it (both directions probed).

Verified affected (2026-10-30): accelerate 1.15.0 (latest PyPI release)
and master 1.16.0.dev0 @ 90b233c, both with torchdata 0.11.0 (CPU,
seedable sampler + stateful dataloader, the documented resume path). The
set_epoch workaround restores parity on both.
Verified correct: mid-epoch resume on the same stacks.
Not claimed: other versions (untested) and torch-generator samplers
(post-restore permutation divergence there is known upstream as
pytorch/data#1561).
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
    "epoch (DataLoaderShard drives the sampler permutation from its own "
    "iteration counter, which is not checkpointed and is not advanced when "
    "the training loop is broken at an epoch boundary)"
)

# Verified affected (2026-10-30): accelerate 1.15.0 (latest PyPI release)
# and current master 1.16.0.dev0 @ commit 90b233c, both with
# torchdata 0.11.0. 1.16.0.dev0 is a moving dev version string; the
# snapshot tested is the named commit.
_VERIFIED_AFFECTED = (
    _importlib_metadata.version("accelerate") in ("1.15.0", "1.16.0.dev0")
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
    "versions (accelerate 1.15.0 and master 1.16.0.dev0 @ 90b233c, with "
    "torchdata 0.11.0); other versions are covered by "
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
