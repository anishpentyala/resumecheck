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
and master 1.16.0.dev0 @ commit 90b233c, both with torchdata 0.11.0 (CPU,
seedable sampler + stateful dataloader, the documented resume path). Both
snapshots carry byte-identical DataLoaderShard.__iter__/set_epoch source;
the spec gate binds to that mechanism fingerprint, not a version string,
so an unknown future master asserts compatibility normally. The set_epoch
workaround restores parity on both.
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

def _shard_mechanism_fingerprint() -> str:
    """SHA-256 of the source of the two DataLoaderShard methods that carry
    the boundary-replay mechanism (the iteration counter driving the
    sampler permutation)."""
    import hashlib
    import inspect

    from accelerate import data_loader

    src = inspect.getsource(
        data_loader.DataLoaderShard.__iter__
    ) + inspect.getsource(data_loader.DataLoaderShard.set_epoch)
    return hashlib.sha256(src.encode()).hexdigest()


# Verified-affected mechanism fingerprint (2026-10-30): identical source
# on accelerate 1.15.0 (latest PyPI release) and master 1.16.0.dev0
# @ commit 90b233c, both with torchdata 0.11.0. The gate binds to the
# mechanism's source identity, NOT the version string: a dev-version
# string is shared by many snapshots, so a future master that changes
# these methods (e.g. checkpointing the iteration counter) falls through
# to the normal must-pass parity assertion instead of xfailing.
_VERIFIED_AFFECTED = (
    _shard_mechanism_fingerprint()
    == "586ee0fae2f9bc9d59d28b9cd83a9a395fa02fc063793c65463832b195c364e4"
    and _importlib_metadata.version("torchdata").startswith("0.11.0")
)


def _workers_region_fingerprint() -> str:
    """SHA-256 of the source of the accelerate classes carrying the
    workers>0 mechanism region: the seedable sampler, the eager
    prepare()-time state snapshot in DataLoaderAdapter, and the shard's
    iteration lifecycle."""
    import hashlib
    import inspect

    from accelerate import data_loader

    src = (
        inspect.getsource(data_loader.SeedableRandomSampler)
        + inspect.getsource(data_loader.DataLoaderAdapter)
        + inspect.getsource(data_loader.DataLoaderShard.__iter__)
        + inspect.getsource(data_loader.DataLoaderShard.set_epoch)
    )
    return hashlib.sha256(src.encode()).hexdigest()


# Verified-affected stacks for the workers>0 divergence (2026-10-30),
# each confirmed by direct reproduction: accelerate 1.15.0 (latest PyPI
# release) and master 1.16.0.dev0 @ commit 90b233c, both with
# torchdata 0.11.0. The gate binds to the exact source fingerprints of
# the verified stacks - a verified-stack marker, not a claim that every
# build sharing the fingerprint is affected; any other build is covered
# by the positive parity test asserting compatibility normally.
_VERIFIED_AFFECTED_WORKERS_FINGERPRINTS = {
    "b38c0ad141ace80fca50ccbb72e5a20e0e7708f341c52333704f13622aa15aa7": "accelerate 1.15.0",
    "a77f17159649213cb3cff147b2f1af768d3475dbe73500a692d75164434aa273": "accelerate master 1.16.0.dev0 @ 90b233c",
}
_VERIFIED_AFFECTED_WORKERS = (
    _workers_region_fingerprint() in _VERIFIED_AFFECTED_WORKERS_FINGERPRINTS
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
    "mechanism fingerprint (accelerate 1.15.0 and master 1.16.0.dev0 "
    "@ commit 90b233c share it, with torchdata 0.11.0); other builds are "
    "covered by the positive parity test asserting compatibility normally",
)
def test_accelerate_epoch_boundary_resume_regression():
    # Negative control: resumecheck must FLAG the epoch replay on the
    # verified-affected versions.
    report = _epoch_boundary_report()
    assert not report.passed, (
        "accelerate boundary replay appears fixed - update this spec"
    )
    assert report.first_divergence.field == "batch"

def test_accelerate_boundary_resume_with_set_epoch_workaround():
    """Opt-in mitigation mode: with the documented caller-side set_epoch
    workaround applied, boundary resume parity must hold - on every
    accelerate build (it verifies the mitigation, not the raw path)."""
    report = check_accelerate(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=8,
        resume_at=4,  # exact epoch boundary: 4 steps of bs=2 over n=8
        seed=42,
        boundary_set_epoch_workaround=True,
    )
    assert report.passed, report.summary


def test_accelerate_workaround_mode_rejects_mid_epoch():
    """Fail-closed: the mitigation is verified only at exact boundaries;
    a mid-epoch resume_at with the flag on must be rejected, not silently
    applied."""
    with pytest.raises(ValueError, match="exact epoch boundaries"):
        check_accelerate(
            model_factory=TinyModel,
            train_dataset=TinyDataset(n=8),
            max_steps=8,
            resume_at=3,
            seed=42,
            boundary_set_epoch_workaround=True,
        )


_SKIP_PATH_SHIFT = (
    "accelerate skip_first_batches resume shifts every post-resume epoch "
    "back one permutation (the skip wrapper consumes the sampler without "
    "advancing DataLoaderShard's iteration counter)"
)


@pytest.mark.xfail(
    _VERIFIED_AFFECTED,
    strict=True,
    raises=AssertionError,
    reason=_SKIP_PATH_SHIFT,
)
def test_accelerate_skip_path_resume_parity():
    """Positive check: the documented skip_first_batches resume SHOULD
    reproduce the uninterrupted run. Strict-xfail only on the
    verified-affected mechanism fingerprint; elsewhere it asserts
    compatibility normally."""
    from resumecheck.accelerate import check_accelerate_skip_path

    report = check_accelerate_skip_path(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=8,
        resume_at=3,  # mid-epoch: 4 steps/epoch over n=8 at bs=2
        seed=42,
    )
    assert report.passed, report.summary


@pytest.mark.skipif(
    not _VERIFIED_AFFECTED,
    reason="skip-path shift reproduction is pinned to the "
    "verified-affected mechanism fingerprint; other builds are covered by "
    "the positive parity test asserting compatibility normally",
)
def test_accelerate_skip_path_resume_regression():
    """Negative control: resumecheck must FLAG the post-resume epoch
    permutation shift on the verified-affected mechanism."""
    from resumecheck.accelerate import check_accelerate_skip_path

    report = check_accelerate_skip_path(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=8,
        resume_at=3,
        seed=42,
    )
    assert not report.passed, (
        "accelerate skip-path shift appears fixed - update this spec"
    )


def test_accelerate_skip_path_resume_with_set_epoch_workaround():
    """Opt-in mitigation mode: with set_epoch(epoch) applied to each
    post-resume epoch, skip-path resume parity must hold - on every
    accelerate build (it verifies the mitigation, not the raw path)."""
    from resumecheck.accelerate import check_accelerate_skip_path

    report = check_accelerate_skip_path(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=8,
        resume_at=3,
        seed=42,
        skip_path_set_epoch_workaround=True,
    )
    assert report.passed, report.summary


def test_accelerate_skip_path_workaround_rejects_boundary():
    """Fail-closed: the skip-path mitigation is verified only for
    mid-epoch resumes; a boundary resume_at with the flag on must be
    rejected, not silently applied."""
    from resumecheck.accelerate import check_accelerate_skip_path

    with pytest.raises(ValueError, match="mid-epoch"):
        check_accelerate_skip_path(
            model_factory=TinyModel,
            train_dataset=TinyDataset(n=8),
            max_steps=8,
            resume_at=4,
            seed=42,
            skip_path_set_epoch_workaround=True,
        )


_WORKERS_MIDEPOCH_DIVERGENCE = (
    "accelerate stateful path with num_workers>=1: prepare() eagerly "
    "creates the iterator through the original plain RandomSampler "
    "(global-RNG draw), the seedable sampler never iterates, and the "
    "saved state carries no restorable sample cursor - mid-epoch resume "
    "duplicates and drops samples"
)


@pytest.mark.xfail(
    _VERIFIED_AFFECTED_WORKERS,
    strict=True,
    raises=AssertionError,
    reason=_WORKERS_MIDEPOCH_DIVERGENCE,
)
def test_accelerate_workers_mid_epoch_resume_parity():
    """Positive check: with num_workers=2, mid-epoch resume SHOULD
    reproduce the uninterrupted run. Strict-xfail only on the two
    verified-affected stacks; everywhere else it asserts compatibility
    normally - a real pass on fixed upstream, a loud failure on any
    other broken build."""
    report = check_accelerate(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=6,
        resume_at=3,  # mid-epoch: 4 steps/epoch over n=8 at bs=2
        seed=42,
        num_workers=2,
    )
    assert report.passed, report.summary


@pytest.mark.skipif(
    not _VERIFIED_AFFECTED_WORKERS,
    reason="workers>0 divergence reproduction is pinned to the two "
    "verified-affected stacks; other builds are covered by the positive "
    "parity test asserting compatibility normally",
)
def test_accelerate_workers_mid_epoch_resume_regression():
    """Negative control: resumecheck must FLAG the workers>0 mid-epoch
    divergence on the verified-affected stacks."""
    report = check_accelerate(
        model_factory=TinyModel,
        train_dataset=TinyDataset(n=8),
        max_steps=6,
        resume_at=3,
        seed=42,
        num_workers=2,
    )
    assert not report.passed, (
        "accelerate workers>0 mid-epoch divergence appears fixed - "
        "update this spec"
    )
