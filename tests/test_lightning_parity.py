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


import importlib.metadata as _importlib_metadata

from packaging.version import Version as _Version

_LIGHTNING_V = _Version(_importlib_metadata.version("lightning"))
# Version grounding (2026-10-30, see issue #22077 "Versions"):
#   - VERIFIED AFFECTED: lightning==2.6.6 (latest release at filing) and
#     unpatched master 2.7.0.dev0 (6557204).
#   - VERIFIED FIXED: master 2.7.0.dev0 with the #22080 patch applied.
# Other versions are not claimed either way: the positive parity test
# asserts compatibility normally there - a failure on an untested version
# is a genuine finding, not an expected one. Widen this set only with a
# tested version, never by assumption.
_VERIFIED_AFFECTED = _LIGHTNING_V == _Version("2.6.6")

_LIGHTNING_22077 = (
    "lightning#22077: epoch-boundary max_steps resume replays/drops batches "
    "(https://github.com/Lightning-AI/pytorch-lightning/issues/22077)"
)


def _epoch_boundary_report():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return check_lightning(
            module_factory=TinyModel,
            datamodule_factory=TinyDM,
            max_steps=8,
            resume_at=4,
            seed=42,
        )


@pytest.mark.xfail(
    _VERIFIED_AFFECTED,
    strict=True,
    raises=AssertionError,
    reason=_LIGHTNING_22077,
)
def test_epoch_boundary_resume_parity():
    """Positive check: epoch-boundary resume SHOULD reproduce the
    uninterrupted run. Strict-xfail only on the verified-affected version
    (2.6.6); everywhere else it asserts compatibility normally - a real
    pass on verified-fixed upstream, a loud failure on any other broken
    version."""
    report = _epoch_boundary_report()
    assert report.passed, report.summary


@pytest.mark.skipif(
    not _VERIFIED_AFFECTED,
    reason="bug-presence reproduction is pinned to the verified-affected "
    "lightning==2.6.6; other versions are covered by the positive parity "
    "test asserting compatibility normally",
)
def test_epoch_boundary_max_steps_resume_regression_lightning_22077():
    # Negative control: resumecheck must FLAG the bug on the
    # verified-affected version. With shuffle=False the duplicate IS the
    # same data as the expected batch, so the mismatch surfaces one step
    # later.
    report = _epoch_boundary_report()
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


@pytest.mark.parametrize("resume_at", [2, 4])
def test_epoch_limit_cannot_verify_a_truncated_run(resume_at):
    # Four batches at max_epochs=1 cannot verify a requested six-step run.
    # At resume_at=4 the old implementation returned PASS for four steps,
    # even though no post-resume optimizer step ran at all.
    with pytest.raises(ValueError, match="uninterrupted phase.*1\\.\\.6"):
        check_lightning(TinyModel, TinyDM, max_steps=6, resume_at=resume_at,
                        trainer_kwargs={"max_epochs": 1})


def test_min_steps_cannot_override_requested_crash_boundary():
    # Lightning honors min_steps before should_stop; this used to let the
    # pre-crash phase consume the full run and certify an empty resume phase.
    with pytest.raises(ValueError, match="pre-crash phase.*1\\.\\.3"):
        check_lightning(TinyModel, StatefulDM, max_steps=6, resume_at=3,
                        trainer_kwargs={"min_steps": 6})


@pytest.mark.parametrize("phase,first,last,observed", [
    ("uninterrupted", 1, 4, [1, 2]),
    ("pre-crash", 1, 2, [1, 2, 3]),
    ("post-resume", 3, 4, []),
    ("post-resume", 3, 4, [2, 3]),
    ("post-resume", 3, 4, [3, 5]),
])
def test_phase_step_validation_rejects_incomplete_or_misaligned_ranges(phase, first, last, observed):
    from resumecheck.lightning import _require_phase_steps
    from resumecheck.recorder import StepRecord, Trajectory
    with pytest.raises(ValueError, match=phase):
        _require_phase_steps(Trajectory("test", [StepRecord(step=i) for i in observed]), first, last, phase)


@pytest.mark.parametrize("first,last", [(1, 6), (1, 3), (4, 6)])
def test_phase_step_validation_accepts_exact_ranges(first, last):
    from resumecheck.lightning import _require_phase_steps
    from resumecheck.recorder import StepRecord, Trajectory
    _require_phase_steps(Trajectory("test", [StepRecord(step=i) for i in range(first, last + 1)]),
                         first, last, "control")


@pytest.mark.parametrize("drop_all", [False, True])
def test_post_resume_phase_cannot_be_empty_or_truncated(monkeypatch, drop_all):
    import resumecheck.lightning as adapter
    BaseRecorder = adapter._make_callback_cls()

    class IncompleteRecorder(BaseRecorder):
        def on_train_end(self, trainer, pl_module):
            super().on_train_end(trainer, pl_module)
            if self.recorder.trajectory.name == "resumed":
                if drop_all:
                    self.recorder.trajectory.steps.clear()
                else:
                    self.recorder.trajectory.steps.pop()

    monkeypatch.setattr(adapter, "_make_callback_cls", lambda: IncompleteRecorder)
    with pytest.raises(ValueError, match="post-resume phase.*4\\.\\.6"):
        check_lightning(TinyModel, StatefulDM, max_steps=6, resume_at=3)


def test_sufficient_epoch_limit_runs_all_requested_steps():
    report = check_lightning(TinyModel, StatefulDM, max_steps=6, resume_at=3,
                             trainer_kwargs={"max_epochs": 3})
    assert report.passed, report.summary
    assert report.steps_compared == 6
