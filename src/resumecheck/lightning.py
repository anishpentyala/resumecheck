"""PyTorch Lightning adapter: verify resume parity end to end.

Runs the same deterministic training twice:
  1. uninterrupted to ``max_steps``
  2. up to ``resume_at``, checkpoint, then resume to ``max_steps``
and compares the recorded trajectories.

Scope and limitations (read before trusting a PASS):
- Both phases run in the same process with the same seed; model
  initialization must be a deterministic function of the seed.
- Parity covers what is recorded: loss values, batch content
  fingerprints, and gradient norms. Optimizer/RNG/sampler state matter
  only insofar as they change those observables.
- Nondeterministic kernels (some CUDA ops), multi-worker dataloader
  timing, and distributed training can produce legitimate differences
  without any defect. This harness is designed for deterministic CPU
  verification; widen ``rtol``/``atol`` only with a stated reason.
- A PASS means "these observables matched on this setup", not
  "resume is correct for every configuration".
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from .compare import CompareReport, compare_trajectories
from .recorder import Trajectory, TrajectoryRecorder


def _require_lightning():
    try:
        import lightning.pytorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "resumecheck.lightning requires torch and lightning "
            "(pip install resumecheck[lightning])"
        ) from e


def _make_callback_cls():
    from lightning.pytorch.callbacks import Callback

    class RecorderCallback(Callback):
        def __init__(self, recorder: TrajectoryRecorder) -> None:
            self.recorder = recorder
            self._pending_batch_fp: Optional[str] = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            from .recorder import fingerprint_tensor

            self._pending_batch_fp = fingerprint_tensor(batch)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = None
            if isinstance(outputs, dict):
                l = outputs.get("loss")
                loss = float(l.detach().cpu()) if l is not None else None
            grad_norm = None
            total = 0.0
            seen = False
            for p in pl_module.parameters():
                if p.grad is not None:
                    total += float(p.grad.detach().norm().cpu()) ** 2
                    seen = True
            if seen:
                grad_norm = math.sqrt(total)
            self.recorder.record(
                step=int(trainer.global_step),
                loss=loss,
                batch=batch,  # fingerprint already taken; recompute avoided below
                grad_norm=grad_norm,
            )
            # use the cached fingerprint instead of the recomputed one
            self.recorder.trajectory.steps[-1].batch_fingerprint = (
                self._pending_batch_fp
            )

    return RecorderCallback


def check_lightning(
    module_factory: Callable[[], Any],
    datamodule_factory: Callable[[], Any],
    max_steps: int,
    resume_at: int,
    seed: int = 1234,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> CompareReport:
    """Verify that checkpoint+resume at ``resume_at`` reproduces an
    uninterrupted ``max_steps`` run for the given factories.

    ``module_factory``/``datamodule_factory`` must return fresh, identically
    seeded instances on every call. Returns a CompareReport.
    """
    _require_lightning()
    import lightning.pytorch as pl

    if not 0 < resume_at < max_steps:
        raise ValueError("resume_at must be in (0, max_steps)")

    import tempfile as _tf
    kwargs = {
        "default_root_dir": _tf.mkdtemp(prefix="resumecheck-"),
        "accelerator": "cpu",
        "devices": 1,
        "enable_checkpointing": True,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }
    if trainer_kwargs:
        kwargs.update(trainer_kwargs)

    RecorderCallback = _make_callback_cls()

    def run(name: str, steps: int, ckpt_path: Optional[str] = None) -> Trajectory:
        pl.seed_everything(seed, workers=True)
        rec = TrajectoryRecorder(name, seed=seed)
        cb = RecorderCallback(rec)
        trainer = pl.Trainer(max_steps=steps, callbacks=[cb], **kwargs)
        trainer.fit(module_factory(), datamodule_factory(), ckpt_path=ckpt_path)
        return rec.trajectory

    # Phase 1: uninterrupted
    full = run("uninterrupted", max_steps)

    # Phase 2: pre-crash segment, then resume
    import tempfile, os

    class _StopAt:
        pass

    from lightning.pytorch.callbacks import Callback as _Cb

    class StopAtCallback(_Cb):
        """Simulate a crash: request training stop after step ``stop_at``.

        The trainer is configured for the FINAL max_steps so any scheduler or
        training-length-dependent setup matches the uninterrupted run; only
        the stop is early.
        """

        def __init__(self, stop_at: int) -> None:
            self.stop_at = stop_at

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step >= self.stop_at:
                trainer.should_stop = True

    with tempfile.TemporaryDirectory() as tmp:
        pl.seed_everything(seed, workers=True)
        rec_a = TrajectoryRecorder("pre-crash", seed=seed)
        cb_a = RecorderCallback(rec_a)
        trainer_a = pl.Trainer(max_steps=max_steps, callbacks=[cb_a, StopAtCallback(resume_at)], **kwargs)
        trainer_a.fit(module_factory(), datamodule_factory())
        ckpt = os.path.join(tmp, "crash.ckpt")
        trainer_a.save_checkpoint(ckpt)

        rec_b = TrajectoryRecorder("resumed", seed=seed)
        cb_b = RecorderCallback(rec_b)
        trainer_b = pl.Trainer(max_steps=max_steps, callbacks=[cb_b], **kwargs)
        trainer_b.fit(module_factory(), datamodule_factory(), ckpt_path=ckpt)

    resumed = Trajectory(name="resumed", meta={"seed": seed})
    resumed.steps = rec_a.trajectory.steps + rec_b.trajectory.steps

    return compare_trajectories(full, resumed, rtol=rtol, atol=atol)
