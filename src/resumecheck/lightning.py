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
            self._window_fingerprints = []
            self._window_losses = []
            self._last_optimizer_step = 0

        def on_train_start(self, trainer, pl_module):
            self._last_optimizer_step = int(trainer.global_step)

        def on_train_end(self, trainer, pl_module):
            if self._window_fingerprints:
                raise ValueError("Lightning recorder stopped with an incomplete optimizer window")

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            from .recorder import fingerprint_tensor

            self._pending_batch_fp = fingerprint_tensor(batch)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = None
            if isinstance(outputs, dict):
                l = outputs.get("loss")
                loss = float(l.detach().cpu()) if l is not None else None
            self._window_fingerprints.append(self._pending_batch_fp)
            self._window_losses.append(loss)
            optimizer_step = int(trainer.global_step)
            if optimizer_step == self._last_optimizer_step:
                return
            if optimizer_step != self._last_optimizer_step + 1:
                raise ValueError("Lightning recorder requires one optimizer step per accumulation window")
            from .recorder import fingerprint_tensor

            # Record one optimizer window, including every consumed microbatch.
            # Preserve the single-batch fingerprint for the default configuration.
            window_fingerprint = (
                self._window_fingerprints[0] if len(self._window_fingerprints) == 1
                else fingerprint_tensor(self._window_fingerprints)
            )
            window_loss = (
                sum(self._window_losses) / len(self._window_losses)
                if all(value is not None for value in self._window_losses) else None
            )
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
                step=optimizer_step,
                loss=window_loss,
                batch=None,  # use the fingerprints captured before training_step
                grad_norm=grad_norm,
            )
            self.recorder.trajectory.steps[-1].batch_fingerprint = window_fingerprint
            self._last_optimizer_step = optimizer_step
            self._window_fingerprints = []
            self._window_losses = []

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

    With gradient accumulation, each record is one optimizer window: batch
    fingerprints cover every microbatch in order, loss is the mean of the
    callback-provided microbatch losses, and gradient norm is sampled at the
    window's final batch. This compares window-level loss, not each individual
    microbatch loss. Unsupported optimizer-step jumps and incomplete windows
    raise rather than silently dropping data.
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
