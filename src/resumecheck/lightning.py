"""PyTorch Lightning adapter: verify resume parity end to end.

Runs the same deterministic training twice:
  1. uninterrupted to ``max_steps``
  2. up to ``resume_at``, checkpoint, then resume to ``max_steps``
and compares the recorded trajectories.

Scope and limitations (read before trusting a PASS):
- Both phases run in the same process with the same seed; model
  initialization must be a deterministic function of the seed.
- Parity covers what is recorded: loss values, batch content
  fingerprints, and gradient norms. Optional exact model-state fingerprints
  include parameters and persistent buffers. Optimizer/RNG/sampler state matter
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
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

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
        def __init__(self, recorder: TrajectoryRecorder, record_model_state: bool = False,
                     record_learning_rates: bool = False,
                     record_optimizer_state: bool = False) -> None:
            self.record_optimizer_state = record_optimizer_state
            self.record_model_state = record_model_state
            self.record_learning_rates = record_learning_rates
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
            if self.record_optimizer_state:
                self.recorder.trajectory.steps[-1].optimizer_fingerprint = fingerprint_tensor(
                    [optimizer.state_dict() for optimizer in trainer.optimizers])
            if self.record_model_state:
                self.recorder.trajectory.steps[-1].model_fingerprint = fingerprint_tensor(pl_module.state_dict())
            if self.record_learning_rates:
                rates = {}
                for index, optimizer in enumerate(trainer.optimizers):
                    for name, value in _optimizer_rates(optimizer).items():
                        rates[f"optimizer{index}.{name}"] = value
                if not rates:
                    raise ValueError("no optimizer learning rates were recorded")
                self.recorder.trajectory.steps[-1].extras.update(rates)
            self._last_optimizer_step = optimizer_step
            self._window_fingerprints = []
            self._window_losses = []

    return RecorderCallback


def _require_phase_steps(trajectory: Trajectory, first: int, last: int, phase: str) -> None:
    """Never certify equal prefixes when a requested training phase did not run."""
    observed = [record.step for record in trajectory.steps]
    expected_count = last - first + 1
    if len(observed) != expected_count or any(
        step != first + offset for offset, step in enumerate(observed)
    ):
        raise ValueError(
            f"Lightning {phase} phase did not record the requested optimizer steps "
            f"{first}..{last}: got {observed}. Resume parity was not verified; "
            "check epoch limits, stopping callbacks, and the available training data."
        )


def check_lightning(
    module_factory: Callable[[], Any],
    datamodule_factory: Callable[[], Any],
    max_steps: int,
    resume_at: int,
    seed: int = 1234,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    artifacts_dir: Optional[Union[str, Path]] = None,
    record_model_state: bool = False,
    record_learning_rates: bool = False,
    record_optimizer_state: bool = False,
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
    raise rather than silently dropping data. Every phase must record its exact
    requested optimizer-step range. Epoch limits or other stopping conditions
    that prevent this raise ValueError instead of certifying a shorter run.

    ``record_model_state=True`` additionally fingerprints ``pl_module.state_dict()``
    at each completed optimizer window. Parameters and persistent buffers are
    compared exactly, not with loss tolerances. This is opt-in because hashing
    model state at every step can be expensive. Unsupported custom extra-state
    values raise rather than being omitted. It does not fingerprint optimizer,
    scheduler, RNG or nonpersistent-buffer state, and does not validate arbitrary
    manual-optimization or distributed configurations. The default observables
    and legacy JSON shape are unchanged.

    ``record_learning_rates=True`` additionally samples each optimizer group's
    learning rate at the completed-window batch-end callback. These numeric
    extras use the comparison's rtol/atol and are retained for artifact replay.
    Missing/invalid rates fail closed. This samples the rate visible at that
    callback, not every scheduler update (epoch/metric-driven updates may occur
    later), and does not verify full scheduler or optimizer state. Default off
    preserves the existing comparison scope and artifact schema.

    ``artifacts_dir`` optionally retains the completed recorded trajectories,
    report and comparison settings in a new local directory below an existing
    parent. Existing paths are never replaced. Exceptions produce no bundle;
    a final manifest marks completeness. These are recorded observables, not
    model checkpoints. Review potentially sensitive evidence before sharing.

    ``record_optimizer_state=True`` additionally fingerprints the complete
    optimizer ``state_dict()`` after each recorded update (including parameter
    groups and moments). Fingerprints are compared exactly, not with loss
    tolerances, and persisted in schema-4 evidence. This copies optimizer tensors
    to CPU and adds time/memory cost proportional to state size. Unsupported
    custom values raise instead of being skipped. Default off is unchanged.
    This observes state_dict contents, not parameter identity, scheduler/RNG,
    sharded/distributed state or an arbitrary optimizer implementation's hidden
    state. Compare like-for-like deterministic runs with stable group ordering.

    """
    if type(record_optimizer_state) is not bool:
        raise TypeError("record_optimizer_state must be boolean")

    from .validation import validate_step_limits

    max_steps, resume_at = validate_step_limits(max_steps, resume_at)
    _require_lightning()
    import lightning.pytorch as pl

    if type(record_learning_rates) is not bool:
        raise TypeError("record_learning_rates must be boolean")
    if type(record_model_state) is not bool:
        raise TypeError("record_model_state must be boolean")


    if artifacts_dir is not None:
        from .artifacts import validate_artifacts_dir

        artifacts_dir = validate_artifacts_dir(artifacts_dir)

    import tempfile as _tf
    with _tf.TemporaryDirectory(prefix="resumecheck-") as output_root:
        kwargs = {
            "default_root_dir": output_root,
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
            cb = RecorderCallback(rec, record_model_state, record_learning_rates, record_optimizer_state)
            trainer = pl.Trainer(max_steps=steps, callbacks=[cb], **kwargs)
            trainer.fit(module_factory(), datamodule_factory(), ckpt_path=ckpt_path)
            return rec.trajectory

        # Phase 1: uninterrupted
        full = run("uninterrupted", max_steps)
        _require_phase_steps(full, 1, max_steps, "uninterrupted")

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
            cb_a = RecorderCallback(rec_a, record_model_state, record_learning_rates, record_optimizer_state)
            trainer_a = pl.Trainer(max_steps=max_steps, callbacks=[cb_a, StopAtCallback(resume_at)], **kwargs)
            trainer_a.fit(module_factory(), datamodule_factory())
            _require_phase_steps(rec_a.trajectory, 1, resume_at, "pre-crash")
            ckpt = os.path.join(tmp, "crash.ckpt")
            trainer_a.save_checkpoint(ckpt)

            rec_b = TrajectoryRecorder("resumed", seed=seed)
            cb_b = RecorderCallback(rec_b, record_model_state, record_learning_rates, record_optimizer_state)
            trainer_b = pl.Trainer(max_steps=max_steps, callbacks=[cb_b], **kwargs)
            trainer_b.fit(module_factory(), datamodule_factory(), ckpt_path=ckpt)
            _require_phase_steps(rec_b.trajectory, resume_at + 1, max_steps, "post-resume")

        resumed = Trajectory(name="resumed", meta={"seed": seed})
        resumed.steps = rec_a.trajectory.steps + rec_b.trajectory.steps

        selected_rates = sorted({name for trajectory in (full, resumed)
                                 for record in trajectory.steps for name in record.extras}) if record_learning_rates else None
        if record_learning_rates and not selected_rates:
            raise ValueError("no optimizer learning rates were recorded")
        report = compare_trajectories(full, resumed, rtol=rtol, atol=atol,
                                      check_extras=selected_rates)
        if artifacts_dir is not None:
            from .artifacts import save_comparison_artifacts

            save_comparison_artifacts(artifacts_dir, full, resumed, report, rtol=rtol, atol=atol,
                                      check_extras=selected_rates)
        return report


def _optimizer_rates(optimizer) -> dict[str, float]:
    """Post-step optimizer-group rates, not a scheduler-state fingerprint."""
    import math
    import numbers
    import torch

    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("optimizer has no learning-rate parameter groups")
    rates = {}
    for index, group in enumerate(groups):
        value = group.get("lr")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or value.ndim != 0 or value.dtype == torch.bool:
                raise ValueError("optimizer learning rate must be a finite nonnegative scalar")
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError("optimizer learning rate must be a finite nonnegative scalar")
        try:
            valid = math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("optimizer learning rate must be a finite nonnegative scalar")
        rates[f"lr.group{index}"] = float(value)
    return rates
