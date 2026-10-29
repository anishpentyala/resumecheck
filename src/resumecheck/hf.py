"""Hugging Face Trainer adapter: verify resume parity end to end.

Runs the same deterministic training twice:
  1. uninterrupted to ``max_steps``
  2. up to ``resume_at``, checkpoint, then resume to ``max_steps``
and compares the recorded trajectories.

Same scope and limitations as the Lightning adapter: deterministic CPU
verification; parity covers recorded observables (loss, batch content
fingerprint); a PASS is scoped to the tested configuration, not a general
guarantee. Resume correctness here depends on the Trainer restoring its
sampler/dataloader skip logic, RNG, and optimizer state from the checkpoint.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .compare import CompareReport, compare_trajectories
from .recorder import Trajectory, TrajectoryRecorder, fingerprint_tensor


def _require_hf():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "resumecheck.hf requires torch and transformers "
            "(pip install transformers torch)"
        ) from e


def _make_trainer_cls(recorder: TrajectoryRecorder):
    from transformers import Trainer

    class RecordingTrainer(Trainer):
        """Trainer that records per-step loss and batch fingerprints.

        Steps are recorded at optimizer-step granularity via compute_loss;
        with gradient_accumulation_steps=1 (the only supported setting here)
        that is one record per optimizer step.
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            fp = fingerprint_tensor(inputs)
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            self._rc_step += 1
            recorder.record(
                step=self._rc_step,
                loss=float(loss.detach().cpu()),
                batch=inputs,
            )
            self._pending_fp = fp
            return (loss, outputs) if return_outputs else loss

        _rc_step = 0

    return RecordingTrainer


def check_hf(
    model_factory: Callable[[], Any],
    train_dataset: Any,
    max_steps: int,
    resume_at: int,
    seed: int = 1234,
    training_args_kwargs: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> CompareReport:
    """Verify that checkpoint+resume at ``resume_at`` reproduces an
    uninterrupted ``max_steps`` run.

    ``model_factory`` must return a fresh nn.Module whose forward returns a
    dict (or tuple) with a "loss" entry; ``train_dataset`` is a map-style
    dataset of dicts of tensors. Deterministic CPU only.
    """
    _require_hf()
    import tempfile

    from transformers import Trainer, TrainingArguments, set_seed

    if not 0 < resume_at < max_steps:
        raise ValueError("resume_at must be in (0, max_steps)")

    base_args: Dict[str, Any] = {
        "output_dir": tempfile.mkdtemp(prefix="resumecheck-hf-"),
        "max_steps": max_steps,
        "per_device_train_batch_size": 2,
        "seed": seed,
        "full_determinism": True,
        "use_cpu": True,
        "report_to": [],
        "disable_tqdm": True,
        "logging_steps": 10**9,
        "save_strategy": "no",
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 0,
    }
    if training_args_kwargs:
        base_args.update(training_args_kwargs)

    def run(name: str, steps: int, ckpt: Optional[str] = None, save_at_end: bool = False) -> Trajectory:
        set_seed(seed)
        rec = TrajectoryRecorder(name, seed=seed)
        RecordingTrainer = _make_trainer_cls(rec)
        args_dict = dict(base_args, max_steps=steps)
        if save_at_end:
            args_dict.update(save_strategy="steps", save_steps=resume_at, save_total_limit=1)
        trainer = RecordingTrainer(
            model=model_factory(),
            args=TrainingArguments(**args_dict),
            train_dataset=train_dataset,
        )
        trainer.train(resume_from_checkpoint=ckpt)
        if save_at_end:
            # ensure the checkpoint exists exactly at resume_at
            import glob, os

            ckpts = sorted(glob.glob(os.path.join(args_dict["output_dir"], "checkpoint-*")))
            assert ckpts, "expected a checkpoint at resume_at"
            return rec.trajectory
        return rec.trajectory

    full = run("uninterrupted", max_steps)

    from transformers import TrainerCallback

    class _StopAt(TrainerCallback):
        """Simulate a crash: stop training after step ``stop_at``."""

        def __init__(self, stop_at: int) -> None:
            self.stop_at = stop_at

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.stop_at:
                control.should_training_stop = True
            return control

    with tempfile.TemporaryDirectory() as tmp:
        set_seed(seed)
        rec_a = TrajectoryRecorder("pre-crash", seed=seed)
        RecordingTrainer = _make_trainer_cls(rec_a)
        # configured for the FINAL max_steps so scheduler/training plan match
        # the uninterrupted run; only the stop is early
        args_a = TrainingArguments(
            **dict(base_args, output_dir=tmp, max_steps=max_steps,
                   save_strategy="steps", save_steps=resume_at, save_total_limit=1)
        )
        trainer_a = RecordingTrainer(
            model=model_factory(), args=args_a, train_dataset=train_dataset,
            callbacks=[_StopAt(resume_at)],
        )
        trainer_a.train()
        import glob, os

        ckpts = glob.glob(os.path.join(tmp, "checkpoint-*"))
        assert ckpts, f"no checkpoint saved at step {resume_at} in {tmp}"

        rec_b = TrajectoryRecorder("resumed", seed=seed)
        RecordingTrainerB = _make_trainer_cls(rec_b)
        trainer_b = RecordingTrainerB(
            model=model_factory(),
            args=TrainingArguments(**dict(base_args, max_steps=max_steps)),
            train_dataset=train_dataset,
        )
        trainer_b.train(resume_from_checkpoint=ckpts[0])

    resumed = Trajectory(name="resumed", meta={"seed": seed})
    # offset the resumed phase's step numbers so they continue the pre-crash run
    rec_b.trajectory.steps = [
        type(s)(
            step=s.step + resume_at,
            loss=s.loss,
            batch_fingerprint=s.batch_fingerprint,
            grad_norm=s.grad_norm,
            extras=s.extras,
        )
        for s in rec_b.trajectory.steps
    ]
    resumed.steps = rec_a.trajectory.steps + rec_b.trajectory.steps

    return compare_trajectories(full, resumed, rtol=rtol, atol=atol)
