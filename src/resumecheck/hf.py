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

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

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


def _make_trainer_cls(recorder: TrajectoryRecorder, record_model_state: bool = False):
    from transformers import Trainer, TrainerCallback

    class RecordingTrainer(Trainer):
        """Record every consumed microbatch in one optimizer-window record."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._rc_fingerprints = []
            self._rc_losses = []
            self._rc_recording = False
            owner = self

            class WindowCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if not owner._rc_fingerprints:
                        raise ValueError("HF optimizer step has no recorded training microbatches")
                    fingerprints = owner._rc_fingerprints
                    fp = fingerprints[0] if len(fingerprints) == 1 else fingerprint_tensor(fingerprints)
                    recorder.record(step=state.global_step, loss=sum(owner._rc_losses) / len(owner._rc_losses))
                    recorder.trajectory.steps[-1].batch_fingerprint = fp
                    if record_model_state:
                        recorder.trajectory.steps[-1].model_fingerprint = fingerprint_tensor(owner.model.state_dict())
                    owner._rc_fingerprints = []
                    owner._rc_losses = []

                def on_train_end(self, args, state, control, **kwargs):
                    if owner._rc_fingerprints:
                        raise ValueError("HF recorder stopped with an incomplete optimizer window")

            self.add_callback(WindowCallback())

        def training_step(self, *args, **kwargs):
            self._rc_recording = True
            try:
                return super().training_step(*args, **kwargs)
            finally:
                self._rc_recording = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            fp = fingerprint_tensor(inputs) if self._rc_recording else None
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if self._rc_recording:
                self._rc_fingerprints.append(fp)
                self._rc_losses.append(float(loss.detach().cpu()))
            return (loss, outputs) if return_outputs else loss

    return RecordingTrainer


def _require_phase_steps(trajectory: Trajectory, first: int, last: int, phase: str) -> None:
    observed = [record.step for record in trajectory.steps]
    if len(observed) != last - first + 1 or any(step != first + offset for offset, step in enumerate(observed)):
        raise ValueError(
            f"HF {phase} phase did not record the requested optimizer steps "
            f"{first}..{last}: got {observed}. Resume parity was not verified."
        )


def check_hf(
    model_factory: Callable[[], Any],
    train_dataset: Any,
    max_steps: int,
    resume_at: int,
    seed: int = 1234,
    training_args_kwargs: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    artifacts_dir: Optional[Union[str, Path]] = None,
    record_model_state: bool = False,
) -> CompareReport:
    """Verify that checkpoint+resume at ``resume_at`` reproduces an
    uninterrupted ``max_steps`` run.

    ``model_factory`` must return a fresh nn.Module whose forward returns a
    dict (or tuple) with a "loss" entry; ``train_dataset`` is a map-style
    dataset of dicts of tensors. Deterministic CPU only. One trajectory record
    covers one optimizer step, including every consumed training microbatch
    fingerprint in order and the mean of its raw model losses. This compares
    window-level loss, not individual microbatch losses or Trainer's normalized
    backward loss. Evaluation calls do not enter training windows. Every phase
    must record its exact requested optimizer-step range; incomplete windows
    and missing phases raise instead of certifying a shorter run.

    ``record_model_state=True`` additionally fingerprints ``model.state_dict()``
    after each optimizer step. This is an exact-content check of parameters and
    persistent buffers, independent of loss tolerances, and can detect changes
    hidden by equal losses. It adds CPU copies and hashing work proportional to
    model size. Unsupported custom state raises; optimizer, scheduler, RNG and
    nonpersistent buffers are not directly checked. Deterministic CPU only.

    ``artifacts_dir`` optionally saves both completed trajectories, the report,
    and comparison settings for offline diagnosis. It must not exist, and its
    parent must exist. No model checkpoint is retained. The directory is only
    written after a completed comparison, including a divergent result; training
    exceptions propagate without a bundle. Recorded losses, fingerprints and
    metadata may be sensitive: review before sharing.
    """
    _require_hf()
    import tempfile

    from transformers import TrainingArguments, set_seed

    if type(record_model_state) is not bool:
        raise TypeError("record_model_state must be boolean")

    if not 0 < resume_at < max_steps:
        raise ValueError("resume_at must be in (0, max_steps)")

    if artifacts_dir is not None:
        from .artifacts import validate_artifacts_dir

        artifacts_dir = validate_artifacts_dir(artifacts_dir)

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

    def run(name: str, steps: int) -> Trajectory:
        set_seed(seed)
        rec = TrajectoryRecorder(name, seed=seed)
        RecordingTrainer = _make_trainer_cls(rec, record_model_state=True) if record_model_state else _make_trainer_cls(rec)
        args_dict = dict(base_args, max_steps=steps)
        trainer = RecordingTrainer(
            model=model_factory(),
            args=TrainingArguments(**args_dict),
            train_dataset=train_dataset,
        )
        trainer.train()
        return rec.trajectory

    full = run("uninterrupted", max_steps)
    _require_phase_steps(full, 1, max_steps, "uninterrupted")

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
        RecordingTrainer = _make_trainer_cls(rec_a, record_model_state=True) if record_model_state else _make_trainer_cls(rec_a)
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
        _require_phase_steps(rec_a.trajectory, 1, resume_at, "pre-crash")
        import glob, os

        ckpts = glob.glob(os.path.join(tmp, "checkpoint-*"))
        assert ckpts, f"no checkpoint saved at step {resume_at} in {tmp}"

        rec_b = TrajectoryRecorder("resumed", seed=seed)
        RecordingTrainerB = _make_trainer_cls(rec_b, record_model_state=True) if record_model_state else _make_trainer_cls(rec_b)
        trainer_b = RecordingTrainerB(
            model=model_factory(),
            args=TrainingArguments(**dict(base_args, max_steps=max_steps)),
            train_dataset=train_dataset,
        )
        trainer_b.train(resume_from_checkpoint=ckpts[0])
        _require_phase_steps(rec_b.trajectory, resume_at + 1, max_steps, "post-resume")

    resumed = Trajectory(name="resumed", meta={"seed": seed})
    resumed.steps = rec_a.trajectory.steps + rec_b.trajectory.steps

    report = compare_trajectories(full, resumed, rtol=rtol, atol=atol)
    if artifacts_dir is not None:
        from .artifacts import save_comparison_artifacts

        save_comparison_artifacts(artifacts_dir, full, resumed, report, rtol=rtol, atol=atol)
    return report
