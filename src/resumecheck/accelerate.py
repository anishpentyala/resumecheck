"""Accelerate adapter: verify resume parity end to end.

Runs the same deterministic training twice:
  1. uninterrupted to ``max_steps``
  2. up to ``resume_at``, ``save_state``, then ``load_state`` and resume
     to ``max_steps``
and compares the recorded trajectories.

Uses the documented stateful resume path:
``DataLoaderConfiguration(use_stateful_dataloader=True,
use_seedable_sampler=True)`` with ``accelerator.save_state`` /
``load_state`` - the configuration accelerate documents for resuming a
DataLoader "from where it left off".

Same scope and limitations as the other adapters: deterministic CPU
verification; parity covers recorded observables (loss, batch content
fingerprint); a PASS is scoped to the tested configuration, not a general
guarantee.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

from .compare import CompareReport, compare_trajectories
from .recorder import Trajectory, TrajectoryRecorder, fingerprint_tensor


def _require_accelerate():
    try:
        import torch  # noqa: F401
        import accelerate  # noqa: F401
        import torchdata  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "resumecheck.accelerate requires torch, accelerate and torchdata "
            "(pip install accelerate torchdata torch)"
        ) from e


def check_accelerate(
    model_factory: Callable[[], Any],
    train_dataset: Any,
    max_steps: int,
    resume_at: int,
    seed: int = 1234,
    batch_size: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    boundary_set_epoch_workaround: bool = False,
    num_workers: int = 0,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> CompareReport:
    """Verify that save_state+load_state at ``resume_at`` reproduces an
    uninterrupted ``max_steps`` run.

    ``model_factory`` must return a fresh nn.Module whose forward takes the
    dataset's batch keys and returns a scalar loss tensor (or a dict/tuple
    with a "loss" entry); ``train_dataset`` is a map-style dataset of dicts
    of tensors. Deterministic CPU only.

    ``num_workers`` (default 0): worker processes for the DataLoader.
    NOTE: with num_workers >= 1 the verified behavior of the stateful
    path changes fundamentally (see README): prepare() eagerly creates
    the iterator with the original plain RandomSampler, bypassing the
    seedable sampler, and mid-epoch resume diverges on the verified
    stacks. ``boundary_set_epoch_workaround`` is verified only with
    num_workers=0.

    ``boundary_set_epoch_workaround`` (opt-in, default off): apply the
    verified caller-side mitigation for the known epoch-boundary replay on
    accelerate's stateful path - after ``load_state``, call
    ``train_dataloader.set_epoch(resume_epoch_index)`` so the shard's
    uncheckpointed iteration counter is realigned before training
    continues. With this flag on, the check verifies that parity holds
    WITH the mitigation applied; it does not alter or excuse the raw
    documented-path finding, and it is not an upstream fix. Fail-closed:
    the mitigation is only verified at exact epoch boundaries, so the flag
    requires ``resume_at`` to be an exact multiple of the steps per epoch.

    ``artifacts_dir`` optionally retains the completed recorded trajectories,
    report and comparison settings in a new local directory below an existing
    parent. Existing paths are never replaced. Exceptions produce no bundle;
    a final manifest marks completeness. These are recorded observables, not
    model checkpoints. Review potentially sensitive evidence before sharing.
    """
    _require_accelerate()
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    from accelerate import Accelerator, DataLoaderConfiguration

    if not 0 < resume_at < max_steps:
        raise ValueError("resume_at must be in (0, max_steps)")

    if artifacts_dir is not None:
        from .artifacts import validate_artifacts_dir

        artifacts_dir = validate_artifacts_dir(artifacts_dir)

    steps_per_epoch = -(-len(train_dataset) // batch_size)  # ceil
    if boundary_set_epoch_workaround:
        if resume_at % steps_per_epoch != 0:
            raise ValueError(
                "boundary_set_epoch_workaround is verified only at exact "
                "epoch boundaries: resume_at="
                f"{resume_at} is not a multiple of steps_per_epoch="
                f"{steps_per_epoch}"
            )
        resume_epoch_index = resume_at // steps_per_epoch
    else:
        resume_epoch_index = None

    def _loss_of(out):
        if isinstance(out, dict):
            return out["loss"]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def run(
        name: str,
        target: int,
        stop_and_save_at: Optional[int] = None,
        save_dir: Optional[str] = None,
        ckpt: Optional[str] = None,
        set_epoch_on_start: Optional[int] = None,
    ) -> Trajectory:
        torch.manual_seed(seed)
        cfg = DataLoaderConfiguration(
            use_stateful_dataloader=True,
            use_seedable_sampler=True,
            data_seed=seed,
        )
        accelerator = Accelerator(cpu=True, dataloader_config=cfg)
        model = model_factory()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
        if ckpt:
            accelerator.load_state(ckpt)
        if set_epoch_on_start is not None:
            dl.set_epoch(set_epoch_on_start)

        rec = TrajectoryRecorder(name, seed=seed)
        model.train()
        step = 0
        done = False
        while not done:
            for batch in dl:
                batch_fp = fingerprint_tensor(batch)
                out = model(**batch)
                loss = _loss_of(out)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                rec.record(step=step, loss=float(loss.detach().cpu()))
                rec.trajectory.steps[-1].batch_fingerprint = batch_fp
                if stop_and_save_at is not None and step >= stop_and_save_at:
                    accelerator.save_state(save_dir)
                    done = True
                    break
                if step >= target:
                    done = True
                    break
        return rec.trajectory

    full = run("uninterrupted", max_steps)

    with tempfile.TemporaryDirectory() as tmp:
        pre = run("pre-crash", max_steps, stop_and_save_at=resume_at, save_dir=tmp)
        post = run(
            "resumed",
            max_steps - resume_at,
            ckpt=tmp,
            set_epoch_on_start=resume_epoch_index,
        )

    # offset the resumed phase's step numbers so they continue the pre-crash run
    post.steps = [
        type(s)(
            step=s.step + resume_at,
            loss=s.loss,
            batch_fingerprint=s.batch_fingerprint,
            grad_norm=s.grad_norm,
            extras=s.extras,
        )
        for s in post.steps
    ]
    resumed = Trajectory(name="resumed", meta={"seed": seed})
    resumed.steps = pre.steps + post.steps

    report = compare_trajectories(full, resumed, rtol=rtol, atol=atol)
    if artifacts_dir is not None:
        from .artifacts import save_comparison_artifacts

        save_comparison_artifacts(artifacts_dir, full, resumed, report, rtol=rtol, atol=atol)
    return report


def check_accelerate_skip_path(
    model_factory: Callable[[], Any],
    train_dataset: Any,
    max_steps: int,
    resume_at: int,
    seed: int = 1234,
    batch_size: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    skip_path_set_epoch_workaround: bool = False,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> CompareReport:
    """Verify the documented ``skip_first_batches`` resume path: run to
    ``resume_at``, ``save_state``, then ``load_state`` and resume with
    ``skip_first_batches`` for the remainder of the resume epoch (the
    pattern accelerate's checkpointing docs and by_feature example show),
    comparing against an uninterrupted ``max_steps`` run.

    Known divergence (verified 2026-10-30 on accelerate 1.15.0 and master
    1.16.0.dev0 @ 90b233c, torchdata 0.11.0): the skip wrapper consumes
    the sampler without advancing ``DataLoaderShard``'s iteration counter,
    so every post-resume epoch is shifted back one permutation.

    ``skip_path_set_epoch_workaround`` (opt-in, default off): apply the
    verified caller-side mitigation - call
    ``train_dataloader.set_epoch(epoch)`` for each epoch AFTER the resume
    epoch. Fail-closed: the mitigation is verified only for mid-epoch
    resumes, so the flag requires ``resume_at`` to land strictly inside an
    epoch. Caller-side mitigation, not an upstream fix; the raw check is
    unchanged by it.

    ``artifacts_dir`` optionally retains the completed recorded trajectories,
    report and comparison settings in a new local directory below an existing
    parent. Existing paths are never replaced. Exceptions produce no bundle;
    a final manifest marks completeness. These are recorded observables, not
    model checkpoints. Review potentially sensitive evidence before sharing.
    """
    _require_accelerate()
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    from accelerate import (
        Accelerator,
        DataLoaderConfiguration,
        skip_first_batches,
    )

    if not 0 < resume_at < max_steps:
        raise ValueError("resume_at must be in (0, max_steps)")

    if artifacts_dir is not None:
        from .artifacts import validate_artifacts_dir

        artifacts_dir = validate_artifacts_dir(artifacts_dir)

    steps_per_epoch = -(-len(train_dataset) // batch_size)  # ceil
    resume_epoch = resume_at // steps_per_epoch
    resume_batches_in_epoch = resume_at % steps_per_epoch
    if skip_path_set_epoch_workaround and resume_batches_in_epoch == 0:
        raise ValueError(
            "skip_path_set_epoch_workaround is verified only for mid-epoch "
            "resumes: resume_at="
            f"{resume_at} is an exact multiple of steps_per_epoch="
            f"{steps_per_epoch}"
        )

    def _loss_of(out):
        if isinstance(out, dict):
            return out["loss"]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def run(
        name: str,
        target: int,
        stop_and_save_at: Optional[int] = None,
        save_dir: Optional[str] = None,
        ckpt: Optional[str] = None,
        start_epoch: int = 0,
        skip_batches: int = 0,
    ) -> Trajectory:
        torch.manual_seed(seed)
        cfg = DataLoaderConfiguration(
            use_stateful_dataloader=False,
            use_seedable_sampler=True,
            data_seed=seed,
        )
        accelerator = Accelerator(cpu=True, dataloader_config=cfg)
        model = model_factory()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
        if ckpt:
            accelerator.load_state(ckpt)

        rec = TrajectoryRecorder(name, seed=seed)
        model.train()
        step = 0
        epoch = start_epoch
        first_epoch = True
        done = False
        while not done:
            if first_epoch and skip_batches:
                active = skip_first_batches(dl, skip_batches)
            else:
                if (
                    not first_epoch
                    and ckpt
                    and skip_path_set_epoch_workaround
                ):
                    dl.set_epoch(epoch)
                active = dl
            first_epoch = False
            for batch in active:
                batch_fp = fingerprint_tensor(batch)
                out = model(**batch)
                loss = _loss_of(out)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                rec.record(step=step, loss=float(loss.detach().cpu()))
                rec.trajectory.steps[-1].batch_fingerprint = batch_fp
                if stop_and_save_at is not None and step >= stop_and_save_at:
                    accelerator.save_state(save_dir)
                    done = True
                    break
                if step >= target:
                    done = True
                    break
            epoch += 1
        return rec.trajectory

    full = run("uninterrupted", max_steps)

    with tempfile.TemporaryDirectory() as tmp:
        pre = run("pre-crash", max_steps, stop_and_save_at=resume_at, save_dir=tmp)
        post = run(
            "resumed",
            max_steps - resume_at,
            ckpt=tmp,
            start_epoch=resume_epoch,
            skip_batches=resume_batches_in_epoch,
        )

    post.steps = [
        type(s)(
            step=s.step + resume_at,
            loss=s.loss,
            batch_fingerprint=s.batch_fingerprint,
            grad_norm=s.grad_norm,
            extras=s.extras,
        )
        for s in post.steps
    ]
    resumed = Trajectory(name="resumed", meta={"seed": seed})
    resumed.steps = pre.steps + post.steps

    report = compare_trajectories(full, resumed, rtol=rtol, atol=atol)
    if artifacts_dir is not None:
        from .artifacts import save_comparison_artifacts

        save_comparison_artifacts(artifacts_dir, full, resumed, report, rtol=rtol, atol=atol)
    return report
