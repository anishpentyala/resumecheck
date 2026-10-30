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

from typing import Any, Callable, Optional

from .compare import CompareReport, compare_trajectories
from .recorder import Trajectory, TrajectoryRecorder


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
) -> CompareReport:
    """Verify that save_state+load_state at ``resume_at`` reproduces an
    uninterrupted ``max_steps`` run.

    ``model_factory`` must return a fresh nn.Module whose forward takes the
    dataset's batch keys and returns a scalar loss tensor (or a dict/tuple
    with a "loss" entry); ``train_dataset`` is a map-style dataset of dicts
    of tensors. Deterministic CPU only.
    """
    _require_accelerate()
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    from accelerate import Accelerator, DataLoaderConfiguration

    if not 0 < resume_at < max_steps:
        raise ValueError("resume_at must be in (0, max_steps)")

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
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
        if ckpt:
            accelerator.load_state(ckpt)

        rec = TrajectoryRecorder(name, seed=seed)
        model.train()
        step = 0
        done = False
        while not done:
            for batch in dl:
                out = model(**batch)
                loss = _loss_of(out)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                rec.record(step=step, loss=float(loss.detach().cpu()), batch=batch)
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
        post = run("resumed", max_steps - resume_at, ckpt=tmp)

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

    return compare_trajectories(full, resumed, rtol=rtol, atol=atol)
