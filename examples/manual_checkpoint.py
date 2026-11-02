"""CPU-only checkpoint example with explicit state and omission controls.

No framework restores state here: the application owns the model, AdamW,
MultiStepLR, Torch CPU RNG, and a single fixed permutation plus batch cursor.
Synthetic data, no downloads, no workers, no distributed/CUDA support.
"""
import argparse
import copy
import tempfile
from pathlib import Path

import torch

from resumecheck.artifacts import save_comparison_artifacts, validate_artifacts_dir
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import TrajectoryRecorder, fingerprint_tensor


COMPONENTS = ("model", "optimizer", "scheduler", "rng", "sampler")


def make_training(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Dropout(0.25),
        torch.nn.Linear(8, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.3)
    # This generator never consumes the model's global dropout RNG.
    order = torch.randperm(32, generator=torch.Generator().manual_seed(seed + 1))
    return model, optimizer, scheduler, order


def train_steps(model, optimizer, scheduler, order, start, stop, recorder):
    data = torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64
    targets = data[:, :1] * 2 - data[:, 1:]
    model.train()
    for step in range(start, stop):
        indices = order[step * 4:(step + 1) * 4]
        batch = {"x": data[indices], "y": targets[indices]}
        loss = (model(batch["x"]) - batch["y"]).square().mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        recorder.record(step + 1, loss=float(loss.detach()), batch=batch,
                        grad_norm=float(grad_norm))
        recorder.trajectory.steps[-1].model_fingerprint = fingerprint_tensor(model.state_dict())


def check_manual(omit=None, seed=42, artifacts_dir=None):
    if omit is not None and omit not in COMPONENTS:
        raise ValueError("unsupported omitted checkpoint component")
    if artifacts_dir is not None:
        artifacts_dir = validate_artifacts_dir(artifacts_dir)
    full = TrajectoryRecorder("uninterrupted", seed=seed)
    train_steps(*make_training(seed), 0, 6, full)
    resumed = TrajectoryRecorder("resumed", seed=seed)
    model, optimizer, scheduler, order = make_training(seed)
    train_steps(model, optimizer, scheduler, order, 0, 2, resumed)
    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "rng": torch.get_rng_state(), "order": order, "cursor": 2,
    }
    # Exercise a real serialized checkpoint, not shared in-memory tensors.
    # Never load untrusted checkpoint files. This file is created by this run.
    with tempfile.TemporaryDirectory(prefix="resumecheck-manual-") as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save(checkpoint, path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer, scheduler, order = make_training(seed)
    if omit != "model":
        model.load_state_dict(saved["model"])
    # Construct scheduler before loading optimizer state, then restore both.
    if omit != "optimizer":
        optimizer.load_state_dict(saved["optimizer"])
    if omit != "scheduler":
        scheduler.load_state_dict(saved["scheduler"])
    cursor = 0 if omit == "sampler" else saved["cursor"]
    if omit != "sampler":
        order = saved["order"]
    if omit != "rng":
        torch.set_rng_state(saved["rng"])
    # Label records by completed optimizer step, not the restored data cursor.
    post = TrajectoryRecorder("post-resume", seed=seed)
    train_steps(model, optimizer, scheduler, order, cursor, cursor + 4, post)
    for offset, record in enumerate(post.trajectory.steps, start=3):
        record.step = offset
    resumed.trajectory.steps.extend(post.trajectory.steps)
    report = compare_trajectories(full.trajectory, resumed.trajectory)
    if artifacts_dir is not None:
        save_comparison_artifacts(artifacts_dir, full.trajectory, resumed.trajectory,
                                  report, rtol=1e-5, atol=1e-7)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omit", choices=COMPONENTS, help="negative control: expect divergence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args(argv)
    report = check_manual(args.omit, args.seed, args.artifacts_dir)
    print(report.summary)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
