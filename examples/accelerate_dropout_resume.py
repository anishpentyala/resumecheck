"""Offline CPU dropout-MLP resume checks with explicit loader RNG isolation.

No weights or datasets are downloaded. Stateful mid-epoch, stateful boundary
and skip-path mid-epoch runs span multiple epochs. Sampler workarounds are
explicit; this is a diagnostic example, not a production training recipe.
"""
import argparse
from pathlib import Path

import torch
from resumecheck.accelerate import check_accelerate, check_accelerate_skip_path


class Data(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"x": torch.tensor([float(index) / 16]),
                "y": torch.tensor([float(index) / 8 + 1])}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 8), torch.nn.ReLU(), torch.nn.Dropout(.2),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x, y):
        return {"loss": (self.net(x) - y).square().mean()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["stateful", "boundary", "skip"], default="stateful")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--without-loader-isolation", action="store_true",
                        help="negative control: expect dropout/model-state divergence")
    parser.add_argument("--without-sampler-workaround", action="store_true",
                        help="negative control for boundary/skip paths: expect batch divergence")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.path == "stateful" and args.without_sampler_workaround:
        parser.error("stateful mid-epoch path has no sampler workaround to disable")
    kwargs = {}
    if not args.without_sampler_workaround:
        if args.path == "boundary":
            kwargs["boundary_set_epoch_workaround"] = True
        elif args.path == "skip":
            kwargs["skip_path_set_epoch_workaround"] = True
    check = check_accelerate_skip_path if args.path == "skip" else check_accelerate
    report = check(
        Model, Data(), max_steps=19, resume_at=8 if args.path == "boundary" else 3,
        seed=args.seed, record_model_state=True,
        isolate_dataloader_rng=not args.without_loader_isolation,
        artifacts_dir=args.artifacts_dir, **kwargs,
    )
    print(report.summary)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
