"""Small CPU dropout resume check with caller-owned RNG checkpoint hooks.

Synthetic constant inputs isolate RNG behavior from sampler position. This
example restores only Torch CPU RNG, not Python/NumPy/CUDA/worker RNG.
"""

import argparse
from pathlib import Path
import torch
import lightning.pytorch as pl
from resumecheck.lightning import check_lightning


class Data(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.ones(48, 1),
            batch_size=2,
            shuffle=False,
            generator=torch.Generator().manual_seed(42),
        )


class Model(pl.LightningModule):
    def __init__(self, restore_rng=True):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1)
        )
        self.pending_rng = None
        self.restore_rng = restore_rng

    def training_step(self, batch, batch_idx):
        return {"loss": (self.net(batch) - 2.0).square().mean()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["example_torch_cpu_rng"] = torch.get_rng_state()

    def on_load_checkpoint(self, checkpoint):
        if self.restore_rng:
            self.pending_rng = checkpoint["example_torch_cpu_rng"]

    # Restore after constructing the model/optimizer, before training consumes
    # randomness. The loader uses its own generator so iterator creation does
    # not advance the model RNG. This is an application-owned checkpoint hook.
    def on_train_start(self):
        if self.pending_rng is not None:
            torch.set_rng_state(self.pending_rng)
            self.pending_rng = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accumulation", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument(
        "--without-rng-restore",
        action="store_true",
        help="negative control: expect divergence",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    report = check_lightning(
        lambda: Model(restore_rng=not args.without_rng_restore),
        Data,
        4,
        2,
        seed=42,
        record_model_state=True,
        artifacts_dir=args.artifacts_dir,
        trainer_kwargs={"accumulate_grad_batches": args.accumulation},
    )
    print(report.summary)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
