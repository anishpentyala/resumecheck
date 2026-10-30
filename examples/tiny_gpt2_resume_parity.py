"""Integration example: resumecheck on a real pretrained model.

Model: sshleifer/tiny-gpt2 (public, ~2MB) - a real pretrained GPT-2 with
attention/dropout/optimizer state, not a toy module.
Data: SYNTHETIC token sequences (fixed-seed random ids); no external text.

What it verifies: checkpoint+resume at step 4 reproduces an uninterrupted
8-step run bit-for-bit within tight tolerance, on the real HF Trainer
integration path.

Requires: pip install torch transformers accelerate  (CPU is fine)
Runtime: ~2 minutes on CPU including the one-time model download.
Reference run: PASS, 8 steps identical within rtol=1e-5, atol=1e-7.
"""
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel

from resumecheck.hf import check_hf


class SyntheticLMDataset(Dataset):
    """Fixed-seed random token ids - SYNTHETIC, not real text."""

    def __init__(self, n=32, seq=16, vocab=50257, seed=7):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(0, vocab, (n, seq), generator=g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = self.ids[i]
        return {"input_ids": ids, "labels": ids.clone()}


def main():
    report = check_hf(
        model_factory=lambda: GPT2LMHeadModel.from_pretrained(
            "sshleifer/tiny-gpt2"),
        train_dataset=SyntheticLMDataset(),
        max_steps=8,
        resume_at=4,
        seed=99,
        training_args_kwargs={"per_device_train_batch_size": 4},
    )
    print("passed:", report.passed)
    print("steps_compared:", report.steps_compared)
    print("summary:", report.summary)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
