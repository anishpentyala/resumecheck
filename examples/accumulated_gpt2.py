"""Offline HF Trainer resume example with gradient accumulation.

Uses a small, randomly initialized GPT-2 architecture and fixed synthetic token
IDs. No model, tokenizer, weights or dataset is downloaded. This demonstrates
recorded loss/batch parity on deterministic CPU, not pretrained-model quality,
GPU performance or general checkpoint correctness.

    pip install -e '.[hf]'
    python examples/accumulated_gpt2.py --accumulation 3

A nonzero exit means parity was not verified. The standard model's dropout is
left enabled, so Trainer's checkpoint RNG restoration participates in the check.
"""
import argparse

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from resumecheck.hf import check_hf


class SyntheticTokens(Dataset):
    def __init__(self):
        self.ids = torch.randint(0, 32, (24, 8), generator=torch.Generator().manual_seed(7))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ids = self.ids[index]
        return {'input_ids': ids, 'labels': ids.clone()}


def fresh_model():
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=.1, embd_pdrop=.1, attn_pdrop=.1,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accumulation', type=int, default=3,
                        help='training microbatches per optimizer step (default 3)')
    parser.add_argument('--artifacts-dir', help='new local directory for replayable trajectory evidence')
    args = parser.parse_args(argv)
    if args.accumulation < 1:
        parser.error('--accumulation must be positive')
    torch.set_num_threads(1)
    report = check_hf(
        fresh_model, SyntheticTokens(), max_steps=4, resume_at=2, seed=99,
        artifacts_dir=args.artifacts_dir,
        training_args_kwargs={'per_device_train_batch_size': 2,
                              'gradient_accumulation_steps': args.accumulation},
    )
    print('Random initialization, synthetic tokens, CPU; no pretrained weights.')
    print(report.summary)
    return 0 if report.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
