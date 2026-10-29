# resumecheck

Verify that **checkpoint-and-resume training matches uninterrupted training**.

If your training job stops at step *k* and you resume from the checkpoint, do
you get the same model you would have gotten by training straight through?
Silent state-restoration bugs — lost dataloader position, replayed or skipped
batches, wrong RNG — corrupt long training runs **without raising any error**.
resumecheck records what a training run actually did and compares it against
the same run interrupted and resumed, reporting the first step where they
diverge.

It exists because these bugs are real and recurring: while building it, its
own end-to-end test caught a silent batch-duplication bug in current
PyTorch Lightning ([Lightning-AI/pytorch-lightning#22077](https://github.com/Lightning-AI/pytorch-lightning/issues/22077)),
and the resume-bug track record of that ecosystem
([litData#775](https://github.com/Lightning-AI/litData/issues/775) and the
lightning resume-issue cluster) shows how often practitioners hit them.

## How it works

1. Run your deterministic training to `max_steps`, recording a *trajectory*:
   per-step loss, a content fingerprint of each batch, and gradient norm.
2. Run it again: stop at `resume_at`, save a checkpoint, resume to `max_steps`
   in a fresh trainer.
3. Compare. The report names the first divergent step and field — e.g.
   *"different training data consumed at step 5 — dataloader position or
   shuffling state was not restored"*.

```python
from resumecheck.lightning import check_lightning

report = check_lightning(
    module_factory=TinyModel,        # fresh, seed-deterministic instances
    datamodule_factory=TinyDM,
    max_steps=100,
    resume_at=40,
    seed=42,
)
print(report.summary)
assert report.passed
```

There is also a framework-agnostic core (`TrajectoryRecorder` +
`compare_trajectories`) you can wire into any training loop, and a CLI:

```bash
resumecheck compare uninterrupted.json resumed.json
```

## What a PASS means — and what it does not

- Parity covers what is recorded: losses, batch content fingerprints, gradient
  norms. Optimizer/RNG/sampler/worker state matter only insofar as they change
  those observables.
- A PASS means *"these observables matched on this setup"*, not *"resume is
  correct for every configuration"*.
- Designed for **deterministic CPU verification**. Nondeterministic kernels
  (some CUDA ops), multi-worker dataloader timing, and distributed training
  can produce legitimate differences without any defect. Widen `rtol`/`atol`
  only with a stated reason.
- Some divergences are documented framework behavior, not bugs: PyTorch
  Lightning warns that resuming mid-epoch with a non-stateful DataLoader
  replays the epoch. resumecheck flags those too — deciding whether the
  divergence is acceptable is your call, and the test suite encodes known
  upstream cases as executable specs (see `tests/test_lightning_parity.py`).

## Install & test

```bash
pip install -e ".[test]"   # torch + lightning included in test extra
pytest tests/
```

## Status

Early prototype. Core compare engine and recorder are stable; the Lightning
adapter covers single-process CPU training. Synthetic fixtures only — no real
datasets or private training data are used anywhere in the repo.
