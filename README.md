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

A Hugging Face `Trainer` adapter provides the same check, and an
Accelerate adapter covers the documented
`use_stateful_dataloader` + `use_seedable_sampler` resume path
(`resumecheck.accelerate.check_accelerate`).

```python
from resumecheck.hf import check_hf

report = check_hf(
    model_factory=TinyModel,     # fresh nn.Module returning {"loss": ...}
    train_dataset=ds,            # map-style dataset of dicts of tensors
    max_steps=100,
    resume_at=40,
    seed=42,
)
```

There is also a framework-agnostic core (`TrajectoryRecorder` +
`compare_trajectories`) you can wire into any training loop, and a CLI:

```bash
resumecheck compare uninterrupted.json resumed.json
```

## How the crash is simulated

The interrupted run is configured for the **final** step count and then
stopped early (Lightning `trainer.should_stop`, HF `should_training_stop`),
not configured with a smaller `max_steps`. This matters: LR schedules and
other training-length-dependent setup see the true total, exactly as in a
real crash-and-resume. Resuming with a different total than the original run
changes the schedule by design — that is a different experiment, not a resume
bug.

## What a PASS means — and what it does not

- Parity covers what is recorded: losses, batch content fingerprints, gradient
  norms. Optimizer/RNG/sampler/worker state matter only insofar as they change
  those observables.
- A PASS means *"these observables matched on this setup"*, not *"resume is
  correct for every configuration"*.
- Statefulness alone is not a guarantee: even with a stateful DataLoader,
  shuffling across an epoch boundary and sampler/RNG details can still
  diverge on resume. A divergence there can be framework behavior, not a
  defect - investigate before blaming your checkpoint code.
- Designed for **deterministic CPU verification**. Nondeterministic kernels
  (some CUDA ops), multi-worker dataloader timing, and distributed training
  can produce legitimate differences without any defect. Widen `rtol`/`atol`
  only with a stated reason.
- Some divergences are documented framework behavior, not bugs: PyTorch
  Lightning warns that resuming mid-epoch with a non-stateful DataLoader
  replays the epoch. resumecheck flags those too — deciding whether the
  divergence is acceptable is your call, and the test suite encodes known
  upstream cases as executable specs (see `tests/test_lightning_parity.py`).

Validated upstream versions (2026-10-30): the suite is verified against
Lightning 2.6.6 (latest release at filing; #22077 verified affected) and
master 2.7.0.dev0 with the #22080 patch (verified fixed). The regression
spec is pinned to those verified versions: bug-presence is asserted only
on 2.6.6, while the positive parity test asserts compatibility normally
everywhere else - a failure on any untested version is a genuine finding,
not an expected one. The HF adapter is verified against transformers
5.20.2 + accelerate: both the mid-epoch and the epoch-boundary parity
tests pass there, so the #22077-class boundary defect does not reproduce
on the HF Trainer path in this configuration.

Accelerate adapter validation (2026-10-30, CPU, torchdata 0.11.0):
mid-epoch resume parity PASSES on the documented stateful path. A
checkpoint saved exactly at an epoch boundary does NOT resume correctly:
the completed epoch is fully replayed. Verified affected: accelerate
1.15.0 (latest PyPI release) AND current master 1.16.0.dev0 @ 90b233c -
the bug is not fixed upstream.

Verified mechanism: `DataLoaderShard.__iter__` calls
`set_epoch(self.iteration)` at the start of every iteration, so the
shuffle permutation is driven by the shard's own iteration counter. That
counter is not part of the checkpointed state, and a crash at the epoch
boundary stops the loop before the counter increments - after
`load_state`, the next epoch reseeds the sampler with epoch 0 and replays
the completed epoch. The same counter explains the `skip_first_batches`
path: the skip wrapper consumes the sampler without advancing the shard's
counter, so every post-resume epoch is shifted back one permutation.

Verified workarounds (both directions probed, on 1.15.0 and on master):
on the stateful path, call
`train_dataloader.set_epoch(next_epoch_index)` after `load_state` when
resuming at an epoch boundary; on the `skip_first_batches` path, call
`train_dataloader.set_epoch(epoch)` for each epoch after the resume
epoch. The boundary case is encoded as a pinned executable spec
(strict-xfail positive + negative control). The spec gate binds to the
SHA-256 fingerprint of the mechanism's source (both verified snapshots
carry byte-identical `DataLoaderShard.__iter__`/`set_epoch`), not a
version string - an unknown future master asserts compatibility normally
instead of expected-failing. Not covered: torch-generator samplers, whose post-restore
permutation divergence is known upstream as pytorch/data#1561.

The adapter also ships the mitigation as an opt-in mode:
`check_accelerate(..., boundary_set_epoch_workaround=True)` applies the
verified `set_epoch` realignment after `load_state` and verifies parity
WITH the mitigation. It is fail-closed (rejects non-boundary `resume_at`)
and is a caller-side workaround, not an upstream fix - the raw
documented-path check above is unchanged.

The documented `skip_first_batches` path is covered by a separate check,
`check_accelerate_skip_path(...)`. On the verified-affected mechanism,
mid-epoch resume shifts every post-resume epoch back one permutation
(the skip wrapper never advances the shard's counter) - encoded as a
fingerprint-gated strict-xfail spec with a negative control. Its opt-in
mitigation mode (`skip_path_set_epoch_workaround=True`) applies the
verified `set_epoch(epoch)` realignment to each post-resume epoch and
verifies parity WITH it; fail-closed to mid-epoch resumes.

## Examples

`examples/tiny_gpt2_resume_parity.py` runs `check_hf` against a real
pretrained model - `sshleifer/tiny-gpt2` (public, ~2MB) - on the real HF
Trainer path, using SYNTHETIC fixed-seed token ids (no external text).
Requires `pip install torch transformers accelerate`; about 2 minutes on
CPU including the one-time model download. Reference run: PASS, 8 steps
identical within rtol=1e-5, atol=1e-7.

## Install & test

```bash
pip install -e ".[test]"   # torch + lightning included in test extra
pytest tests/
```

## Status

Early prototype. Core compare engine and recorder are stable; the Lightning
adapter covers single-process CPU training. Synthetic fixtures only — no real
datasets or private training data are used anywhere in the repo.
