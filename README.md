# resumecheck

![tests](https://github.com/anishpentyala/resumecheck/actions/workflows/ci.yml/badge.svg)

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

Step indices must use Python's integer-index protocol; booleans, floats,
strings and nonfinite values do not verify a training trajectory. This applies
to directly constructed records and loaded JSON as well as recorded runs.
Integer-like indices are normalized for comparison without changing caller
records. Indices must still be strictly increasing. Invalid records anywhere
in either run fail even when partial-prefix comparison is requested.

Raw-byte providers without both `dtype` and `shape` metadata (including
`array.array` and `memoryview`) are rejected rather than silently discarding
element types or shape. Convert them explicitly to supported arrays or native
containers before recording.

NumPy masked arrays are rejected because their raw-byte representation discards
masks and substitutes fill values, which can hide a change in batch content.
For explicit comparison, convert them to a native container holding the data,
mask and any relevant fill value before recording.

NumPy arrays containing Python objects are rejected, including structured dtypes
with object fields: their raw bytes hold object addresses rather than object
content. Opaque objects are also rejected: `repr()` is display text, not proof
of batch content. Convert those batches to supported tensors, non-object arrays,
native containers, strings/bytes or built-in scalars before recording. Built-in
scalar types are None, bool, int, float and complex; scalar subclasses with
additional state are not supported. Numeric structured arrays and fixed-width
strings remain supported.

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

## Lightning phase completeness

The Lightning adapter requires every phase to record its exact requested
optimizer-step range: uninterrupted `1..max_steps`, pre-crash `1..resume_at`,
and post-resume `resume_at+1..max_steps`. If an epoch limit, `min_steps`,
stopping condition or incomplete recording prevents this, it raises `ValueError`.
Equal shorter trajectories do not verify the requested checkpoint-and-resume
experiment. Choose training limits and enough data to let all three phases run.

## Hugging Face gradient accumulation

The HF Trainer adapter records one optimizer window when
`gradient_accumulation_steps` is greater than one. Its batch fingerprint includes
all consumed training microbatches in order, including short final windows. Loss
is the mean of the raw model losses in that window, not Trainer's normalized
backward loss. Evaluation losses are excluded. This compares window-level loss;
it does not compare every individual microbatch loss.

Each uninterrupted, pre-crash and resumed phase must record the exact requested
optimizer-step range. Missing phases and incomplete windows raise rather than
certify a shorter run. A PASS remains scoped to the recorded observables and the
tested deterministic CPU setup, not general resume correctness.

### Run an offline attention-model example

```bash
pip install -e '.[hf]'
python examples/accumulated_gpt2.py --accumulation 3
```

This uses a small, randomly initialized GPT-2 architecture with dropout and fixed
synthetic token IDs. No model weights, tokenizer or dataset are downloaded.
The example runs four optimizer steps, checkpoints after two, and exits nonzero
if the recorded resumed trajectory does not match. It is a CPU smoke example,
not pretrained-model training, a quality benchmark or a guarantee for other
configurations. Accumulation2/3 are exercised in the HF CI job.

## Lightning gradient accumulation

The Lightning adapter records one optimizer-step window when
`accumulate_grad_batches` is greater than one. It fingerprints every
microbatch in order, compares the mean of callback-provided microbatch losses,
and samples gradient norm at the final batch of the window. A PASS covers
those window-level observables, not every individual microbatch loss.
Intermediate microbatches are not dropped. Incomplete windows and unsupported
multi-step jumps raise instead of returning a misleading comparison.

Earlier builds recorded repeated optimizer-step IDs for accumulated batches.
The strict step-order check rejected those records even when training resumed
correctly. Regenerate affected trajectories with this correction.

## Low-precision batch tensors

Batch fingerprints support CPU `bfloat16` and float8 tensors even when NumPy
cannot represent their dtype. The fallback hashes exact storage bits plus
dtype and logical shape, with a compact clone so sliced views do not include
unrelated storage. It does not cast values to float32. Existing fingerprints
for NumPy-supported tensor dtypes are unchanged. Sparse, quantized and other
unsupported tensor representations are not made supported by this fallback.

## Fingerprint compatibility

New batch fingerprints use a `v2:` prefix and encode container sizes and
list/tuple type. Older unversioned fingerprints omitted container boundaries:
for example, `[[1], 2]` and `[[1, 2]]` could share the same fingerprint.
Comparisons with batch checking enabled now fail closed on those old records.
Record both runs again to check batches. `--no-batch` explicitly limits the CLI
comparison to the other recorded observables; it cannot verify input data.
A comparison also fails when no checks are enabled or any compared step has
no shared enabled observable. Partial-prefix mode does not waive these checks. Step IDs must be strictly
increasing in recorded order; duplicates and reordered records fail instead
of being sorted into apparent parity. Nonconsecutive increasing IDs are allowed.

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

CI runs these exact pins on every push and PR (`.github/workflows/ci.yml`),
with a no-silent-skip guard so a missing extra fails the leg instead of
skipping the adapter module.

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

num_workers >= 1 (verified 2026-10-30 on accelerate 1.15.0 and master
1.16.0.dev0 @ 90b233c, torchdata 0.11.0, single process CPU): resume
DIVERGES in every configuration characterized so far. Traced mechanism:
`prepare()` eagerly creates the loader iterator (the state snapshot in
`DataLoaderAdapter.__init__`), and on the workers>0 path that snapshot
iterator runs the user's ORIGINAL plain `RandomSampler`
(`shuffle=True` default, `generator=None`) drawn from the process
global RNG. Epoch 0 iterates through it - the `SeedableRandomSampler`
accelerate swaps in afterwards never drives epoch 0, so `data_seed` is
inert for epoch 0 (verified: data_seed 7 vs 8 give identical epoch-0
permutations; changing the global seed changes them). Later epochs of
an uninterrupted run create fresh iterators through the seedable
sampler with an aligned epoch counter. Raw torchdata
`StatefulDataLoader` without accelerate restores exactly at
workers=0/1/2/4, so the divergences below are introduced by
accelerate's prepared-loader layer. This shares the eager
prepare()-time iterator root with the open multi-process issue
huggingface/accelerate#4204; the single-process seedable-bypass and
resume divergences are beyond that issue's scope. No workaround is
verified - the `set_epoch` mitigations do not apply to this path.

Three verified consequences (each across num_workers=1/2/4 and seeds
42/7):

1. Mid-epoch resume duplicates and drops samples - in epoch 0 AND in
   later epochs. The saved state does not serialize the sampler
   permutation (`_sampler_iter_state` is null; it carries batch counts
   and a worker base seed), and the restored iterator re-draws a stale
   permutation rather than the interrupted epoch's.
2. An epoch-boundary resume matches only the first post-resume epoch
   (the frozen counter regimes coincide, so two-epoch horizons pass
   coincidentally); the SECOND post-resume epoch then duplicates the
   first's permutation, because the shard's uncheckpointed iteration
   counter restarts at 0 and post-resume epoch N draws epoch N-1's
   seed.
3. Epoch 0's permutation depends on the process global RNG at
   prepare() time, not on `data_seed`.

Encoded as specs gated on the exact source fingerprints of the two
verified stacks (a verified-stack marker, not a claim that every build
sharing the fingerprint is affected); any other build asserts
compatibility normally.

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
