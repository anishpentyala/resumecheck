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

The recorder preserves integer-index numeric observations (loss, gradient norm
and extras) as exact integers instead of converting them to floats. This keeps
large counter differences visible to zero-tolerance comparisons and JSON
round trips. Other float-convertible observations keep their existing
conversion behavior; native Python booleans are rejected without appending a
partial step (NumPy or tensor boolean scalars keep their existing 1/0
conversion). This cannot recover precision already lost by the caller or an
adapter before recording.

Present batch or model-state fingerprints must be strings with the supported
`v2:` prefix when their check is enabled. Malformed types fail the comparison
instead of raising or being treated as absent. Caller-supplied versioned labels
remain accepted; replay does not authenticate their provenance or validate
that a fingerprint was produced by the recorder.

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

## Run a framework-independent checkpoint example

```bash
pip install . torch
python examples/manual_checkpoint.py --artifacts-dir manual-evidence
resumecheck replay manual-evidence
python examples/manual_checkpoint.py --omit optimizer
```

This executable CPU example uses a dropout MLP, AdamW and MultiStepLR on
synthetic data. Application code explicitly saves and restores model,
optimizer, scheduler, Torch CPU RNG, and a fixed shuffled permutation plus
batch cursor through an actual local checkpoint file. Nothing is downloaded.
The all-restored run compares six steps and exact model-state fingerprints.
`--omit model`, `optimizer`, `scheduler`, `rng` or `sampler` deliberately leaves
one component unrestored and exits 1 when the recorded trajectories diverge.
The scheduler control diverges after its first missed milestone, not necessarily
at the first resumed step. Evidence bundles can be replayed with saved settings.

This shows how to wire the existing manual recorder into a loop. It is not a
checkpoint manager or a production sampler implementation. The saved file is
created and loaded by this run; never load an untrusted checkpoint. CPU only,
one fixed permutation, one process, no workers, Python/NumPy/CUDA RNG or
arbitrary data pipeline restoration. A PASS covers these recorded observables,
not every optimizer or application state.

## Observe optimizer state even when losses match

A matching loss, model and learning rate do not imply matching optimizer
moments. For example, during a zero-learning-rate warmup, AdamW accumulates
moments without moving the weights. Restoring the weights but resetting the
optimizer can leave all three observables equal, then change the next update
when the learning rate becomes nonzero.

All adapters offer an opt-in exact optimizer-state check:

```python
report = check_hf(
    model_factory=TinyModel,
    train_dataset=ds,
    max_steps=100,
    resume_at=40,
    record_optimizer_state=True,
    record_model_state=True,       # independent optional model observation
    artifacts_dir="optimizer-evidence",
)
```

The same `record_optimizer_state=True` flag works with `check_lightning`,
`check_accelerate` and `check_accelerate_skip_path`. It fingerprints each
optimizer's complete `state_dict()` after each recorded optimizer update,
including per-parameter moments/counters and parameter-group settings.
Lightning records the ordered list of optimizer dictionaries; the other
adapters record their optimizer dictionary. Compare runs made with the same
adapter, optimizer configuration and stable parameter/group ordering.
Accelerate's experiments still use their existing SGD optimizer; this flag
adds observation, not an optimizer factory or a different training setup.

For a manual loop, set the observation on the step you just recorded:

```python
from resumecheck.recorder import fingerprint_tensor

rec.record(step, loss=loss.item(), batch=batch)
rec.trajectory.steps[-1].optimizer_fingerprint = fingerprint_tensor(
    optimizer.state_dict()
)
```

`compare_trajectories` compares recorded optimizer fingerprints by default,
even if numeric tolerances are loose. A mismatch reports `optimizer_state`.
One-sided evidence, malformed fingerprints, or a step missing both
fingerprints when either run otherwise records optimizer evidence fail verification.
Optimizer labels must have the recorder's `v2:` plus 16 lowercase hexadecimal
character format. This checks syntax, not authorship or integrity. When neither
run records optimizer evidence, the old comparison scope is unchanged.
`check_optimizer_state=False` explicitly excludes it from a direct comparison;
artifact export rejects reports that omit recorded optimizer evidence.

The flag is off by default: copying tensors to CPU and hashing optimizer
state adds time and memory cost proportional to state size. Unsupported
custom values raise instead of being silently ignored. Schema-4 evidence
bundles preserve the enabled optimizer comparison and replay rejects disabled
scope, missing evidence, or a downgrade to an older schema. Schemas 1-3 and
default trajectory JSON remain unchanged when optimizer evidence is absent.

This is an observation, not a checkpoint loader or a general resume guarantee.
It does not check parameter identity, hidden state outside `state_dict()`,
scheduler/RNG state, sharded/distributed optimizers, mixed precision, or CUDA.
The current proof covers deterministic single-process CPU runs. Fingerprints
are the existing truncated content hashes, not a collision-proof or adversarial
integrity scheme. No raw optimizer checkpoint is stored in an evidence bundle.

## Manual-loop named optimizer ownership (experimental)

Raw optimizer state dictionaries use positional IDs, not stable model names.
The existing `record_optimizer_state=True` flags still observe those raw
state dictionaries and require stable parameter/group ordering. They do not
change with this optional helper.

For a manual single-process CPU loop, opt into an observation that also binds
each optimizer parameter to its current live model name:

```python
from resumecheck.optimizer_identity import fingerprint_named_optimizer

rec.record(step, loss=loss.item(), batch=batch)
rec.trajectory.steps[-1].optimizer_fingerprint = fingerprint_named_optimizer(
    model, optimizer
)
```

Use this helper at the same observation point on both sides of a comparison.
Do not mix raw and named optimizer fingerprints. They use the same existing
`v2:` label syntax and schema-4 persistence but encode different observations.
The helper adds a distinct kind tag, optimizer type, ordered per-group live
names, shape, dtype and requires-grad status to the complete raw state dict.
It resolves names through actual Parameter object identity, not loaded
`param_names` metadata, which can be overwritten during a load.

The supported optimizer types are exact built-in SGD, Adam and AdamW. Each
optimizer parameter must have exactly one model name. Partial-model optimizers
are supported; tied-name aliases, repeated/foreign parameters, empty groups,
optimizer subclasses/wrappers, state-dict hooks, non-CPU parameters and
non-strided layouts are rejected. Torch is imported only when the helper is
called. This does not add a torch dependency to core imports. Each call copies
and hashes the complete optimizer state, including CPU tensor contents. Use
it for test-scale comparisons; cost grows with optimizer state size.

The positional relationship between serialized IDs and live groups relies on
the supported built-in torch optimizer implementation enumerating groups and
parameters in order. This is a cooperative API observation, not independent
proof against a modified torch runtime. Unknown hook-registry layouts fail
with an explicit unsupported-layout error.

A same-shape parameter permutation can leave raw optimizer dictionaries and
named model weights identical immediately after load while binding moments to
the wrong live parameters. The CPU tests reproduce this for SGD, Adam and
AdamW with one or two groups, and include a numerical SGD continuation where
raw optimizer observations still agree but live weights diverge. A named
optimizer observation distinguishes the ownership before that next update.
This is added observation scope, not a fix for the raw mode's documented
identity exclusion.

This is not a checkpoint loader, save-time identity manifest, preflight,
name-based remapping hook or identity authenticator. It cannot certify names,
missing checkpoint moments, shared-storage relationships between distinct
Parameters, hidden optimizer state, distributed/sharded/wrapped models, CUDA
behavior, scheduler/RNG state or resume correctness. A caller can still choose
misleading names. It uses existing truncated content hashes, not adversarial
attestation. Current execution evidence is CPU torch 2.14.1 only.

PyTorch documents positional optimizer loading [without additional identity
verification](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.md)
and that [loaded names do not affect loading](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.md).
Its [custom load hooks](https://docs.pytorch.org/docs/main/optim.html)
and DCP/FQN checkpoint helpers are existing name-based loading alternatives;
this helper does not replace them or claim to invent name-based checkpointing.

## How the crash is simulated

The interrupted run is configured for the **final** step count and then
stopped early (Lightning `trainer.should_stop`, HF `should_training_stop`),
not configured with a smaller `max_steps`. This matters: LR schedules and
other training-length-dependent setup see the true total, exactly as in a
real crash-and-resume. Resuming with a different total than the original run
changes the schedule by design — that is a different experiment, not a resume
bug.

## Integer experiment limits

All adapters require `max_steps` and `resume_at` to implement Python's integer
index protocol, with `0 < resume_at < max_steps`. Booleans, floats (including
integral-looking values), strings and nonfinite values are rejected before
training or artifact creation. Integer-index objects are normalized to built-in
integers. Fractional limits are never silently rounded into a different experiment.

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

### Temporary training outputs

HF and Lightning adapters remove their internally created training-output
folders after returning or raising. This includes default temporary checkpoint
outputs. Explicit `artifacts_dir` evidence is separate and remains available
for replay. Caller-selected `training_args_kwargs["output_dir"]` or
`trainer_kwargs["default_root_dir"]` directories are never removed by this
cleanup. A killed process can still leave temporary output behind.

### Retain evidence after an adapter check

For small deterministic CPU HF, Lightning or Accelerate checks, opt into exact model-state evidence:

```python
report = check_hf(model_factory, dataset, max_steps=12, resume_at=5,
                  record_model_state=True, artifacts_dir="./state-evidence")
# Lightning accepts the same opt-in flag:
report = check_lightning(module_factory, datamodule_factory, max_steps=12, resume_at=5,
                         record_model_state=True, artifacts_dir="./lightning-state-evidence")
```

This hashes `model.state_dict()` after each optimizer step, including parameters
and persistent buffers. Equal loss and batch fingerprints can hide a changed
parameter or unused buffer; this adds a separate `model_state` divergence check.
It compares dtype, shape and content bits, not tolerance-based numeric closeness.
`rtol` and `atol` still apply to the numeric observables, not these fingerprints.
It adds CPU copies and hashing proportional to model size at every step, so it
is opt-in and intended for small diagnostic models, not production throughput.
Unsupported custom state raises rather than being silently dropped. It does not
directly verify optimizer, scheduler, RNG, nonpersistent buffers, GPU/distributed
state, or arbitrary custom checkpoint behavior. Hashes are diagnostic evidence,
not collision-proof attestations. Default checks do not record model-state
fingerprints. The Accelerate adapters fingerprint the unwrapped model's full
`state_dict()`; Lightning samples after the final batch in each optimizer
window. This does not make stochastic training deterministic or restore RNG state
that the underlying checkpoint path does not preserve.

Model-state artifact bundles use schema 2 and require this check on replay.
Older resumecheck versions reject schema 2 rather than silently ignoring the
extra evidence. Default bundles remain schema 1 and omit absent model-state
fields, preserving their legacy JSON shape. The GPT-2 example accepts
`--model-state` to exercise this check without downloading a model.

Pass `artifacts_dir="./new-evidence"` to `check_hf`, `check_lightning`,
`check_accelerate` or `check_accelerate_skip_path`, or run:

```bash
python examples/accumulated_gpt2.py --accumulation 3 --artifacts-dir ./new-evidence
resumecheck replay ./new-evidence
```

The directory must not already exist, and its parent must exist. After a
completed comparison, including a divergent result, it contains both recorded
trajectories, `report.json`, and `manifest.json` with the comparison settings.
`resumecheck replay` uses the manifest's exact comparison settings, requires
the completion manifest and fixed files, and recomputes the result instead of
trusting a saved PASS. Exit codes are 0 for matching recorded observables, 1
for divergence or an unverified comparison, and 2 for malformed or incomplete
input. It is a replay of local evidence, not authentication of its origin.
The two-file `compare` command remains available for manual recordings. The manifest is written last; without it a
bundle is incomplete. Training exceptions do not produce a bundle. Existing
paths are never replaced, including a path created while training runs. Export
errors propagate instead of silently discarding evidence.

This is opt-in local output: there are no uploads and no retained model
checkpoints. Losses, data fingerprints and metadata may be sensitive. Review
artifacts before sharing. A replay checks only the evidence that was recorded. Model-state fingerprints
are present only when explicitly enabled; no bundle verifies all checkpoint state.

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

Dictionary key ordering keeps the historical string sort and uses typed key
encoding to break ties such as `1` versus `"1"`. Distinct typed keys with the
same display text no longer make a batch depend on insertion order. Existing
fingerprints without such ties stay unchanged. Regenerate both trajectories
when comparing batches with tied key display strings across this change;
older stored fingerprints are not rewritten. This does not normalize arbitrary
Python objects or promise collision-free fingerprints.

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
  norms, and opt-in HF/Lightning model-state fingerprints. Optimizer/RNG/sampler/worker
  state matter only insofar as they change
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

### Isolate Accelerate loader iterator RNG

Both own Accelerate adapters accept `isolate_dataloader_rng=True`. This gives
the DataLoader a separate seed-matched Torch generator, so building an iterator
after resume does not consume model/dropout CPU RNG. The default remains off:
this is a different, explicitly selected experiment, not a claim that arbitrary
existing training scripts resume correctly. CPU dropout-MLP checks cover two
seeds and multiple epochs, with exact model-state evidence enabled.

It does not repair sampler epoch counters. Stateful exact-boundary checks still
need the existing `boundary_set_epoch_workaround=True`, and skip-path checks
crossing epochs still need `skip_path_set_epoch_workaround=True`. Skip-path
boundary resumes, workers, CUDA/distributed and stochastic datasets are outside
this validation. No RNG workaround is silently applied to the default path.

### Run an offline Accelerate dropout example

```bash
pip install . accelerate torchdata torch
python examples/accelerate_dropout_resume.py --path stateful --artifacts-dir evidence
resumecheck replay evidence
python examples/accelerate_dropout_resume.py --path boundary
python examples/accelerate_dropout_resume.py --path skip
python examples/accelerate_dropout_resume.py --path skip --without-loader-isolation
python examples/accelerate_dropout_resume.py --path boundary --without-sampler-workaround
```

This input-dependent dropout MLP takes 19 SGD steps over fixed synthetic data,
checkpoints after step3 (mid-epoch) or step8 (boundary), and compares exact
post-step model-state fingerprints. Nothing is downloaded. The example
explicitly selects separate loader RNG and the documented caller sampler
workaround where needed. Removing loader isolation exits1 on model-state
divergence. Removing the boundary/skip sampler workaround exits1 on changed
training data at step9. Completed evidence can be replayed with saved settings.
This is CPU-only, single-process, no workers or stochastic dataset transforms,
and does not guarantee arbitrary application/optimizer state restoration.

### Accelerate model-state evidence

Both Accelerate adapters accept `record_model_state=True` to retain exact
post-optimizer-step fingerprints of the unwrapped model's full `state_dict()`,
including buffers. The default remains off and preserves legacy evidence.
Equal losses can hide parameter or buffer changes; enabling state evidence
checks these too and exports replayable schema-2 bundles. This copies and
hashes the whole model each step and is intended for small CPU checks, not
production training. It does not verify optimizer or arbitrary application
state, or expand the adapters' documented sampler/workaround support.

### Lazy conjugate and negative tensor views

Batch and model-state fingerprints use logical tensor values, including lazy
PyTorch conjugate and negative views. Fingerprinting materializes those bits
without changing the caller's tensor, then records the existing dtype/shape/content
encoding. A lazy view and its resolved value match; the unconjugated or unnegated
source generally does not. This is CPU-checked for dense tensors, not sparse,
quantized or device-specific view behavior.

### Sparse COO tensor observations

`fingerprint_tensor` also accepts PyTorch sparse COO tensors, including
uncoalesced and hybrid sparse/dense shapes, for batch, model and optimizer
observations. It hashes the stored representation: dtype, logical shape,
sparse/dense dimension counts, coalesced flag, ordered indices and value bytes.
It does not allocate the logical dense tensor or sum duplicate entries.

This is exact representation comparison, not mathematical sparse equivalence.
Different index order, duplicate decomposition, explicit zeros or coalesced
state may differ even when dense values would match. Inputs are not modified;
dense v2 fingerprints remain unchanged. CPU-checked; compressed sparse layouts
(CSR/CSC/BSR/BSC), quantized tensors and device-specific sparse behavior are
not added to the supported scope. Fingerprinting is not sparse-invariant
validation: callers remain responsible for constructing valid tensors.

The tests include a real CPU sparse `Embedding` + momentum-SGD checkpoint
round trip at zero learning rate. Restoring sparse momentum matches the full
run; omitting it leaves loss and model observations equal but produces an
optimizer-state divergence at the first resumed step. Saved comparison
artifacts replay that result. This is a small manual-loop diagnostic, not a
new checkpoint manager or a claim that every framework supports sparse state.

## Examples

`examples/tiny_gpt2_resume_parity.py` runs `check_hf` against a real
pretrained model - `sshleifer/tiny-gpt2` (public, ~2MB) - on the real HF
Trainer path, using SYNTHETIC fixed-seed token ids (no external text).
Requires `pip install torch transformers accelerate`; about 2 minutes on
CPU including the one-time model download. Reference run: PASS, 8 steps
identical within rtol=1e-5, atol=1e-7.

### Lightning dropout with application-owned CPU RNG checkpoints

```bash
python examples/lightning_dropout_resume.py --accumulation 3 --artifacts-dir ./dropout-evidence
python examples/lightning_dropout_resume.py --accumulation 3 --without-rng-restore
```

The first command is a small four-step CPU check with exact model-state evidence.
The second deliberately disables the example model's RNG restoration and exits1
on divergence. Both can retain replayable artifacts. Tests exercise accumulation1/2/3/4
with restoration on and off.

The model stores Torch CPU RNG state in its own checkpoint hook and restores it
before resumed training. Its DataLoader uses a separate seeded generator so
iterator creation does not consume model RNG. Constant synthetic input isolates
this RNG check from sampler-position restoration; it does not demonstrate shuffling
or real-data correctness. This is example application code, not a resumecheck
feature that restores RNG for arbitrary models. Python, NumPy, CUDA, distributed
and worker RNG are outside its scope. Requires Torch and Lightning; no downloads.

## Install & test

```bash
pip install -e ".[test]"   # torch + lightning included in test extra
pytest tests/
```

## Status

Early prototype. Core compare engine and recorder are stable; the Lightning
adapter covers single-process CPU training. Synthetic fixtures only — no real
datasets or private training data are used anywhere in the repo.

## Compare recorded learning rates and other numeric extras

The manual recorder already saves named numeric extras. Opt into comparing
specific fields when loss and batch evidence alone would miss their drift:

```python
rec.record(step, loss=loss, batch=batch, lr=optimizer.param_groups[0]["lr"])
report = compare_trajectories(full, resumed, check_extras=["lr"])
```

```bash
resumecheck compare full.json resumed.json --extra lr
# Repeat --extra NAME for more fields.
```

Defaults still ignore extras. Selected extras use the same `rtol` and `atol`
as losses and must be present and finite in both runs at every compared step.
Arbitrary-size Python/JSON integer counters remain exact and finite, even beyond
float range. Finite floats keep their represented values and use the same exact
tolerance comparison; no integer-to-float conversion is needed for this check.
A missing, null, boolean, string or nonfinite selected value fails verification
even when matching loss evidence is available. A numeric mismatch reports
`extras.NAME` and its first differing step. With partial comparison enabled,
only the common prefix is checked, as for the existing fields.

This compares the recorded scalar values, not complete optimizer or scheduler
state. Equal learning rates do not prove their internal state was restored.
Adapters do not automatically record these extras; use the manual recorder or
previously recorded trajectory JSON. Existing adapter artifact replay keeps
its original saved comparison scope and does not opt into extras.

### Retain a manual numeric-extra comparison

Manual recorders can save the selected scope for offline replay:

```python
from resumecheck.compare import compare_trajectories
from resumecheck.artifacts import save_comparison_artifacts

selected = ["lr"]
report = compare_trajectories(full, resumed, check_extras=selected)
save_comparison_artifacts(
    "new-evidence-directory", full, resumed, report,
    rtol=1e-5, atol=1e-7, check_extras=selected,
)
```

`resumecheck replay new-evidence-directory` recomputes those selected fields
with the saved tolerances, including missing or invalid evidence. The writer
checks that the supplied report matches that scope before creating a directory.
Selected names must be a list or tuple of unique, nonempty strings. Nonempty
selection uses schema 3; omitted or empty selection keeps existing schema 1/2
output unchanged. Older readers reject schema 3 rather than ignoring extras.
Automatic adapter capture is unchanged. This is local recorded evidence, not a
checkpoint, proof of provenance, tamper-proof storage, or complete optimizer or
scheduler-state verification. A person who edits both scope and evidence can
change what is checked. Review potentially sensitive recorded values before
sharing. Existing destinations are never replaced.

### HF optimizer-group learning-rate evidence

`check_hf(..., record_learning_rates=True)` records each optimizer parameter
group's learning rate at the optimizer-window end, after the Trainer's ordinary
scheduler step. Fields are `lr.group0`, `lr.group1`, etc., keyed by group position.
The selected fields are compared with `rtol`/`atol`; missing group evidence fails.
The option is off by default and can be combined with `record_model_state=True`.

```python
report = check_hf(model_factory, dataset, max_steps=6, resume_at=3,
                  record_learning_rates=True, artifacts_dir="./rate-evidence")
```

Schema-3 evidence retains the selected rate scope for `resumecheck replay`. This
can catch omitted scheduler restoration even when every recorded loss is constant.
It observes rates, not the complete scheduler or optimizer state: equal rates can
hide different scheduler internals, and group reordering is not identified by name.
Metric-driven schedulers may update later at evaluation, outside this sampling
point. No scheduler checkpoint or automatic recovery is added. The tested scope
is deterministic CPU Trainer runs, including gradient accumulation1and3.

### Machine-readable CLI reports

Use `resumecheck compare full.json resumed.json --json` or
`resumecheck replay evidence/ --json` to emit one strict JSON document to stdout.
The versioned envelope contains `schema_version: 1`, `status` (`pass`, `fail`,
or `error`), `passed`, `error`, and `report`. A completed `report` includes
`steps_compared`, `first_divergence`, `missing_fields`, and `summary` as well as
its own `passed` field. Nonfinite divergence values are strings (`NaN`,
`Infinity`, `-Infinity`), never invalid JSON numeric literals.

Exit codes remain 0 for verified matching evidence, 1 for a completed failed
comparison, and 2 for input or verification errors. On a handled error, `report`
is null and `passed` is false; a human-readable error also goes to stderr.
Argument-parsing errors still use argparse's stderr/exit-2 behavior and do not
promise JSON. Without `--json`, the existing text output is unchanged. Reports
verify only the recorded observables, not original training or provenance.

### Ambiguous evidence JSON

Trajectory loading and artifact replay reject repeated decoded JSON object
member names anywhere in the raw input, including equal repeated values and
escaped aliases such as `loss` and `lo\u0073s`. A last-value-wins parser could
otherwise discard divergent observables or conflicting replay settings.
The error omits member names and values, which may contain private metadata.
CLI comparisons/replays return input-error exit 2, including a structured
error when `--json` is selected. Distinct step objects may reuse field names
as usual; recorder-generated files are unchanged.

This detects ambiguous raw objects, not authenticity or tampering in general.
An ordinary dictionary already lost any duplicates discarded by an earlier
parser, so in-memory comparison cannot recover them. Replay still recomputes
the recorded observations rather than trusting a saved report verdict.

### Consistent artifact reports

Before writing a new evidence directory, the artifact writer recomputes the
full saved comparison scope and checks the supplied report's structured verdict,
step count, divergence and missing fields against it. A stale PASS from looser
tolerances, a disabled check or another trajectory cannot be saved alongside
settings that replay as FAIL. Tolerances must be finite nonnegative numbers,
not booleans or numeric strings. This applies to legacy schemas 1/2 as well as
selected-extras schema 3; mismatches leave no new directory.

The saved summary is regenerated for the stored settings, so caller wording
cannot contradict replay. The caller's report is not modified. Valid matching
reports keep the same schema and content. This is local consistency checking,
not provenance, authentication or tamper-proof storage. Replay still recomputes
from the saved trajectories and never trusts the stored PASS flag.

CI keeps explicit test-file groups so core, Lightning, HF and Accelerate legs
install only their intended extras. The Python-floor leg runs
`scripts/check_ci_test_coverage.py`, which fails when a `tests/test_*.py` module
has no explicit one-line pytest-command assignment in the workflow. Node-specific
assignments count at the file level. This guard does not prove every test or
conditional branch was selected or executed: per-leg logs and skip enforcement
still own that evidence, including intentional `-k` deselections.

Loss and gradient-norm comparisons reject booleans and nonnumeric recorded
values instead of treating them as numeric evidence. JSON integers retain
their exact value when compared with integers or floats, including values
beyond float precision or range. The same tolerance formula applies, with
exact rational arithmetic on integer-containing pairs. Existing matching-NaN
and same-signed infinity behavior remains unchanged; matching nonfinite
observables is not evidence that training was numerically healthy.

Both Accelerate adapter paths seed Python's global `random`, NumPy's global
RNG and Torch through `accelerate.utils.set_seed(seed)` at the start of each
phase, before constructing the model and loader. This lets seed-deterministic
factories use those global streams without
starting the uninterrupted and pre-crash runs from unrelated ambient state.
The resumed phase still relies on checkpoint restoration for the continuation
state. A corrupted restored RNG is not hidden by re-seeding after load.
See https://huggingface.co/docs/accelerate/main/package_reference/utilities
(`set_seed`, checked 2026-11-05) and
https://huggingface.co/docs/accelerate/concept_guides/performance.

This deliberately resets those process-global streams; it does not preserve the
caller's ambient RNG state. Independent `random.Random`/NumPy Generator objects,
external data sources and mutable dataset state are not reset by this helper.
Main-process random dataset transforms can still diverge at resume because
loader prefetch or batch skipping consumes extra random draws; phase seeding
is not a fix for that continuation problem. Worker/sampler limitations and the
separate loader-generator option above still apply. Torch-only factories keep the same seed and checked setup.

Batch/model-state fingerprint encoding is hashed incrementally: nested tensors
and arrays do not retain every encoded payload until the final digest. The
existing v2 encoding and fingerprints are unchanged. This reduces temporary
memory for batches or state dictionaries containing many separate buffers, but
one tensor/array still needs its CPU/contiguous/byte representation. Dictionary
keys with tied display strings retain encoded key bytes for deterministic
ordering. This is not a constant-memory or large-model throughput guarantee.

### Lightning learning-rate evidence

Pass `record_learning_rates=True` to `check_lightning` to sample all optimizer
parameter-group rates at each completed optimizer-window batch-end callback.
Fields such as `optimizer0.lr.group0` are compared with the selected `rtol` and
`atol`, and saved schema-3 evidence replays the same selected extra fields.
Nonfinite, negative, non-scalar or absent rates fail closed. A controlled
checkpoint scheduler-counter change can now diverge even with constant losses
and unchanged parameters; without this option those observables still pass.

This is rate evidence, not a scheduler-state fingerprint. Epoch-end or
metric-driven scheduler updates may happen after the callback, so the last
sample does not certify the final scheduler state. Unchanged rates may hide
other state changes. Manual optimization, multiple-optimizer sequencing,
distributed or custom callback orderings are not newly validated. The default
remains off and preserves the prior observable scope and artifact schema.

### Accelerate no-progress guard

Both Accelerate checks require a nonempty sized training dataset and a positive
integer-index `batch_size` (not bool, a rounded float, or a numeric string).
Two consecutive completed dataloader iterations with no optimizer step raise
`ValueError` rather than cycling indefinitely. One empty iteration is allowed
because a restored boundary iterator can be exhausted before the next epoch.
An exception leaves no completed artifact bundle.

This guards repeated empty iterations only, not a blocking iterator, slow
model, hanging worker or general time limit. A custom loader deliberately
empty for two whole iterations is rejected. Nonempty tiny datasets remain
supported, and the existing documented sampler/resume limitations are unchanged.

### Finite float tolerance boundaries

Loss and gradient-norm comparisons now evaluate the existing rule
`abs(a - b) <= atol + rtol * abs(b)` exactly for the represented finite
Python numbers. Rounding in an intermediate difference, product or sum must
not turn an out-of-tolerance pair into a PASS. This includes subnormal values
and finite pairs with very different magnitudes, not only overflow cases.

This can change a previous boundary PASS into a divergence, and a
previous boundary divergence into a PASS; both directions move boundary
cases onto the exact rule. Input floats
still carry their original binary rounding; this does not recover values
lost before recording. Existing matching-NaN and same-signed-infinity
behavior for loss and gradient norm is unchanged. Selected extras retain
their existing finite-value policy and exact tolerance comparisons. Exact
rational comparison costs more CPU than rounded float arithmetic.

### Unambiguous JSON output members

`Trajectory.save` and comparison-artifact trajectory writes validate their
serialized JSON before opening a trajectory output or creating a bundle
folder. Python JSON encoding can convert different dictionary keys to the
same member name, for example integer `1` and string `"1"`, or `None` and
`"null"`. Such collisions now raise a generic `ValueError` without echoing
the names or values, instead of publishing a trajectory the duplicate-safe
reader cannot load. This applies at every serialized object depth, including
metadata and extras.

Existing trajectory files are not truncated on this validation error, and
new bundle folders are not created. Noncolliding non-string keys retain their
existing JSON conversion; this does not promise lossless arbitrary Python
object serialization. Existing NaN/infinity value handling is unchanged.
Serialization validation adds a decode pass and some memory/CPU work before
writing. Later disk errors and process interruption keep their existing
behavior; this is not an atomic file-replacement or authenticity guarantee.


### Explicit comparison policy

The direct `compare_trajectories` API requires actual booleans for `check_loss`,
`check_batch`, `check_grad_norm`, `check_model_state` and `allow_partial`. Strings
such as `"false"`, numeric 0/1 and containers raise `TypeError` rather than
silently changing comparison scope through Python truthiness. Real `False`
still disables a selected check, and real `allow_partial=True` still scopes a
comparison to the common prefix with the step-count warning retained. This
validates API policy values; it does not expand which training state is recorded.
