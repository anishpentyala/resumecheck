## Unreleased safety correction

- Fingerprint dense bfloat16 and float8 batch tensors without requiring a
  NumPy dtype or casting away their exact bits. Preserve existing fingerprints
  for NumPy-supported tensor dtypes.

- Record Lightning gradient accumulation as optimizer windows, fingerprinting
  every microbatch in order instead of producing duplicate optimizer-step IDs.
  Mean callback loss and final-batch gradient norm are window observables.
  Reject incomplete windows and unsupported optimizer-step jumps.

- Reject duplicate or reordered step IDs rather than sorting records into
  apparent parity; increasing nonconsecutive IDs remain supported.

- Encode container sizes and list/tuple type in versioned batch fingerprints.
  Old fingerprints could conflate nested batch structures; regenerate both
  trajectories for batch checking. Unversioned fingerprints fail closed.
- Compare infinities explicitly so a finite value cannot match infinity through
  an infinite relative-tolerance bound. Equal signed infinities still match.
- Reject comparisons with no enabled checks or an unobserved compared step.
- Cover nested batch differences through the JSON/CLI path.

# Changelog

## Unreleased

### Added

- GitHub Actions CI: a 3.9 stdlib-only floor leg (the package core never
  imports torch), torch-only core legs on 3.11/3.13, and one leg per
  adapter pinned to the validated upstream versions below (Lightning
  2.6.6; transformers 5.20.2 + accelerate 1.15.0; accelerate 1.15.0 +
  torchdata 0.11.0; CPU torch 2.14.1). Adapter legs run with a
  no-silent-skip guard (RESUMECHECK_REQUIRE_NO_SKIPS): a missing extra
  fails the leg instead of skipping the adapter module green.

- Characterized the workers>0 boundary and later-epoch mid-epoch
  resume behavior (follow-up to the num_workers axis): an epoch-boundary
  resume with num_workers>=1 matches only the first post-resume epoch,
  then duplicates its permutation (uncheckpointed shard counter
  restarts at 0); a mid-epoch resume diverges in ANY epoch, not just
  epoch 0 (sampler permutation not serialized; restored iterator
  re-draws a stale permutation). Verified across num_workers=1/2/4 and
  seeds 42/7 on both verified stacks; encoded as strict-xfail positive
  specs + negative controls on the same fingerprints. 4 new tests.


### Added

- `check_accelerate(..., num_workers=N)`: worker-count axis for the
  stateful comparator. With it comes a verified finding: on the two
  verified stacks, num_workers>=1 makes `prepare()` run the epoch
  through the original plain RandomSampler (global-RNG draw at prepare
  time), bypassing the seedable sampler entirely, and mid-epoch resume
  duplicates/drops samples (no restorable cursor). Strict-xfail positive
  spec + negative control gated on the exact source fingerprints of the
  verified stacks; no verified workaround; raw torchdata-only control
  restores exactly at all worker counts. 2 new tests.

### Added

- `check_accelerate_skip_path(...)`: comparator for the documented
  `skip_first_batches` resume path. On the verified-affected mechanism
  fingerprint, mid-epoch resume shifts every post-resume epoch back one
  permutation - encoded as a strict-xfail positive spec + negative
  control. Opt-in `skip_path_set_epoch_workaround=True` applies the
  verified `set_epoch(epoch)` realignment to post-resume epochs and
  verifies parity WITH the mitigation (fail-closed to mid-epoch resumes;
  caller-side mitigation, not an upstream fix). 4 new tests.

### Added

- `check_accelerate(..., boundary_set_epoch_workaround=True)`: opt-in
  comparator mode that applies the verified caller-side `set_epoch`
  realignment after `load_state` and verifies boundary-resume parity with
  the mitigation. Fail-closed: rejects `resume_at` values that are not an
  exact epoch boundary. Distinct from the raw documented-path check,
  which is unchanged; the mode verifies the mitigation, not an upstream
  fix. 2 new tests (parity holds with mitigation on 1.15.0 and master;
  mid-epoch misuse raises).

### Fixed

- Rebound the boundary-replay spec gate from the `1.16.0.dev0` dev version
  string (shared by every master snapshot, fixed or not) to the SHA-256
  fingerprint of the mechanism-carrying `DataLoaderShard.__iter__` +
  `set_epoch` source. Both verified snapshots (1.15.0 release, master
  @ 90b233c) have byte-identical mechanism source; any build whose
  mechanism source differs asserts parity normally.

### Changed

- Extended the Accelerate boundary-replay spec's verified-affected range
  after installing a standalone Python 3.11 and testing current master:
  the replay reproduces identically on master 1.16.0.dev0 @ 90b233c
  (torchdata 0.11.0), and the `set_epoch` workaround restores parity
  there too. 1.15.0 remains the latest PyPI release, so the bug is
  unpatched upstream. Gate, spec header and README updated.

### Fixed

- Corrected the Accelerate boundary-replay mechanism text (adapter spec
  header + README): the decisive cause is `DataLoaderShard`'s `iteration`
  counter driving `set_epoch` at every iteration start while being
  neither checkpointed nor advanced by the documented resume patterns -
  not primarily the sampler's exhaustion-side-effect. The
  `skip_first_batches` path is now explained by the same mechanism
  (previously listed as unresolved), and verified `set_epoch` caller
  workarounds are documented for both paths.

### Added

- **Accelerate adapter** (`resumecheck.accelerate.check_accelerate`):
  verifies save_state/load_state resume parity on the documented
  `use_stateful_dataloader` + `use_seedable_sampler` path. Mid-epoch
  resume passes on accelerate 1.15.0 + torchdata 0.11.0; a checkpoint
  saved exactly at an epoch boundary is caught replaying the completed
  epoch (pinned executable spec, verified-affected versions only).
  New `accel` extra: `accelerate>=1.1`, `torchdata>=0.8`, `torch>=2.0`.

- HF adapter: added an epoch-boundary resume parity test (n=8, batch 2 ->
  4 steps/epoch, resume_at=4, max_steps=8), the HF analog of the Lightning
  #22077 boundary case. It PASSES on transformers 5.20.2 + accelerate,
  and a fingerprint probe confirms epoch 2 batch order differs from
  epoch 1, so the test is sensitive to boundary mishandling rather than
  vacuously green. This is also the first local end-to-end run of the HF
  suite: it was previously skip-gated because transformers/accelerate were
  not installed in the verification environments.
## 0.2.1 (2026-10-30)

### Fixed

- **Batch-fingerprint one-sidedness now fails closed.** A batch
  fingerprint recorded in only one of the two trajectories previously
  skipped the batch comparison silently - the run could PASS on partial
  evidence with the strongest signal unchecked. It is now reported in
  `missing_fields` and fails the comparison, matching the documented
  fail-closed behavior for loss and grad_norm.
- **Batch fingerprints now encode structure, dtype, and shape.** The
  previous encoding hashed raw bytes joined by a separator, which could
  not distinguish a `b"\x01|\x02"` tensor from two `b"\x01"`/`b"\x02"`
  tensors, nor identical bytes under different dtypes or shapes.
  Fingerprints are now tagged and length-prefixed with dtype/shape
  headers, removing that collision class. Fingerprints change versus
  0.2.0; re-record baselines when upgrading (trajectories are
  run-scoped, so this only matters for saved trajectories compared
  across versions).

### Tests

- 5 new regression tests covering the one-sided-batch fail and the
  structure/dtype/shape fingerprint cases. 29 passed locally.

## Unreleased

### Tests

- The Lightning #22077 regression spec is now pinned to verified versions
  instead of asserting bug-presence unconditionally: a positive
  epoch-boundary parity test asserts compatibility normally on every
  version (strict-xfail only on 2.6.6, the verified-affected release), so
  a failure on any untested version is a genuine finding; the
  negative-control reproduction test runs only on 2.6.6. Verified:
  lightning 2.6.6 (30 passed, 1 skipped, 1 xfailed) and master
  2.7.0.dev0 + #22080 patch (30 passed, 2 skipped).
- README now states the validated upstream versions and the gate design.
