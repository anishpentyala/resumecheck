# Changelog

## Unreleased

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
