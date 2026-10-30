# Changelog

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
