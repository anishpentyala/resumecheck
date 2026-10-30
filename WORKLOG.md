# Worklog

## 2026-10-30 ~9:30 AM PT - accelerate adapter: two resume paths, verified divergences, mitigations

- F2 (skip path) mechanism resolved and unified with F1: DataLoaderShard.__iter__ calls set_epoch(self.iteration) at every iteration start, so the shuffle permutation is driven by the shard's own iteration counter. That counter is not part of the checkpointed state and neither documented resume pattern advances it: stateful boundary crash-breaks before the counter increments (full replay of the completed epoch); the skip_first_batches wrapper bypasses the shard entirely (every post-resume epoch shifts back one permutation). Mid-epoch stateful self-corrects.
- Verified caller workarounds, both directions probed: set_epoch(next_epoch_index) after load_state on the stateful path; set_epoch(epoch) per post-resume epoch on the skip path. set_epoch BEFORE the skip epoch breaks the resume epoch itself (probed).
- Python 3.11.18 standalone installed (/tmp/py311, python-build-standalone 20261026) -> /tmp/acc-master-venv. Master comparison unblocked: bug reproduces identically on accelerate master 1.16.0.dev0 @ 90b233c (torchdata 0.11.0); 1.15.0 is still the latest PyPI release - unpatched upstream. Workaround verified on master too.
- Gate review feedback applied: verified-affected gate binds to the SHA-256 fingerprint of DataLoaderShard.__iter__ + set_epoch source (byte-identical on both verified snapshots), NOT the dev version string; unknown builds assert compatibility normally.
- PRs merged: #10 (adapter + boundary spec), #11 (mechanism correction + skip-path resolution), #12 (master verification), #13 (fingerprint gate + stateful workaround mode), #14 (skip-path comparator + workaround mode).
- Suites on main: 39 passed + 3 xfailed (pinned 1.15.0), 39 passed + 1 skipped + 2 xfailed (lightning master), 33 passed + 2 skipped + 2 xfailed (accelerate master).
- Env hygiene: /tmp/rcv and /tmp/plv had stale non-editable resumecheck installs shadowing the worktree (caught when new kwargs went missing); both reinstalled editable.
- HELD: upstream issue candidate for the unified finding (no public issues/comments without the user's scope answer).

## 2026-10-30 ~10:45 AM PT - F4: workers>0 seedable bypass + resume divergence

- Demand research: F2 skip path already reported upstream (huggingface/accelerate#4330, open, stale-marked); FIVE fix PRs in the resume-order family closed unmerged Sept-Oct; SeverinVisionary asked maintainer for exactly the regression coverage resumecheck ships. F1 (stateful boundary) remains unreported.
- F4 found via #4204 qualification: single-process, seedable+stateful, num_workers>=1 -> mid-epoch resume duplicates/drops samples. Robust across workers=1/2/4, n, seeds, resume points, 1.15.0 + master.
- Mechanism traced with PID-aware logging + identity checks (first "worker-side sampler" hypothesis withdrawn after main's challenge): prepare()'s eager state_dict creates the iterator, which runs the ORIGINAL plain RandomSampler (generator=None -> global RNG); epoch loop reuses it; SeedableRandomSampler never iterates; saved state has no cursor (_num_yielded=None). Raw torchdata-only restores exactly at all worker counts -> accelerate's prepared-loader layer introduces the divergence.
- data_seed ignored with workers>0 (7 vs 8 identical); global seed changes permutation. Boundary-with-workers passed 4 configs - uncharacterized, no claim. No verified workaround.
- Spec: dual source-fingerprint gate (1.15.0: b38c0ad1..., master@90b233c: a77f1715...) + torchdata 0.11.0 pin; positive strict-xfail + negative control; num_workers axis added to check_accelerate.

## 2026-10-30 (later): workers>0 boundary + later-epoch mid-epoch characterization

Extended the F4 lane after PR #15. Probe-driven characterization
(/home/sandbox/probes/boundary_workers_probe.py and
boundary_workers_matrix.py; PID-aware sampler instrumentation plus a
workers {1,2,4} x seeds {42,7} matrix with full nested state dumps):

- The seedable bypass is epoch-0-only in an uninterrupted run: epoch 0
  iterates through prepare()'s eager snapshot iterator (original plain
  RandomSampler, global RNG); epochs 1+ create fresh iterators through
  SeedableRandomSampler with an aligned counter.
- Boundary resume with workers>0 passes two-epoch horizons
  COINCIDENTALLY (frozen counter regimes coincide); the second
  post-resume epoch duplicates the first's permutation (shard counter
  restarts at 0 post-resume). Verified across the full matrix.
- Mid-epoch resume diverges in ANY epoch (epoch 0 and epoch 1
  verified): saved state carries batch counts + worker base seed but
  _sampler_iter_state is null; restored iterator re-draws a stale
  permutation.
- Encoded as 4 spec tests (2 strict-xfail positive + 2 negative
  controls) on the existing dual-fingerprint workers gate; suites:
  42 passed + 6 xfailed (rcv), 42 passed + 1 skipped + 5 xfailed (plv),
  36 passed + 2 skipped + 5 xfailed (acc-master). Published as PR #16.

CI added (2026-10-30): main flagged that a public tool whose promise is
"verified against upstream versions" had no continuous verification.
Design: floor leg (3.9, stdlib-only import - core never imports torch;
pinned to ubuntu-24.04 because actions/python-versions has no 3.9 build
for ubuntu-26.04), core legs (torch+numpy fixtures, py 3.11/3.13 - torch
requires >=3.11), one adapter leg per pinned validated version set from
README (lightning 2.6.6 / transformers 5.20.2+accelerate 1.15.0 /
accelerate 1.15.0+torchdata 0.11.0, CPU torch 2.14.1). Adapter modules
guard imports with pytest.importorskip, which would skip green in CI;
tests/conftest.py adds an opt-in RESUMECHECK_REQUIRE_NO_SKIPS session
hook that converts any skip into exit 1. Verified locally before
publishing: pinned env 42 passed+6 xfailed exit 0; missing-lightning env
with the guard exit 1 with a clear message (without it: exit 5, no tests
collected). Strict xfails on verified-affected versions double as
upstream-fix alarms on the pinned legs. Env-label correction (main):
rcv = lightning 2.6.6 + accelerate 1.15.0 + torch 2.14.1+cpu; plv =
lightning master 2.7.0.dev0 + accelerate 1.15.0; acc-master-venv =
accelerate master 1.16.0.dev0@90b233c. (Earlier I wrote "rcv
(pytorch-lightning 1.15.0)" - conflated accelerate's version onto
pytorch-lightning; correct labels used from here.)
