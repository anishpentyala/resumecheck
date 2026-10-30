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
