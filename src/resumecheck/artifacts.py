"""Opt-in local evidence for replaying a completed comparison.

Artifacts contain recorded observables, not model checkpoints. They may expose
training data fingerprints, losses, and caller-supplied metadata. Review them
before sharing. No upload or network operation is performed.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Union

from .compare import CompareReport
from .recorder import Trajectory


def validate_artifacts_dir(path: Union[str, Path]) -> Path:
    """Require a new directory below an existing parent, without modifying it."""
    destination = Path(path).absolute()
    # lexists semantics also reject a dangling symlink.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"artifact directory already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"artifact parent directory does not exist: {destination.parent}")
    return destination


def save_comparison_artifacts(
    path: Union[str, Path], uninterrupted: Trajectory, resumed: Trajectory,
    report: CompareReport, *, rtol: float, atol: float,
) -> None:
    """Write a replayable comparison without replacing existing paths.

    Serialize before creating the directory. mkdir is the exclusive claim, so
    a destination created while training was running is never overwritten.
    On ordinary write errors remove our incomplete directory and re-raise;
    interruption or process death can still leave partial artifacts. A bundle
    is complete only when its manifest.json exists.
    """
    destination = validate_artifacts_dir(path)
    trajectories = {
        'uninterrupted.json': uninterrupted,
        'resumed.json': resumed,
    }
    payloads = {
        name: json.dumps({'name': trajectory.name, 'meta': trajectory.meta,
                          'steps': [asdict(record) for record in trajectory.steps]}, indent=2)
        for name, trajectory in trajectories.items()
    }
    payloads['report.json'] = json.dumps(asdict(report), indent=2)
    manifest = json.dumps({
        'schema_version': 1,
        'kind': 'resumecheck.comparison',
        'files': list(payloads),
        'comparison': {'rtol': rtol, 'atol': atol, 'check_loss': True,
                       'check_batch': True, 'check_grad_norm': True, 'allow_partial': False},
        'scope': 'Recorded observables only; not a model checkpoint or general resume guarantee.',
    }, indent=2)
    destination.mkdir(mode=0o700)  # Atomic non-overwrite check, including symlinks.
    try:
        for name, contents in payloads.items():
            (destination / name).write_text(contents)
        # Written last: an interrupted bundle must not look complete.
        (destination / 'manifest.json').write_text(manifest)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
