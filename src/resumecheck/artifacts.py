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


def replay_comparison_artifacts(path: Union[str, Path]) -> CompareReport:
    """Recompute a completed evidence bundle using its saved settings.

    This checks recorded observables, not original training, provenance, or
    integrity against an attacker. The saved report's PASS is never trusted.
    Only the fixed trajectory names are read; arbitrary paths in the manifest
    are not followed. The final manifest is required to reject partial bundles.
    """
    import math

    from .compare import compare_trajectories

    directory = Path(path)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('artifact manifest must be an object')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('unsupported artifact schema version')
    if manifest.get('kind') != 'resumecheck.comparison':
        raise ValueError('unsupported artifact kind')
    expected_files = ['uninterrupted.json', 'resumed.json', 'report.json']
    if manifest.get('files') != expected_files:
        raise ValueError('artifact manifest must name the fixed comparison files in order')
    settings = manifest.get('comparison')
    required = {'rtol', 'atol', 'check_loss', 'check_batch', 'check_grad_norm', 'allow_partial'}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError('artifact comparison settings are missing or unsupported')
    for key in ('rtol', 'atol'):
        if type(settings[key]) not in (int, float):
            raise ValueError(f'artifact {key} must be a finite nonnegative number')
        try:
            valid = math.isfinite(settings[key]) and settings[key] >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f'artifact {key} must be a finite nonnegative number')
    for key in ('check_loss', 'check_batch', 'check_grad_norm', 'allow_partial'):
        if type(settings[key]) is not bool:
            raise ValueError(f'artifact {key} must be boolean')
    # Schema 1 is exported by the full adapter comparison, never a partial or
    # disabled-field check. Reject settings edits that silently weaken scope.
    if not all(settings[key] for key in ('check_loss', 'check_batch', 'check_grad_norm')) or settings['allow_partial']:
        raise ValueError('schema 1 requires full comparisons with all recorded fields enabled')
    # Require the saved report to be present and structurally readable, but
    # never use its passed flag to decide this replay's result.
    saved = json.loads((directory / 'report.json').read_text())
    if not isinstance(saved, dict) or type(saved.get('passed')) is not bool:
        raise ValueError('artifact report is missing or malformed')
    return compare_trajectories(Trajectory.load(directory / 'uninterrupted.json'),
                                Trajectory.load(directory / 'resumed.json'), **settings)
