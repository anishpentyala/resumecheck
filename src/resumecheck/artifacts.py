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
from typing import Union, Optional, Sequence

from .compare import CompareReport
from .recorder import Trajectory, step_to_dict


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
    check_extras: Optional[Sequence[str]] = None,
) -> None:
    """Write a replayable comparison without replacing existing paths.

    Serialize before creating the directory. mkdir is the exclusive claim, so
    a destination created while training was running is never overwritten.
    On ordinary write errors remove our incomplete directory and re-raise;
    interruption or process death can still leave partial artifacts. A bundle
    is complete only when its manifest.json exists.

    Nonempty ``check_extras`` writes schema 3 and verifies that ``report``
    matches the selected scope with these tolerances. Old schemas/default
    output remain unchanged. This stores observations, not scheduler state.
    """
    extra_names = _selected_extras(check_extras)
    if extra_names:
        from .compare import compare_trajectories

        expected = compare_trajectories(uninterrupted, resumed, rtol=rtol, atol=atol,
                                        check_extras=extra_names)
        if asdict(expected) != asdict(report):
            raise ValueError('artifact report does not match the selected comparison scope')
    destination = validate_artifacts_dir(path)
    trajectories = {
        'uninterrupted.json': uninterrupted,
        'resumed.json': resumed,
    }
    payloads = {
        name: json.dumps({'name': trajectory.name, 'meta': trajectory.meta,
                          'steps': [step_to_dict(record) for record in trajectory.steps]}, indent=2)
        for name, trajectory in trajectories.items()
    }
    payloads['report.json'] = json.dumps(asdict(report), indent=2)
    has_model_state = any(record.model_fingerprint is not None
                          for trajectory in trajectories.values() for record in trajectory.steps)
    comparison = {'rtol': rtol, 'atol': atol, 'check_loss': True,
                  'check_batch': True, 'check_grad_norm': True, 'allow_partial': False}
    if has_model_state:
        comparison['check_model_state'] = True
    if extra_names:
        comparison['check_extras'] = list(extra_names)
        comparison['check_model_state'] = has_model_state
    manifest = json.dumps({
        'schema_version': 3 if extra_names else (2 if has_model_state else 1),
        'kind': 'resumecheck.comparison',
        'files': list(payloads),
        'comparison': comparison,
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
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] not in (1, 2, 3):
        raise ValueError('unsupported artifact schema version')
    if manifest.get('kind') != 'resumecheck.comparison':
        raise ValueError('unsupported artifact kind')
    expected_files = ['uninterrupted.json', 'resumed.json', 'report.json']
    if manifest.get('files') != expected_files:
        raise ValueError('artifact manifest must name the fixed comparison files in order')
    settings = manifest.get('comparison')
    required = {'rtol', 'atol', 'check_loss', 'check_batch', 'check_grad_norm', 'allow_partial'}
    if manifest['schema_version'] in (2, 3):
        required.add('check_model_state')
    if manifest['schema_version'] == 3:
        required.add('check_extras')
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError('artifact comparison settings are missing or unsupported')
    if manifest['schema_version'] == 3:
        try:
            names = _selected_extras(settings['check_extras'])
        except (ValueError, TypeError) as error:
            raise ValueError('artifact check_extras must contain unique nonempty field names') from error
        if not names or not isinstance(settings['check_extras'], list):
            raise ValueError('schema 3 requires a nonempty selected extras list')
        if type(settings['check_model_state']) is not bool:
            raise ValueError('artifact check_model_state must be boolean')
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
    if manifest['schema_version'] == 2 and settings['check_model_state'] is not True:
        raise ValueError('schema 2 requires model-state comparisons')
    # Schema 1 is exported by the full adapter comparison, never a partial or
    # disabled-field check. Reject settings edits that silently weaken scope.
    if not all(settings[key] for key in ('check_loss', 'check_batch', 'check_grad_norm')) or settings['allow_partial']:
        raise ValueError('artifact requires full comparisons with all recorded fields enabled')
    # Require the saved report to be present and structurally readable, but
    # never use its passed flag to decide this replay's result.
    saved = json.loads((directory / 'report.json').read_text())
    if not isinstance(saved, dict) or type(saved.get('passed')) is not bool:
        raise ValueError('artifact report is missing or malformed')
    full = Trajectory.load(directory / 'uninterrupted.json')
    resumed = Trajectory.load(directory / 'resumed.json')
    state_recorded = any(record.model_fingerprint is not None for trajectory in (full, resumed)
                         for record in trajectory.steps)
    if manifest['schema_version'] == 1 and state_recorded:
        raise ValueError('model-state evidence requires schema 2')
    if manifest['schema_version'] == 2 and not state_recorded:
        raise ValueError('schema 2 has no recorded model-state evidence')
    if manifest['schema_version'] == 3 and settings['check_model_state'] != state_recorded:
        raise ValueError('schema 3 model-state setting must match recorded evidence')
    return compare_trajectories(full, resumed, **settings)


def _selected_extras(names: Optional[Sequence[str]]) -> tuple[str, ...]:
    """An explicit stable scope, never a string, mapping or exhausted iterator."""
    if names is None:
        return ()
    if not isinstance(names, (list, tuple)):
        raise TypeError('check_extras must be a list or tuple of unique nonempty field names')
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('check_extras names must be nonempty strings')
    if len(set(names)) != len(names):
        raise ValueError('check_extras names must be unique')
    return tuple(names)
