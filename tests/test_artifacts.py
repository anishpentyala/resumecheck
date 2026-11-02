"""Replayable local evidence, including failures and non-overwrite behavior."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from resumecheck.artifacts import save_comparison_artifacts, validate_artifacts_dir
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import Trajectory, TrajectoryRecorder


def trajectories(diverge=False):
    left, right = TrajectoryRecorder('full', seed=7), TrajectoryRecorder('resumed', seed=7)
    for step in range(1, 4):
        left.record(step, loss=step / 10, batch=[step])
        right.record(step, loss=step / 10 + (1 if diverge and step == 2 else 0), batch=[step])
    return left.trajectory, right.trajectory


@pytest.mark.parametrize('diverge', [False, True])
def test_bundle_replays_exact_report(tmp_path, diverge):
    left, right = trajectories(diverge)
    report = compare_trajectories(left, right, rtol=.003, atol=.002)
    directory = tmp_path / 'evidence'
    save_comparison_artifacts(directory, left, right, report, rtol=.003, atol=.002)
    assert sorted(p.name for p in directory.iterdir()) == [
        'manifest.json', 'report.json', 'resumed.json', 'uninterrupted.json']
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['schema_version'] == 1
    replay = compare_trajectories(Trajectory.load(directory / 'uninterrupted.json'),
                                 Trajectory.load(directory / 'resumed.json'), **manifest['comparison'])
    assert asdict(replay) == asdict(report) == json.loads((directory / 'report.json').read_text())
    assert replay.passed is not diverge
    assert json.loads((directory / 'uninterrupted.json').read_text())['meta'] == {'seed': 7}


@pytest.mark.parametrize('kind', ['file', 'directory', 'symlink', 'dangling'])
def test_never_replaces_existing_path(tmp_path, kind):
    directory = tmp_path / 'evidence'
    target = tmp_path / 'target'
    target.write_text('private existing data')
    if kind == 'file':
        directory.write_text('existing file')
    elif kind == 'directory':
        directory.mkdir()
        (directory / 'keep').write_text('existing directory')
    else:
        directory.symlink_to(target if kind == 'symlink' else tmp_path / 'absent')
    left, right = trajectories()
    with pytest.raises(FileExistsError):
        save_comparison_artifacts(directory, left, right, compare_trajectories(left, right), rtol=0, atol=0)
    assert target.read_text() == 'private existing data'
    if kind == 'directory':
        assert [p.name for p in directory.iterdir()] == ['keep']
    elif kind == 'file':
        assert directory.read_text() == 'existing file'
    else:
        assert directory.is_symlink()


def test_missing_parent_does_not_create_anything(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_artifacts_dir(tmp_path / 'missing' / 'evidence')
    assert not (tmp_path / 'missing').exists()


def test_serialization_error_does_not_leave_bundle(tmp_path):
    left, right = trajectories()
    left.meta['opaque'] = object()
    directory = tmp_path / 'evidence'
    with pytest.raises(TypeError):
        save_comparison_artifacts(directory, left, right, compare_trajectories(left, right), rtol=0, atol=0)
    assert not directory.exists()


@pytest.mark.parametrize('failed_file', ['resumed.json', 'report.json', 'manifest.json'])
def test_write_error_removes_our_partial_bundle(tmp_path, monkeypatch, failed_file):
    left, right = trajectories()
    original = Path.write_text
    def write(path, *args, **kwargs):
        if path.name == failed_file:
            raise OSError('simulated disk failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', write)
    directory = tmp_path / 'evidence'
    with pytest.raises(OSError, match='disk failure'):
        save_comparison_artifacts(directory, left, right, compare_trajectories(left, right), rtol=0, atol=0)
    assert not directory.exists()


def test_race_after_validation_never_overwrites(tmp_path, monkeypatch):
    directory = tmp_path / 'evidence'
    left, right = trajectories()
    original = Path.mkdir
    def mkdir(path, *args, **kwargs):
        if path == directory:
            original(path)
            (path / 'keep').write_text('other owner')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', mkdir)
    with pytest.raises(FileExistsError):
        save_comparison_artifacts(directory, left, right, compare_trajectories(left, right), rtol=0, atol=0)
    assert [p.name for p in directory.iterdir()] == ['keep']


def test_relative_destination_is_anchored_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert validate_artifacts_dir('evidence') == tmp_path / 'evidence'
    assert not (tmp_path / 'evidence').exists()


@pytest.mark.parametrize('diverge', [False, True])
def test_cli_can_replay_bundle(tmp_path, capsys, diverge):
    from resumecheck.cli import main
    left, right = trajectories(diverge)
    directory = tmp_path / 'evidence'
    report = compare_trajectories(left, right, rtol=.003, atol=.002)
    save_comparison_artifacts(directory, left, right, report, rtol=.003, atol=.002)
    assert main(['compare', str(directory / 'uninterrupted.json'), str(directory / 'resumed.json'),
                 '--rtol', '.003', '--atol', '.002']) == (1 if diverge else 0)
    assert report.summary in capsys.readouterr().out


def test_completion_manifest_is_written_last(tmp_path, monkeypatch):
    left, right = trajectories()
    directory = tmp_path / 'evidence'
    original = Path.write_text
    observed = []
    def write(path, *args, **kwargs):
        assert not (directory / 'manifest.json').exists()
        observed.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', write)
    save_comparison_artifacts(directory, left, right, compare_trajectories(left, right), rtol=0, atol=0)
    assert observed == ['uninterrupted.json', 'resumed.json', 'report.json', 'manifest.json']


def test_directory_is_private_on_posix(tmp_path):
    import os
    import stat
    if os.name != 'posix':
        pytest.skip('POSIX directory modes only')
    left, right = trajectories()
    directory = tmp_path / 'evidence'
    save_comparison_artifacts(directory, left, right, compare_trajectories(left, right), rtol=0, atol=0)
    assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0
