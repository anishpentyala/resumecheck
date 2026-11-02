"""Replay uses exact saved settings, requires completion and recomputes verdict."""
import json
from dataclasses import asdict

import pytest

from resumecheck.artifacts import replay_comparison_artifacts, save_comparison_artifacts
from resumecheck.compare import compare_trajectories
from resumecheck.cli import main
from resumecheck.recorder import TrajectoryRecorder


@pytest.fixture
def bundle(tmp_path):
    def make(delta=0, rtol=.01, atol=.001):
        a, b = TrajectoryRecorder('full'), TrajectoryRecorder('resumed')
        for step in range(1, 4):
            a.record(step, loss=1., batch=[step])
            b.record(step, loss=1. + (delta if step == 2 else 0), batch=[step])
        report = compare_trajectories(a.trajectory, b.trajectory, rtol=rtol, atol=atol)
        path = tmp_path / 'evidence'
        save_comparison_artifacts(path, a.trajectory, b.trajectory, report, rtol=rtol, atol=atol)
        return path, report
    return make


@pytest.mark.parametrize('delta', [0, .005, 1])
def test_exact_saved_tolerances_and_result(bundle, capsys, delta):
    path, report = bundle(delta)
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
    assert main(['replay', str(path)]) == (0 if report.passed else 1)
    assert report.summary in capsys.readouterr().out


def test_saved_pass_is_not_trusted(bundle):
    path, _ = bundle(1)
    reportfile = path / 'report.json'
    saved = json.loads(reportfile.read_text())
    saved['passed'] = True
    reportfile.write_text(json.dumps(saved))
    assert not replay_comparison_artifacts(path).passed


def test_changed_trajectory_is_recomputed(bundle):
    path, original = bundle()
    file = path / 'resumed.json'
    data = json.loads(file.read_text())
    data['steps'][1]['loss'] = 100.
    file.write_text(json.dumps(data))
    result = replay_comparison_artifacts(path)
    assert original.passed and not result.passed
    assert result.first_divergence.step == 2


@pytest.mark.parametrize('file', ['manifest.json', 'uninterrupted.json', 'resumed.json', 'report.json'])
def test_incomplete_bundle_fails_closed(bundle, file, capsys):
    path, _ = bundle()
    (path / file).unlink()
    assert main(['replay', str(path)]) == 2
    assert 'could not verify' in capsys.readouterr().err


@pytest.mark.parametrize('field,value', [
    ('schema_version', 2), ('schema_version', True), ('kind', 'other'),
    ('files', ['../secret', 'resumed.json', 'report.json']),
    ('files', ['uninterrupted.json', 'resumed.json']),
    ('comparison', {}), ('comparison', None),
])
def test_manifest_validation(bundle, field, value):
    path, _ = bundle()
    file = path / 'manifest.json'
    manifest = json.loads(file.read_text());manifest[field] = value
    file.write_text(json.dumps(manifest))
    assert main(['replay', str(path)]) == 2


@pytest.mark.parametrize('field,value', [
    ('rtol', '0.01'), ('atol', True), ('rtol', -1), ('atol', float('nan')),
    ('rtol', float('inf')), ('rtol', 10**400), ('atol', -(10**400)), ('check_loss', 1), ('check_batch', False),
    ('check_grad_norm', None), ('allow_partial', True),
])
def test_bad_or_weakened_settings_fail(bundle, field, value):
    path, _ = bundle()
    file = path / 'manifest.json'
    manifest = json.loads(file.read_text());manifest['comparison'][field] = value
    file.write_text(json.dumps(manifest))
    assert main(['replay', str(path)]) == 2


@pytest.mark.parametrize('file', ['manifest.json', 'uninterrupted.json', 'resumed.json', 'report.json'])
@pytest.mark.parametrize('value', ['not json', '[]', 'null'])
def test_malformed_bundle_errors_are_not_tracebacks(bundle, file, value, capsys):
    path, _ = bundle()
    (path / file).write_text(value)
    assert main(['replay', str(path)]) == 2
    assert 'could not verify' in capsys.readouterr().err


def test_replay_empty_trajectories_does_not_certify(bundle):
    path, _ = bundle()
    for name in ['uninterrupted.json', 'resumed.json']:
        file = path / name
        data = json.loads(file.read_text());data['steps'] = []
        file.write_text(json.dumps(data))
    assert main(['replay', str(path)]) == 1


def test_extra_setting_rejected(bundle):
    path, _ = bundle()
    file = path / 'manifest.json'
    manifest = json.loads(file.read_text());manifest['comparison']['new_flag'] = True
    file.write_text(json.dumps(manifest))
    assert main(['replay', str(path)]) == 2


def test_original_compare_command_still_accepts_two_files(bundle, capsys):
    path, report = bundle(.005)
    # Default tight tolerance diverges; replay uses saved looser tolerance.
    assert main(['compare', str(path / 'uninterrupted.json'), str(path / 'resumed.json')]) == 1
    assert main(['replay', str(path)]) == 0
    assert 'PASS' in capsys.readouterr().out
