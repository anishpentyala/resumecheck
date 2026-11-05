"""Machine-readable output must retain failure semantics and strict JSON."""
import json
import subprocess
import sys

import pytest
from resumecheck.cli import main
from resumecheck.recorder import TrajectoryRecorder


def write(tmp_path, name, loss=1., batch=(1, 2)):
    recorder = TrajectoryRecorder(name)
    recorder.record(0, loss=loss, batch=list(batch))
    path = tmp_path / (name + '.json')
    recorder.trajectory.save(path)
    return str(path)


def strict_load(text):
    def reject(value):
        raise AssertionError('nonstandard JSON number: ' + value)
    return json.loads(text, parse_constant=reject)


def test_json_success_one_document(tmp_path, capsys):
    a, b = write(tmp_path, 'a'), write(tmp_path, 'b')
    assert main(['compare', a, b, '--json']) == 0
    captured = capsys.readouterr()
    result = strict_load(captured.out)
    assert captured.err == ''
    assert result['schema_version'] == 1
    assert result['status'] == 'pass' and result['passed']
    assert result['error'] is None
    assert result['report']['steps_compared'] == 1
    assert result['report']['first_divergence'] is None
    assert result['report']['missing_fields'] == []
    assert result['report']['summary'].startswith('PASS:')



@pytest.mark.parametrize('loss,label', [(float('inf'), 'Infinity'),
                                       (-float('inf'), '-Infinity'),
                                       (float('nan'), 'NaN')])
def test_nonfinite_divergence_is_string_not_json_number(tmp_path, capsys, loss, label):
    a, b = write(tmp_path, 'a', loss), write(tmp_path, 'b', 1.)
    assert main(['compare', a, b, '--json']) == 1
    result = strict_load(capsys.readouterr().out)
    assert result['status'] == 'fail' and not result['passed']
    assert result['report']['first_divergence']['field'] == 'loss'
    assert result['report']['first_divergence']['value_a'] == label
    assert result['report']['first_divergence']['value_b'] == 1.


def test_json_batch_failure_preserves_detail(tmp_path, capsys):
    a, b = write(tmp_path, 'a'), write(tmp_path, 'b', batch=(2, 3))
    assert main(['compare', a, b, '--json']) == 1
    result = strict_load(capsys.readouterr().out)
    assert result['report']['first_divergence']['field'] == 'batch'
    assert result['report']['steps_compared'] == 0


def test_json_invalid_input_never_passes(tmp_path, capsys):
    assert main(['compare', str(tmp_path/'absent'), str(tmp_path/'absent'), '--json']) == 2
    captured = capsys.readouterr()
    result = strict_load(captured.out)
    assert result['status'] == 'error' and not result['passed']
    assert result['report'] is None and result['error']
    assert 'could not verify' in captured.err


def test_json_invalid_tolerance_is_error(tmp_path, capsys):
    a, b = write(tmp_path, 'a'), write(tmp_path, 'b')
    assert main(['compare', a, b, '--rtol', 'nan', '--json']) == 2
    assert strict_load(capsys.readouterr().out)['status'] == 'error'


def test_replay_json_missing_manifest_error(tmp_path, capsys):
    assert main(['replay', str(tmp_path), '--json']) == 2
    assert strict_load(capsys.readouterr().out)['status'] == 'error'


def test_cli_subprocess_json(tmp_path):
    a, b = write(tmp_path, 'a'), write(tmp_path, 'b', batch=(4, 5))
    process = subprocess.run([sys.executable, '-m', 'resumecheck.cli',
                              'compare', a, b, '--json'], capture_output=True, text=True)
    assert process.returncode == 1
    assert strict_load(process.stdout)['status'] == 'fail'
    assert process.stderr == ''


@pytest.mark.parametrize('different', [False, True])
def test_replay_json_recomputes_saved_evidence(tmp_path, capsys, different):
    from resumecheck.artifacts import save_comparison_artifacts
    from resumecheck.compare import compare_trajectories
    from resumecheck.recorder import Trajectory
    a, b = write(tmp_path, 'a'), write(tmp_path, 'b', batch=(4, 5) if different else (1, 2))
    full, resumed = Trajectory.load(a), Trajectory.load(b)
    report = compare_trajectories(full, resumed)
    directory = tmp_path/'evidence'
    save_comparison_artifacts(directory, full, resumed, report, rtol=1e-5, atol=1e-7)
    (directory/'report.json').write_text('{"passed": true}')
    assert main(['replay', str(directory), '--json']) == int(different)
    payload = strict_load(capsys.readouterr().out)
    assert payload['passed'] is not different
    assert payload['status'] == ('fail' if different else 'pass')
