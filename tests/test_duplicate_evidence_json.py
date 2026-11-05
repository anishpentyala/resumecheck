"""A verification input cannot choose the last conflicting JSON member."""
import json

import pytest

from resumecheck.artifacts import save_comparison_artifacts, replay_comparison_artifacts
from resumecheck.cli import main
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import Trajectory, StepRecord


@pytest.mark.parametrize('raw', [
    '{"name":"r","steps":[{"step":0,"loss":9,"loss":1}]}',
    '{"name":"r","steps":[{"step":9,"step":0,"loss":1}]}',
    '{"name":"r","steps":[{"step":0,"loss":1,"extras":{"lr":9,"lr":1}}]}',
    '{"name":"r","steps":[{"step":0,"loss":1}],"steps":[{"step":0,"loss":1}]}',
    '{"name":"r","steps":[{"step":0,"loss":9,"lo\\u0073s":1}]}',
    '{"name":"r","steps":[{"step":0,"loss":1}],"meta":{"private-marker":1,"private-marker":2}}',
])
def test_ambiguous_trajectory_never_reaches_comparison(tmp_path, raw):
    path = tmp_path / 'trajectory.json'; path.write_text(raw)
    with pytest.raises(ValueError, match='duplicate JSON object member') as caught:
        Trajectory.load(path)
    assert 'private-marker' not in str(caught.value)


def test_cli_hidden_divergence_is_error_not_pass(tmp_path, capsys):
    a = tmp_path / 'a.json'; b = tmp_path / 'b.json'
    Trajectory('a', [StepRecord(0, loss=1)]).save(a)
    b.write_text('{"name":"b","steps":[{"step":0,"loss":100,"loss":1}]}')
    assert main(['compare', str(a), str(b), '--json']) == 2
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload['status'] == 'error' and payload['passed'] is False
    assert payload['report'] is None
    assert 'duplicate JSON object member' in out.err


@pytest.mark.parametrize('file,before,after', [
    ('manifest.json', '"schema_version": 1', '"schema_version": 99, "schema_version": 1'),
    ('manifest.json', '"rtol": 1e-05', '"rtol": 0, "rtol": 1e-05'),
    ('manifest.json', '"check_batch": true', '"check_batch": false, "check_batch": true'),
    ('report.json', '"passed": true', '"passed": false, "passed": true'),
    ('resumed.json', '"loss": 1', '"loss": 99, "loss": 1'),
])
def test_bundle_ambiguous_members_are_explicit_input_error(tmp_path, file, before, after):
    trajectory = Trajectory('r', [StepRecord(0, loss=1)])
    report = compare_trajectories(trajectory, trajectory)
    directory = tmp_path / 'evidence'
    save_comparison_artifacts(directory, trajectory, trajectory, report, rtol=1e-5, atol=1e-7)
    path = directory / file
    text = path.read_text(); assert before in text
    path.write_text(text.replace(before, after, 1))
    with pytest.raises(ValueError, match='duplicate JSON object member'):
        replay_comparison_artifacts(directory)


def test_repeated_names_in_distinct_steps_remain_valid(tmp_path):
    trajectory = Trajectory('r', [StepRecord(0, loss=1), StepRecord(1, loss=2)])
    path = tmp_path / 'trajectory.json'; trajectory.save(path)
    loaded = Trajectory.load(path)
    assert compare_trajectories(trajectory, loaded).passed
