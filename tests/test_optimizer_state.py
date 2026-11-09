"""Exact optimizer observations are opt-in and cannot silently lose scope."""
import json
from dataclasses import asdict
import pytest

from resumecheck.recorder import StepRecord, Trajectory, fingerprint_tensor
from resumecheck.compare import compare_trajectories
from resumecheck.artifacts import save_comparison_artifacts, replay_comparison_artifacts
from resumecheck.cli import main

A = fingerprint_tensor({'state': {0: {'moment': [1.0]}}})
B = fingerprint_tensor({'state': {0: {'moment': [2.0]}}})


def trajectory(fp=A, *, extras=False):
    return Trajectory('run', [StepRecord(1, loss=1.0, optimizer_fingerprint=fp,
                                       extras={'lr': 0.0} if extras else {})])


def test_optimizer_only_exact_comparison():
    a = trajectory(); a.steps[0].loss = None
    b = trajectory(B); b.steps[0].loss = None
    assert compare_trajectories(a, a).passed
    report = compare_trajectories(a, b, rtol=1e9, atol=1e9)
    assert not report.passed
    assert report.first_divergence.field == 'optimizer_state'
    assert report.first_divergence.step == 1
    assert not compare_trajectories(a, b, check_optimizer_state=False).passed
    a.steps[0].loss = b.steps[0].loss = 1.0
    assert compare_trajectories(a, b, check_optimizer_state=False).passed


@pytest.mark.parametrize('left,right', [(A, None), (None, A), (None, None)])
def test_missing_optimizer_evidence_fails(left, right):
    a, b = trajectory(left), trajectory(right)
    # Establish optimizer scope at another step, then omit one/both sides.
    a.steps.append(StepRecord(2, loss=1.0, optimizer_fingerprint=A))
    b.steps.append(StepRecord(2, loss=1.0, optimizer_fingerprint=A))
    report = compare_trajectories(a, b, allow_partial=True)
    assert not report.passed and report.missing_fields


@pytest.mark.parametrize('value', ['v2:', 'v2:abc', 'v1:1234567890abcdef',
                                  'v2:1234567890abcdeg', 1, True, [], {}])
def test_malformed_optimizer_fingerprint_never_passes(value):
    assert not compare_trajectories(trajectory(value), trajectory(value)).passed


@pytest.mark.parametrize('value', [0, 1, None, 'false', []])
def test_scope_flag_is_boolean(value):
    with pytest.raises(TypeError, match='check_optimizer_state'):
        compare_trajectories(trajectory(), trajectory(), check_optimizer_state=value)


@pytest.mark.parametrize('extras', [False, True])
@pytest.mark.parametrize('model', [False, True])
@pytest.mark.parametrize('different', [False, True])
def test_schema4_roundtrip_and_cli(tmp_path, capsys, extras, model, different):
    a, b = trajectory(extras=extras), trajectory(B if different else A, extras=extras)
    if model:
        a.steps[0].model_fingerprint = b.steps[0].model_fingerprint = A
    names = ['lr'] if extras else None
    report = compare_trajectories(a, b, check_extras=names)
    path = tmp_path / 'evidence'
    save_comparison_artifacts(path, a, b, report, rtol=1e-5, atol=1e-7, check_extras=names)
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['schema_version'] == 4
    assert manifest['comparison']['check_optimizer_state'] is True
    assert manifest['comparison']['check_model_state'] == model
    assert asdict(replay_comparison_artifacts(path)) == asdict(report)
    assert main(['replay', str(path), '--json']) == (1 if different else 0)
    assert json.loads(capsys.readouterr().out)['passed'] == (not different)
    assert main(['compare', str(path/'uninterrupted.json'), str(path/'resumed.json')]) == (1 if different else 0)


@pytest.mark.parametrize('change', ['scope_false', 'scope_int', 'remove_scope', 'downgrade',
                                   'both_removed', 'one_removed', 'settings_extra'])
def test_schema4_scope_tampering_never_passes(tmp_path, change):
    a = trajectory()
    path = tmp_path / 'evidence'
    save_comparison_artifacts(path, a, a, compare_trajectories(a, a), rtol=1e-5, atol=1e-7)
    file = path/'manifest.json'; manifest = json.loads(file.read_text())
    if change == 'scope_false': manifest['comparison']['check_optimizer_state'] = False
    if change == 'scope_int': manifest['comparison']['check_optimizer_state'] = 1
    if change == 'remove_scope': del manifest['comparison']['check_optimizer_state']
    if change == 'settings_extra': manifest['comparison']['new_flag'] = True
    if change == 'downgrade':
        manifest['schema_version'] = 1
        for key in ('check_optimizer_state', 'check_model_state', 'check_extras'):
            del manifest['comparison'][key]
    file.write_text(json.dumps(manifest))
    if change in ('one_removed', 'both_removed'):
        for name in (['resumed'] if change == 'one_removed' else ['resumed','uninterrupted']):
            file = path/(name+'.json'); data = json.loads(file.read_text())
            del data['steps'][0]['optimizer_fingerprint']; file.write_text(json.dumps(data))
    if change == 'one_removed':
        assert not replay_comparison_artifacts(path).passed
    else:
        with pytest.raises(ValueError): replay_comparison_artifacts(path)


def test_saved_pass_not_trusted_and_weak_report_not_publishable(tmp_path):
    a, b = trajectory(), trajectory(B)
    weak = compare_trajectories(a, b, check_optimizer_state=False)
    assert weak.passed
    with pytest.raises(ValueError, match='scope'):
        save_comparison_artifacts(tmp_path/'bad', a, b, weak, rtol=1e-5, atol=1e-7)
    assert not (tmp_path/'bad').exists()
    path = tmp_path/'good'
    save_comparison_artifacts(path, a, b, compare_trajectories(a,b), rtol=1e-5, atol=1e-7)
    file=path/'report.json'; data=json.loads(file.read_text()); data['passed']=True
    file.write_text(json.dumps(data))
    assert not replay_comparison_artifacts(path).passed


def test_default_json_and_schema_unchanged(tmp_path):
    a=trajectory(None)
    a.save(tmp_path/'trajectory.json')
    assert 'optimizer_fingerprint' not in json.loads((tmp_path/'trajectory.json').read_text())['steps'][0]
    save_comparison_artifacts(tmp_path/'evidence', a, a, compare_trajectories(a,a), rtol=1e-5, atol=1e-7)
    assert json.loads((tmp_path/'evidence'/'manifest.json').read_text())['schema_version'] == 1


def test_existing_record_shape_must_not_ignore_optimizer_observation():
    # Baseline-compatible construction: the old comparator silently ignores
    # these supplied observations, so this is an assertion negative control,
    # not an expected exception from the new constructor or scope flag.
    a=Trajectory('full',[StepRecord(1,loss=1.0)])
    b=Trajectory('resumed',[StepRecord(1,loss=1.0)])
    a.steps[0].optimizer_fingerprint=A
    b.steps[0].optimizer_fingerprint=B
    report=compare_trajectories(a,b)
    assert not report.passed
    assert report.first_divergence.field=='optimizer_state'


def test_malformed_objects_are_not_compared_with_user_equality():
    class EqualityTrap:
        def __ne__(self,other): raise AssertionError('invalid evidence must not be compared')
    assert not compare_trajectories(trajectory(EqualityTrap()),trajectory(EqualityTrap())).passed
