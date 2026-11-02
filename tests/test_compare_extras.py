"""Opt-in recorded extras catch rate drift without changing default scope."""
import json
import pytest
from resumecheck import Trajectory, compare_trajectories
from resumecheck.recorder import StepRecord, TrajectoryRecorder
from resumecheck.cli import main


def _run(name, values, losses=True):
    return Trajectory(name, [StepRecord(i, loss=1.0 if losses else None, extras=value)
                             for i, value in enumerate(values)])


def test_default_ignores_extras_but_opt_in_finds_earliest_rate_drift():
    a = _run('a', [{'lr': .1}, {'lr': .01}, {'lr': .001}])
    b = _run('b', [{'lr': .1}, {'lr': .1}, {'lr': .1}])
    assert compare_trajectories(a, b).passed
    r = compare_trajectories(a, b, check_extras=['lr'])
    assert not r.passed
    assert (r.first_divergence.step, r.first_divergence.field) == (1, 'extras.lr')
    assert (r.first_divergence.value_a, r.first_divergence.value_b) == (.01, .1)


def test_matching_extras_alone_are_a_scoped_observable():
    a = _run('a', [{'lr': .1}, {'lr': .01}], losses=False)
    r = compare_trajectories(a, a, check_loss=False, check_batch=False,
                             check_grad_norm=False, check_model_state=False,
                             check_extras=['lr', 'lr'])
    assert r.passed and r.steps_compared == 2
    assert r.summary.count("'lr'") == 1
    assert 'compared extras' in r.summary


@pytest.mark.parametrize('bad', [None, True, '0.1', float('nan'), float('inf'), 10**400, [], {}])
def test_invalid_selected_extra_fails_closed_with_other_matching_loss(bad):
    a = _run('a', [{'lr': bad}])
    r = compare_trajectories(a, a, check_extras=['lr'])
    assert not r.passed
    assert any('finite number' in detail for detail in r.missing_fields)


@pytest.mark.parametrize('left,right', [({}, {}), ({'lr': .1}, {}), ({}, {'lr': .1}), (None, {'lr': .1})])
def test_missing_selected_extra_on_either_side_is_not_ignored(left, right):
    r = compare_trajectories(_run('a', [left]), _run('b', [right]), check_extras=['lr'])
    assert not r.passed and any('missing' in detail for detail in r.missing_fields)


def test_partial_comparison_does_not_excuse_missing_selected_evidence():
    a = _run('a', [{'lr': .1}, {}])
    b = _run('b', [{'lr': .1}])
    assert compare_trajectories(a, b, allow_partial=True, check_extras=['lr']).passed
    b.steps[0].extras = {}
    assert not compare_trajectories(a, b, allow_partial=True, check_extras=['lr']).passed


def test_named_extras_use_numeric_tolerance_and_ignore_unselected_fields():
    a = _run('a', [{'lr': .1, 'invalid': []}])
    b = _run('b', [{'lr': .10000001, 'invalid': None}])
    assert compare_trajectories(a, b, check_extras=['lr']).passed
    assert not compare_trajectories(a, b, check_extras=['lr'], rtol=0, atol=0).passed


@pytest.mark.parametrize('names', ['lr', b'lr', [''], [None], [1]])
def test_bad_field_selection_is_explicit_input_error(names):
    with pytest.raises((TypeError, ValueError)):
        compare_trajectories(_run('a', [{'lr': .1}]), _run('b', [{'lr': .1}]), check_extras=names)


def test_multiple_named_extras_keep_requested_order_without_mutation():
    a = _run('a', [{'lr': .1, 'scale': 1}])
    b = _run('b', [{'lr': .2, 'scale': 2}])
    names = ['scale', 'lr']
    r = compare_trajectories(a, b, check_extras=names)
    assert r.first_divergence.field == 'extras.scale'
    assert names == ['scale', 'lr'] and a.steps[0].extras == {'lr': .1, 'scale': 1}


def test_cli_roundtripped_recorder_extras_default_and_selected(tmp_path, capsys):
    paths = []
    for name, lr in [('a', .1), ('b', .2)]:
        rec = TrajectoryRecorder(name)
        rec.record(0, loss=1., lr=lr)
        path = tmp_path / (name + '.json')
        rec.trajectory.save(path)
        paths.append(str(path))
    assert main(['compare', *paths]) == 0
    capsys.readouterr()
    assert main(['compare', *paths, '--extra', 'lr']) == 1
    out = capsys.readouterr().out
    assert 'extras.lr' in out
    assert 'DIVERGED at step 0' in out


def test_cli_missing_selected_field_fails_not_parse_error(tmp_path, capsys):
    path = tmp_path / 'run.json'
    _run('x', [{'lr': .1}]).save(path)
    assert main(['compare', str(path), str(path), '--extra', 'unknown']) == 1
    assert 'missing' in capsys.readouterr().out


def test_actual_scheduler_state_control_detects_rate_before_loss_moves(tmp_path):
    import torch
    from copy import deepcopy

    def setup():
        weight = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([weight], lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.1)
        return weight, optimizer, scheduler

    def step(weight, optimizer, scheduler, recorder, index):
        # Zero gradient leaves the model and loss unchanged. The LR is still
        # scheduled and is the only observable that diagnoses the lost state.
        optimizer.zero_grad()
        loss = weight * 0 + 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        recorder.record(index, loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'])

    weight, optimizer, scheduler = setup()
    baseline = TrajectoryRecorder('baseline')
    step(weight, optimizer, scheduler, baseline, 0)
    checkpoint = deepcopy((optimizer.state_dict(), scheduler.state_dict()))
    step(weight, optimizer, scheduler, baseline, 1)

    for restore_scheduler in [False, True]:
        weight2, optimizer2, scheduler2 = setup()
        optimizer2.load_state_dict(checkpoint[0])
        if restore_scheduler:
            scheduler2.load_state_dict(checkpoint[1])
        resumed = TrajectoryRecorder('resumed')
        resumed.trajectory.steps = deepcopy(baseline.trajectory.steps[:1])
        step(weight2, optimizer2, scheduler2, resumed, 1)
        assert compare_trajectories(baseline.trajectory, resumed.trajectory).passed
        r = compare_trajectories(baseline.trajectory, resumed.trajectory, check_extras=['lr'])
        assert r.passed is restore_scheduler
        if not restore_scheduler:
            assert (r.first_divergence.step, r.first_divergence.field) == (1, 'extras.lr')


@pytest.mark.parametrize('left,right', [
    (9007199254740993, 9007199254740992.0),
    (9007199254740992.0, 9007199254740993),
    (10**308, -(10**308)),
    (-(10**308), 10**308),
])
def test_selected_integer_precision_and_finite_pair_overflow(left, right):
    a = _run('a', [{'counter': left}])
    b = _run('b', [{'counter': right}])
    r = compare_trajectories(a, b, check_extras=['counter'], rtol=0, atol=0)
    assert not r.passed
    assert r.first_divergence.field == 'extras.counter'
    assert (r.first_divergence.value_a, r.first_divergence.value_b) == (left, right)


def test_selected_integer_tolerance_uses_exact_represented_values():
    a = _run('a', [{'counter': 9007199254740993}])
    b = _run('b', [{'counter': 9007199254740992.0}])
    assert compare_trajectories(a, b, check_extras=['counter'], rtol=0, atol=1).passed
    assert not compare_trajectories(a, b, check_extras=['counter'], rtol=0, atol=.5).passed
