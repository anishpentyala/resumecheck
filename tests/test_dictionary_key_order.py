import pytest

from resumecheck.recorder import fingerprint_tensor


@pytest.mark.parametrize('keys', [(1, '1'), (None, 'None'), (True, 'True'),
                                 ((1, 2), '(1, 2)'), (2.5, '2.5')])
def test_equal_dictionaries_have_same_fingerprint_regardless_of_insertion(keys):
    left = {keys[0]: [1, 2], keys[1]: [3, 4]}
    right = dict(reversed(list(left.items())))
    assert left == right
    assert fingerprint_tensor(left) == fingerprint_tensor(right)


def test_key_types_and_values_still_distinct():
    assert fingerprint_tensor({1: 'a', '1': 'b'}) != fingerprint_tensor({1: 'b', '1': 'a'})


def test_existing_string_key_fingerprint_stays_compatible():
    assert fingerprint_tensor({'a': 1, 'b': 2}) == 'v2:dff8d803c7635ee1'


@pytest.mark.parametrize('keys', [(1, '1'), (None, 'None'), (True, 'True')])
def test_nested_and_recorded_batches_do_not_falsely_diverge(keys):
    from resumecheck.recorder import TrajectoryRecorder
    from resumecheck.compare import compare_trajectories
    first = {keys[0]: [1, 2], keys[1]: [3, 4]}
    second = dict(reversed(list(first.items())))
    a, b = TrajectoryRecorder('a'), TrajectoryRecorder('b')
    a.record(0, batch={'batch': [first]})
    b.record(0, batch={'batch': [second]})
    result = compare_trajectories(a.trajectory, b.trajectory,
                                  check_loss=False, check_grad_norm=False)
    assert result.passed, result.summary


def test_string_key_sort_order_not_changed_by_typed_tiebreak():
    assert fingerprint_tensor({'10': 0, '2': 1, 'a': 2}) == fingerprint_tensor({'a': 2, '2': 1, '10': 0})


def test_unsupported_key_is_still_rejected():
    class Key:
        def __str__(self):
            return '1'
    with pytest.raises(TypeError, match='unsupported batch value'):
        fingerprint_tensor({1: 0, Key(): 1})


def test_recorded_tied_keys_survive_trajectory_save_and_load(tmp_path):
    from resumecheck.recorder import TrajectoryRecorder, Trajectory
    from resumecheck.compare import compare_trajectories
    a, b = TrajectoryRecorder('a'), TrajectoryRecorder('b')
    a.record(0, batch={1: 'number', '1': 'text'})
    b.record(0, batch={'1': 'text', 1: 'number'})
    a.trajectory.save(tmp_path / 'a.json')
    b.trajectory.save(tmp_path / 'b.json')
    assert compare_trajectories(Trajectory.load(tmp_path / 'a.json'),
                                Trajectory.load(tmp_path / 'b.json'),
                                check_loss=False, check_grad_norm=False).passed
