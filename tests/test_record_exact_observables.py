"""Recorder must not round away a divergence before comparison."""
import pytest
from resumecheck import TrajectoryRecorder, compare_trajectories
from resumecheck.recorder import Trajectory


@pytest.mark.parametrize('field', ['loss', 'grad_norm', 'counter'])
@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('roundtrip', [False, True])
def test_large_integer_difference_reaches_comparison(field, sign, roundtrip, tmp_path):
    left, right = TrajectoryRecorder('left'), TrajectoryRecorder('right')
    left.record(0, **{field: sign * 2**54})
    right.record(0, **{field: sign * (2**54 + 1)})
    a, b = left.trajectory, right.trajectory
    if roundtrip:
        a.save(tmp_path / 'a.json')
        b.save(tmp_path / 'b.json')
        a, b = Trajectory.load(tmp_path / 'a.json'), Trajectory.load(tmp_path / 'b.json')
    report = compare_trajectories(a, b, rtol=0, atol=0,
                                 check_extras=['counter'] if field == 'counter' else None)
    assert not report.passed
    assert report.first_divergence.field == ('extras.counter' if field == 'counter' else field)


@pytest.mark.parametrize('field', ['loss', 'grad_norm', 'counter'])
def test_bool_is_not_numeric_evidence_and_failure_is_atomic(field):
    recorder = TrajectoryRecorder()
    recorder.record(0, loss=1)
    with pytest.raises(TypeError, match='must not be bool'):
        recorder.record(1, **{field: True})
    assert len(recorder.trajectory.steps) == 1


@pytest.mark.parametrize('field', ['loss', 'grad_norm', 'counter'])
def test_equal_integer_and_float_compatible_observations(field):
    a, b = TrajectoryRecorder(), TrajectoryRecorder()
    a.record(0, **{field: 17})
    b.record(0, **{field: 17.0})
    report = compare_trajectories(a.trajectory, b.trajectory,
                                 check_extras=['counter'] if field == 'counter' else None)
    assert report.passed


@pytest.mark.parametrize('field', ['loss', 'grad_norm', 'counter'])
def test_integer_index_scalar_preserves_exact_value(field):
    class IndexScalar:
        def __index__(self):
            return 2**54 + 1
    recorder = TrajectoryRecorder()
    recorder.record(0, **{field: IndexScalar()})
    step = recorder.trajectory.steps[0]
    value = step.extras[field] if field == 'counter' else getattr(step, field)
    assert type(value) is int
    assert value == 2**54 + 1


@pytest.mark.parametrize('value', [0.125, '0.125'])
@pytest.mark.parametrize('field', ['loss', 'grad_norm', 'counter'])
def test_existing_float_conversion_kept(value, field):
    recorder = TrajectoryRecorder()
    recorder.record(0, **{field: value})
    step = recorder.trajectory.steps[0]
    stored = step.extras[field] if field == 'counter' else getattr(step, field)
    assert type(stored) is float
    assert stored == 0.125
