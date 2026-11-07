"""Comparison scope must use actual booleans, never configuration truthiness."""
import pytest

from resumecheck import StepRecord, Trajectory, compare_trajectories

FLAGS = ('check_loss', 'check_batch', 'check_grad_norm', 'allow_partial', 'check_model_state')


@pytest.mark.parametrize('flag', FLAGS)
@pytest.mark.parametrize('value', ['false', 0, 1, None, [], {}])
def test_nonboolean_scope_is_rejected(flag, value):
    a = Trajectory('a', [StepRecord(0, loss=1.0)])
    b = Trajectory('b', [StepRecord(0, loss=1.0)])
    with pytest.raises(TypeError, match=flag + ' must be a boolean'):
        compare_trajectories(a, b, **{flag: value})


def test_false_string_cannot_enable_partial_success():
    a = Trajectory('a', [StepRecord(0, loss=1.0)])
    b = Trajectory('b', [StepRecord(0, loss=1.0), StepRecord(1, loss=2.0)])
    assert not compare_trajectories(a, b, allow_partial=False).passed
    with pytest.raises(TypeError, match='allow_partial must be a boolean'):
        compare_trajectories(a, b, allow_partial='false')
    scoped = compare_trajectories(a, b, allow_partial=True)
    assert scoped.passed and scoped.steps_compared == 1
    assert any('step count differs' in item for item in scoped.missing_fields)


@pytest.mark.parametrize('flag,field', [('check_loss', 'loss'),
                                       ('check_batch', 'batch_fingerprint'),
                                       ('check_grad_norm', 'grad_norm'),
                                       ('check_model_state', 'model_fingerprint')])
def test_explicit_disable_remains_supported(flag, field):
    a, b = StepRecord(0, loss=1.0), StepRecord(0, loss=1.0)
    # Keep a separate observable so disabling loss still verifies something.
    a.grad_norm = b.grad_norm = 0.5
    left, right = (1.0, 2.0) if field in ('loss', 'grad_norm') else ('v2:left', 'v2:right')
    setattr(a, field, left)
    setattr(b, field, right)
    full, resumed = Trajectory('a', [a]), Trajectory('b', [b])
    assert not compare_trajectories(full, resumed, **{flag: True}).passed
    assert compare_trajectories(full, resumed, **{flag: False}).passed


def test_policy_validation_does_not_call_custom_truthiness():
    class Policy:
        def __bool__(self):
            raise AssertionError('custom truthiness must not run')
    a = Trajectory('a', [StepRecord(0, loss=1.0)])
    with pytest.raises(TypeError, match='allow_partial must be a boolean'):
        compare_trajectories(a, a, allow_partial=Policy())
