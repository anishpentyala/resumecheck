import json

import pytest

from resumecheck.cli import main
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import StepRecord, Trajectory, TrajectoryRecorder


@pytest.mark.parametrize("value", [True, False, 0.5, 1.0, "1", None, float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("side", ["both", "uninterrupted", "resumed"])
@pytest.mark.parametrize("position", [0, 1])
def test_invalid_step_indices_cannot_verify(value, side, position):
    a = Trajectory("a", [StepRecord(0, loss=1.0), StepRecord(2, loss=0.5)])
    b = Trajectory("b", [StepRecord(0, loss=1.0), StepRecord(2, loss=0.5)])
    if side in ("both", "uninterrupted"):
        a.steps[position].step = value
    if side in ("both", "resumed"):
        b.steps[position].step = value
    report = compare_trajectories(a, b)
    assert not report.passed
    assert report.steps_compared == 0
    assert "step must be an integer" in report.summary


@pytest.mark.parametrize("value", [True, False, 0.5, 1.0, "1", None, float("inf"), float("-inf"), float("nan")])
def test_cli_loaded_invalid_step_indices_fail(value, tmp_path, capsys):
    paths = []
    for name in ["a", "b"]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name, "steps": [{"step": value, "loss": 1.0}]}))
        paths.append(str(path))
    assert main(["compare", *paths]) == 1
    text = capsys.readouterr().out
    assert "step must be an integer" in text
    assert "PASS" not in text


@pytest.mark.parametrize("values", [[0], [-3, -1, 2], [1, 3, 10], [10**1000, 10**1000 + 1]])
def test_valid_integer_step_indices_still_verify(values):
    a = Trajectory("a", [StepRecord(value, loss=1.0) for value in values])
    b = Trajectory("b", [StepRecord(value, loss=1.0) for value in values])
    assert compare_trajectories(a, b).passed


@pytest.mark.parametrize("allow_partial", [False, True])
def test_partial_mode_does_not_allow_invalid_indices(allow_partial):
    a = Trajectory("a", [StepRecord(1.0, loss=1.0)])
    b = Trajectory("b", [StepRecord(1.0, loss=1.0), StepRecord(2, loss=0.5)])
    report = compare_trajectories(a, b, allow_partial=allow_partial)
    assert not report.passed
    assert "step must be an integer" in report.summary


def test_index_protocol_matches_recorder():
    class Index:
        def __index__(self):
            return 3

    rec = TrajectoryRecorder()
    rec.record(Index(), loss=1.0)
    a = Trajectory("a", [StepRecord(Index(), loss=1.0)])
    # The comparator should accept the same index protocol as the recorder,
    # without trusting the custom object's equality/ordering implementation.
    assert compare_trajectories(a, rec.trajectory).passed


@pytest.mark.parametrize("side", ["uninterrupted", "resumed"])
def test_invalid_uncompared_tail_is_not_ignored(side):
    a = Trajectory("a", [StepRecord(1, loss=1.0)])
    b = Trajectory("b", [StepRecord(1, loss=1.0)])
    (a if side == "uninterrupted" else b).steps.append(StepRecord(2.5, loss=0.5))
    report = compare_trajectories(a, b, allow_partial=True)
    assert not report.passed
    assert "step must be an integer" in report.summary


def test_normalizing_index_protocol_does_not_modify_caller_records():
    class Index:
        def __index__(self):
            return 3

        def __eq__(self, other):
            raise AssertionError("do not compare an unnormalized index")

    value = Index()
    record = StepRecord(value, loss=1.0, extras={"lr": 0.1})
    a = Trajectory("a", [record])
    b = Trajectory("b", [StepRecord(3, loss=1.0)])
    assert compare_trajectories(a, b).passed
    assert a.steps[0] is record
    assert record.step is value
    assert record.extras == {"lr": 0.1}


def test_numpy_integer_indices_match_builtin_integers():
    np = pytest.importorskip("numpy")
    a = Trajectory("a", [StepRecord(np.int64(1), loss=1.0), StepRecord(np.int32(3), loss=0.5)])
    b = Trajectory("b", [StepRecord(1, loss=1.0), StepRecord(3, loss=0.5)])
    assert compare_trajectories(a, b).passed


@pytest.mark.parametrize("steps", [[3, 3], [4, 3], [0, -1]])
def test_valid_integer_protocol_still_checks_ordering(steps):
    class Index:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    a = Trajectory("a", [StepRecord(Index(step), loss=1.0) for step in steps])
    b = Trajectory("b", [StepRecord(step, loss=1.0) for step in steps])
    report = compare_trajectories(a, b)
    assert not report.passed
    assert "strictly increasing" in report.summary


def test_large_misaligned_indices_preserve_exact_diagnostic_values():
    step = 10**1000
    a = Trajectory("a", [StepRecord(step, loss=1.0)])
    b = Trajectory("b", [StepRecord(step + 1, loss=1.0)])
    report = compare_trajectories(a, b)
    assert not report.passed
    assert report.first_divergence.field == "step"
    assert report.first_divergence.value_a == step
    assert report.first_divergence.value_b == step + 1
